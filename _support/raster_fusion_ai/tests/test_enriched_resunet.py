"""Residual paths, artifact compatibility, full-size GPU and execution guards."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.services import enriched_unet as u
from src.services import enriched_resunet as r


def test_zero_projection_preserves_plain_unet_and_shortcuts_receive_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    plain = u.UNet(3, u._model_config(2, 2)).eval()
    residual = u.UNet(3, u._model_config(2, 2, True)).eval()
    residual.load_state_dict(plain.state_dict(), strict=False)
    for block in list(residual.encoders) + list(residual.decoders):
        torch.nn.init.zeros_(block.shortcut.weight)
        torch.nn.init.zeros_(block.shortcut.bias)
    inputs = torch.randn(1, 3, 64, 64)
    torch.testing.assert_close(plain(inputs), residual(inputs), rtol=0, atol=0)
    residual(inputs).square().mean().backward()
    assert sum(float(b.shortcut.weight.grad.abs().sum()) for b in residual.encoders) > 0
    params = {'initial_filters': 2, 'model_depth': 2, 'residual_blocks': True}
    scale = {'input_mean': np.zeros(3), 'input_scale': np.ones(3), 'target_mean': 0., 'target_scale': 1.}
    artifact = u._artifact(residual, (0, 1, 2), scale, params, 1)
    restored, _, _ = u._restore(artifact, torch.device('cpu'))
    torch.testing.assert_close(restored(inputs), residual(inputs))
    artifact['architecture'] = 'unet'
    with pytest.raises(ValueError, match='disagree'):
        u._restore(artifact, torch.device('cpu'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires full-sized GPU')
def test_full_size_resunet_gpu_backward_and_seam():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = u.UNet(24, u._model_config(16, 5, True)).cuda()
    x = torch.randn(1, 24, 960, 960, device='cuda')
    y = model(x)
    assert y.shape == (1, 1, 960, 960)
    y[:, :, 224:736, 224:736].square().mean().backward()
    assert torch.isfinite(model.encoders[0].shortcut.weight.grad).all()
    print('ResUNet GPU peak allocated GiB:', torch.cuda.max_memory_allocated() / 2**30)
    del model, x, y
    torch.cuda.empty_cache()
    model = u.UNet(3, u._model_config(8, 5, True)).cuda().eval()
    x = torch.randn(1, 3, 960, 704, device='cuda')
    # Different convolution shapes may use different TF32 kernels. Disable TF32
    # here to test geometric halo sufficiency rather than reduced precision.
    with torch.inference_mode(), torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
        full = model(x)[0, 0, 224:736, 224:480]
        tiled = torch.cat([model(x[:, :, row:row+704])[0, 0, 224:480, 224:480]
                           for row in (0, 256)])
    torch.testing.assert_close(full, tiled, rtol=0, atol=1e-4)


def test_resume_checks_source_and_completed_run_does_not_retrain(tmp_path, monkeypatch):
    config = tmp_path / 'cfg.yaml'
    config.write_text('same')
    cfg = {'kind': 'enriched_resunet', 'output': tmp_path / 'runs', 'path': config,
           'raw': {'runtime': {'prevent_sleep': False}}}
    folder = cfg['output'] / 'run'
    folder.mkdir(parents=True)
    (folder / 'config.yaml').write_text('same')
    (folder / 'run.json').write_text(json.dumps({'kind': cfg['kind'], 'status': 'failed'}))
    (folder / 'source_files.json').write_text('[]')
    monkeypatch.setattr(r, 'load_config', lambda _: cfg)
    calls = []
    monkeypatch.setattr(r.dense, 'run', lambda path, resume: calls.append(resume))
    r.run(config)
    assert calls == [folder]
    (folder / 'run.json').write_text(json.dumps({'kind': cfg['kind'], 'status': 'completed'}))
    r.run(config)
    assert calls == [folder]
    (folder / 'run.json').write_text(json.dumps({'kind': cfg['kind'], 'status': 'failed'}))
    (folder / 'source_files.json').write_text(json.dumps([{'path': 'src/services/enriched_unet.py', 'sha256': 'wrong'}]))
    with pytest.raises(ValueError, match='Source changed'):
        r.run(config)
