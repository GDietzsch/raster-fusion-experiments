import pytest
import torch
from src.services import enriched_unet as dense
from src.image_pipeline.models.unet import UNet


def test_added_arms_are_exact_channel_unions():
    idx = lambda arm: set(dense.common._arm_indices(arm))
    assert idx('originais_gpa') == idx('originais') | idx('gpa')
    assert idx('originais_grafos') == idx('originais') | idx('grafos')
    assert len(idx('originais_gpa')) == 16
    assert len(idx('originais_grafos')) == 12
    assert len(dense._arms({'extended_arms': True})) == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires GPU')
def test_full_size_training_and_halo():
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = UNet(24, dense._model_config(16, block_type='inception')).cuda().train()
    inputs = torch.randn(1, 24, 960, 960, device='cuda')
    prediction = model(inputs)
    prediction[:, :, 224:736, 224:736].square().mean().backward()
    assert prediction.shape == (1, 1, 960, 960)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('INCEPTION full-size backward peak GiB:', torch.cuda.max_memory_allocated() / 2**30)
    del model, inputs, prediction
    torch.cuda.empty_cache()
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        model = UNet(3, dense._model_config(8, block_type='inception')).cuda().eval()
        raster = torch.randn(1, 3, 960, 704, device='cuda')
        with torch.inference_mode():
            full = model(raster)[0, 0, 224:736, 224:480]
            tiled = torch.cat([model(raster[:, :, row:row+704])[0, 0, 224:480, 224:480]
                               for row in (0, 256)], dim=0)
        difference = torch.max(torch.abs(full - tiled)).item()
        print('INCEPTION seam maximum difference:', difference)
        assert difference < 1e-4
    finally:
        torch.backends.cudnn.allow_tf32 = previous
