"""Experiment 11 entry points; shared dense engine with explicit residual artifacts."""
from src.utils.source_paths import project_root, source_path
from pathlib import Path
import json
from src.services import enriched_unet as dense
from src.services.enriched_unet_controls import exclusive_lock, awake


def load_config(path: Path) -> dict:
    cfg = dense.load_config(path)
    if cfg['kind'] != 'enriched_resunet':
        raise ValueError('ResUNet entry point requires model_family: resunet')
    if cfg['output'] == (project_root() / 'runs/enriched_unet'):
        raise ValueError('ResUNet must not write in the original U-Net output')
    return cfg


def check_only(path: Path) -> None:
    cfg = load_config(path)
    dense.check_only(path)
    print('RESUNET PRECHECK OK: residual blocks; 6 arms x H0/H1; 16 search trials; '
          '30 epochs maximum; patience 6; rotating cores; 5 external pairs.')


def rebuild_report(folder: Path) -> None:
    if json.loads((folder / 'run.json').read_text(encoding='utf-8'))['kind'] != 'enriched_resunet':
        raise ValueError('Expected a ResUNet run')
    dense.rebuild_report(folder)


def run(path: Path, resume: Path | None = None, new_run: bool = False) -> None:
    cfg = load_config(path)
    run_loaded(cfg, path, resume, new_run)


def run_loaded(cfg: dict, path: Path, resume: Path | None = None, new_run: bool = False) -> None:
    """Guarded execution shared by the explicit residual and Inception entry points."""
    cfg['output'].mkdir(parents=True, exist_ok=True)
    with exclusive_lock(cfg['output'] / '.execution.lock'):
        chosen = resume.resolve() if resume else None
        if chosen is None and not new_run:
            for state_path in sorted(cfg['output'].glob('*/run.json'), reverse=True):
                state = json.loads(state_path.read_text(encoding='utf-8'))
                if state.get('kind') == cfg['kind']:
                    chosen = state_path.parent
                    break
        if chosen is not None:
            if chosen.parent != cfg['output']:
                raise ValueError('Resume directory must be in the configured model output')
            state = json.loads((chosen / 'run.json').read_text(encoding='utf-8'))
            if state.get('kind') != cfg['kind']:
                raise ValueError('Wrong model family on resume')
            if state.get('status') == 'completed':
                print(f"Already completed: {chosen / 'RELATORIO.md'}")
                return
            if (chosen / 'config.yaml').read_bytes() != cfg['path'].read_bytes():
                raise ValueError('Configuration changed. Resume with the original config or use --new-run.')
            project = project_root()
            for item in json.loads((chosen / 'source_files.json').read_text(encoding='utf-8')):
                if dense.common._sha256(source_path(item['path'])) != item['sha256']:
                    raise ValueError(f"Source changed since checkpoint: {item['path']}")
            print(f'Resuming {chosen}', flush=True)
        with awake(cfg['raw']['runtime'].get('prevent_sleep', True)):
            dense.run(path, resume=chosen)
