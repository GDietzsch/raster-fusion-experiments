"""Experiment 12: eight input arms and dense Inception encoder/decoder."""
from pathlib import Path
import json
from src.services import enriched_unet as dense
from src.services.enriched_resunet import run_loaded


def load_config(path: Path) -> dict:
    cfg = dense.load_config(path)
    if cfg['kind'] != 'enriched_inception':
        raise ValueError('Inception entry point requires model_family: inception')
    project = Path(__file__).resolve().parents[2]
    for name in ('enriched_unet', 'enriched_resunet', 'enriched_unet_controls'):
        protected = project / 'runs' / name
        if cfg['output'] == protected or cfg['output'].is_relative_to(protected) or protected.is_relative_to(cfg['output']):
            raise ValueError('Inception output must be separate from previous experiments')
    return cfg


def check_only(path: Path) -> None:
    cfg = load_config(path)
    dense.check_only(path)
    for arm in dense._arms(cfg['raw']['unet']):
        print(arm, len(dense.common._arm_indices(arm)), 'channels (before filtering)')
    print('INCEPTION PRECHECK OK: 8 arms x H0/H1; 24 hyperparameter trials; '
          '30 epochs maximum; patience 6; rotating cores; 5 external pairs.')


def rebuild_report(folder: Path) -> None:
    if json.loads((folder / 'run.json').read_text(encoding='utf-8'))['kind'] != 'enriched_inception':
        raise ValueError('Expected an Inception run')
    dense.rebuild_report(folder)


def run(path: Path, resume: Path | None = None, new_run: bool = False) -> None:
    run_loaded(load_config(path), path, resume, new_run)
