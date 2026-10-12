"""Run 06: Extra Trees. Supply your own data in config.yaml."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '_support' / 'raster_fusion_ai'))
from run_enriched_extra_trees import main

if __name__ == '__main__':
    if not any(arg == '--config' or arg.startswith('--config=') for arg in sys.argv[1:]):
        sys.argv.extend(['--config', str(HERE / 'config.yaml')])
    main()
