"""Run U-Net controls 10A, 10B and 10C sequentially, with automatic resume."""
from argparse import ArgumentParser
from pathlib import Path
from src.services.enriched_unet_controls import precheck, run, rebuild_reports


def main(default_experiment='all'):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/enriched_unet_controls.yaml')
    parser.add_argument('--experiment', choices=('all', '10A', '10B', '10C'), default=default_experiment)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--report-only', type=Path)
    mode.add_argument('--resume', type=Path)
    mode.add_argument('--new-run', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        precheck(args.config)
    elif args.report_only:
        rebuild_reports(args.report_only)
    else:
        run(args.config, args.experiment, args.resume, args.new_run)


if __name__ == '__main__':
    main()
