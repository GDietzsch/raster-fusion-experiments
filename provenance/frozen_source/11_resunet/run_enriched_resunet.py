"""Run experiment 11 ResUNet, automatically resuming an unfinished run."""
from argparse import ArgumentParser
from pathlib import Path
from src.services.enriched_resunet import check_only, rebuild_report, run


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/enriched_resunet.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--report-only', type=Path)
    mode.add_argument('--resume', type=Path)
    mode.add_argument('--new-run', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        check_only(args.config)
    elif args.report_only:
        rebuild_report(args.report_only)
    else:
        run(args.config, args.resume, args.new_run)


if __name__ == '__main__':
    main()
