"""Run the enriched-input Ridge experiment or rebuild a completed report."""
from argparse import ArgumentParser
from pathlib import Path

from src.services.enriched_ridge import check_only, rebuild_report, run


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/enriched_ridge.yaml')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--report-only', type=Path, metavar='RUN_DIR')
    args = parser.parse_args()
    if args.check_only and args.report_only:
        parser.error('--check-only and --report-only cannot be combined')
    if args.check_only:
        check_only(args.config)
    elif args.report_only:
        rebuild_report(args.report_only)
    else:
        run(args.config)


if __name__ == '__main__':
    main()
