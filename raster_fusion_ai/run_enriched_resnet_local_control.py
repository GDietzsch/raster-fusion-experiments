"""Run experiment 09A: fixed-recipe local-only ResNet ablation."""
from argparse import ArgumentParser
from pathlib import Path

from src.services.enriched_cnn_local_control import check_only, rebuild_report, run


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parent / 'configs/enriched_resnet_local_control.yaml')
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--check-only', action='store_true')
    operation.add_argument('--report-only', type=Path, metavar='RUN_DIR')
    operation.add_argument('--resume', type=Path, metavar='RUN_DIR')
    args = parser.parse_args()
    if args.check_only:
        check_only(args.config)
    elif args.report_only:
        rebuild_report(args.report_only)
    else:
        run(args.config, resume=args.resume)


if __name__ == '__main__':
    main()
