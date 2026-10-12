"""Run the six enriched-input comparisons for ordinary linear regression.

Use --check-only before the full run. No training is done by check-only.
"""
from argparse import ArgumentParser
from pathlib import Path

from src.services.enriched_linear import check_only, run


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/enriched_linear.yaml')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        check_only(args.config)
    else:
        run(args.config)


if __name__ == '__main__':
    main()
