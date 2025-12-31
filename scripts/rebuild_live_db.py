import argparse
import asyncio
from crawler.match_base import MatchBase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild the live MatchBase database in-place."
    )
    parser.add_argument(
        "--live-path",
        default="data/match_base/live.db",
        help="Target SQLite file to rebuild (default: data/match_base/live.db)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mb = MatchBase(live_path=args.live_path)
    asyncio.run(mb.build_from_scratch())


if __name__ == "__main__":
    main()
