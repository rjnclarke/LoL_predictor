import argparse
import asyncio
from typing import Dict

from crawler.match_base import MatchBase
from crawler.db_handler import DatabaseHandler


def fetch_stats(match_base: MatchBase) -> Dict[str, int]:
    conn = match_base.db.conn
    stats = {
        "players": conn.execute("SELECT COUNT(*) FROM players").fetchone()[0],
        "players_with_features": conn.execute(
            "SELECT COUNT(*) FROM players WHERE has_features=1"
        ).fetchone()[0],
        "matches": conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
        "matches_vector_complete": conn.execute(
            "SELECT COUNT(*) FROM matches WHERE vector_complete=1"
        ).fetchone()[0],
        "player_match_stats": conn.execute(
            "SELECT COUNT(*) FROM player_match_stats"
        ).fetchone()[0],
        "player_features": conn.execute(
            "SELECT COUNT(*) FROM player_features"
        ).fetchone()[0],
    }
    return stats


def print_summary(title: str, stats: Dict[str, int]):
    print(f"\n📊 {title}")
    for key, value in stats.items():
        print(f"  • {key.replace('_', ' '):22s} → {value:>6}")


def print_deltas(before: Dict[str, int], after: Dict[str, int]):
    print("\n✅ Update summary (Δ after - before):")
    for key in before.keys():
        delta = after[key] - before[key]
        sign = f"{delta:+d}"
        print(f"  • {key.replace('_', ' '):22s} → {after[key]:>6} ({sign})")


async def main():
    parser = argparse.ArgumentParser(description="Run MatchBase updates with stats.")
    parser.add_argument(
        "--db",
        default="data/match_base/live.db",
        help="Path to the live database (default: data/live.db)",
    )
    parser.add_argument(
        "--update-db",
        default="data/update.db",
        dest="update_db",
        help="Path to the temporary update database (default: data/update.db)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of players to update (default: all)",
    )
    args = parser.parse_args()

    mb = MatchBase(live_path=args.db, update_path=args.update_db)

    before = fetch_stats(mb)
    print_summary("Database snapshot (before)", before)

    print("\n📀 Copying live → update DB...")
    mb.copy_live_db()

    # Switch MatchBase to operate on the update DB copy
    mb.db.conn.close()
    mb.db = DatabaseHandler(args.update_db)

    print("⚙️  Updating players on update DB...")
    await mb.update_all_players(limit=args.limit)

    after = fetch_stats(mb)
    print_deltas(before, after)

    # Promote the freshly updated DB back to the live location
    print("\n🚀 Promoting update DB → live DB...")
    mb.db.conn.close()
    mb.promote_update_db()
    mb.db = DatabaseHandler(args.db)

    final_stats = fetch_stats(mb)
    print_summary("Database snapshot (after promote)", final_stats)


if __name__ == "__main__":
    asyncio.run(main())
