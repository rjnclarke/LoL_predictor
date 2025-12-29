"""
crawler/build_db.py
===================

Test runner for the MatchBase build pipeline.

Demonstrates how the dual‑database system and minimal‑update logic
work together to build and verify the local dataset.

Usage
-----
$ python -m crawler.build_db
"""

import asyncio
import sqlite3
from .match_base import MatchBase

async def main():
    """
    Entry point for the test workflow.

    Steps:
        1. Instantiate MatchBase.
        2. Trigger a full rebuild.
        3. Inspect resulting tables.
    """
    mb = MatchBase(
        live_path="data/match_base/live.db",
        update_path="data/match_base/update.db"
    )

    # Step 1 – Initial rebuild (runs seeding + limited updates)
    print("\n🏗️  Starting full DB build...")
    await mb.build_from_scratch()

    # Step 2 – Quick inspection of created tables
    conn = sqlite3.connect("data/match_base/live.db")

    def count(table):
        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]

    print("\n📊  Database summary after build:")
    for t in ["players", "matches", "player_match_stats"]:
        try:
            print(f"  {t:<20} →  {count(t)} rows")
        except Exception:
            print(f"  {t:<20} →  not found")

    print("\n✅  Build complete. Inspect 'data/match_basse/live.db' for content.\n")

if __name__ == "__main__":
    asyncio.run(main())