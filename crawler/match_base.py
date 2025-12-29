"""
crawler/match_base.py
=====================

Central manager for all match and player data.

Handles:
    - Database creation, seeding, and sliding‑window updates
    - Dual‑DB copy/swap for safe background updates
    - Player verification and minimal API‑call refresh

Design Principles:
------------------
S – Single Responsibility: maintains data layer & update flow.
O – Open for extension: tier lists, update strategies, caching.
L – Stable API contracts with RiotAPIClient and DatabaseHandler.
"""

import asyncio, aiohttp, shutil, os, time, json
from .riot_api_client import RiotAPIClient
from .db_handler import DatabaseHandler


class MatchBase:
    """Orchestrates Riot‑API data collection and persistence."""

    # ------------------------------------------------------------- #
    # --- INITIALIZATION ------------------------------------------- #
    # ------------------------------------------------------------- #
    def __init__(self, live_path="data/live.db", update_path="data/update.db"):
        self.live_path = live_path
        self.update_path = update_path

        self.db = DatabaseHandler(live_path)
        self.api = RiotAPIClient()

        self.level_min = None
        self.level_max = None

    # ------------------------------------------------------------- #
    # --- Utility helpers ----------------------------------------- #
    # ------------------------------------------------------------- #
    def log(self, msg: str):
        """Timestamped log utility for build output."""
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def compute_level_bounds(self):
        """Compute global min/max level for visualization normalization."""
        cur = self.db.conn.execute(
            "SELECT MIN(last_scraped), MAX(last_scraped) FROM players"
        )
        self.level_min, self.level_max = cur.fetchone()

    # ------------------------------------------------------------- #
    # --- Player verification & seeding --------------------------- #
    # ------------------------------------------------------------- #
    async def verify_player(self, puuid: str):
        """
        Ensure a player record exists in DB.
        If new, adds minimal entry using Riot API.
        """
        cur = self.db.conn.execute("SELECT 1 FROM players WHERE puuid=?", (puuid,))
        if cur.fetchone():
            return True

        async with aiohttp.ClientSession() as session:
            url = (
                f"https://euw1.api.riotgames.com/"
                f"lol/summoner/v4/summoners/by-puuid/{puuid}"
            )
            data = await self.api._safe_get(session, url)
            if not data:
                self.log(f"⚠️  Could not verify new player {puuid}")
                return False

            self.db.conn.execute(
                "INSERT OR IGNORE INTO players(puuid, last_scraped) VALUES(?,?)",
                (puuid, time.time()),
            )
            self.db.conn.commit()
            self.log(f"🟢 Added new player {data.get('name','?')} ({puuid[:8]}...)")
            return True

    async def seed_players(self, tiers=None):
        """
        Seed selected ladder tiers (Challenger/GM/Master).

        Uses the original RiotAPIClient calls for reliability.

        Parameters
        ----------
        tiers : list[str] | None
            Default fetches all three ladders.
            Override with ["challenger"] for testing or smaller builds.
        """
        tiers = tiers or ["challenger", "grandmaster", "master"]

        async with aiohttp.ClientSession() as session:
            all_puuids = []
            for tier in tiers:
                self.log(f"Fetching {tier.title()} ladder...")
                puuids = await self.api.get_ladder_puuids(session, tier)
                all_puuids.extend(puuids)

            # Deduplicate while keeping order
            seen, ordered = set(), []
            for p in all_puuids:
                if p not in seen:
                    seen.add(p)
                    ordered.append(p)

            for p in ordered:
                self.db.insert_player(p)

        count = self.db.conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        self.log(f"✅  Seeded {count} players across: {', '.join(tiers)}")

    # ------------------------------------------------------------- #
    # --- Player update logic ------------------------------------- #
    # ------------------------------------------------------------- #
    async def update_player(self, puuid: str):
        """
        Refresh one player's recent matches using minimal API calls.

        Steps:
            1. Request last 10 match IDs.
            2. For unseen IDs → fetch match details & insert player stats.
            3. Trim records to the newest 10.
        """
        async with aiohttp.ClientSession() as session:
            ids = await self.api.get_match_ids(session, puuid, count=10)
            if not ids:
                return 0

            new_count = 0
            for mid in ids:
                # Skip if already stored
                cur = self.db.conn.execute(
                    "SELECT 1 FROM player_match_stats WHERE puuid=? AND match_id=?",
                    (puuid, mid),
                )
                if cur.fetchone():
                    continue

                match = await self.api.get_match_detail(session, mid)
                if not match or "info" not in match:
                    continue
                info = match["info"]

                # Extract stats for this specific player
                for p in info.get("participants", []):
                    if p["puuid"] == puuid:
                        stat = {
                            "kills": p.get("kills", 0),
                            "deaths": p.get("deaths", 0),
                            "assists": p.get("assists", 0),
                            "gold": p.get("goldEarned", 0),
                            "damage": p.get("totalDamageDealtToChampions", 0),
                            "vision": p.get("visionScore", 0),
                        }
                        self.db.insert_player_match(
                            puuid,
                            mid,
                            info.get("gameStartTimestamp", time.time()),
                            p.get("teamPosition"),
                            stat,
                        )
                        new_count += 1
                        break

            # Keep only the last 10 matches
            self.db.delete_old_matches(puuid, keep=10)
            self.db.mark_scraped(puuid)
            if new_count:
                self.log(f"Updated {puuid[:8]}… (+{new_count} new)")
            return new_count

    async def update_all_players(self, limit=None):
        """
        Iterate through all tracked players and update them.
        Optionally limit for testing.
        """
        cur = self.db.conn.execute("SELECT puuid FROM players")
        players = [r[0] for r in cur.fetchall()]
        if limit:
            players = players[:limit]

        for puuid in players:
            try:
                await self.update_player(puuid)
            except Exception as e:
                self.log(f"⚠️ {puuid[:8]} skipped → {e}")

    # ------------------------------------------------------------- #
    # --- Dual‑DB management -------------------------------------- #
    # ------------------------------------------------------------- #
    def copy_live_db(self):
        """Duplicate live → update DB safely."""
        if os.path.exists(self.update_path):
            os.remove(self.update_path)
        shutil.copy(self.live_path, self.update_path)
        self.log("💾  Copied live → update DB")

    def promote_update_db(self):
        """Replace live DB with fully updated copy."""
        if os.path.exists(self.live_path):
            os.remove(self.live_path)
        shutil.copy(self.update_path, self.live_path)
        self.log("🚀  Promoted update → live DB")

    # ------------------------------------------------------------- #
    # --- Full rebuild (Production) ------------------------------- #
    # ------------------------------------------------------------- #
    async def build_from_scratch(self):
        """
        Perform a complete fresh build:
            • Remove old DB
            • Create new schema
            • Seed Challenger players only
            • Run full update across all players
            • Compute global metadata
        """
        self.log("🚧  Building full Challenger database...")
        if os.path.exists(self.live_path):
            os.remove(self.live_path)
        self.db = DatabaseHandler(self.live_path)

        # Step 1 — Seed only Challenger tier
        await self.seed_players(tiers=["challenger"])

        # Step 2 — Full update (all 300 players)
        await self.update_all_players()

        # Step 3 — Compute normalization metadata
        self.compute_level_bounds()
        self.log("✅  Full Challenger build complete.")