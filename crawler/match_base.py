"""
crawler/match_base.py
=====================

Central data‑management layer for all operations that touch the live dataset.

This class acts as the *Single Source of Truth* for player, match, and per‑match
statistics.  It unifies the Riot API calls, database access, and update logic,
while staying separate from both viz and model code.

SOLID PRINCIPLES
----------------
S – Single Responsibility: Only manages data integrity and update flow.
O – Open/Closed: Extendable through new update modes or validation methods
                  without changing its core interface.
L – Liskov Substitution: Keeps internal contracts stable (query/update methods
                         return predictable structures).
I – Interface Segregation: Exposes simple, focused methods (verify_player,
                           update_player, build_db).
D – Dependency Inversion: Depends on abstractions (RiotAPIClient, DatabaseHandler)
                           not concrete implementations.
"""

import asyncio, aiohttp, shutil, os, time
from .riot_api_client import RiotAPIClient
from .db_handler import DatabaseHandler
import json

class MatchBase:
    """
    High‑level orchestrator connecting the Riot API and persistent storage.

    Handles:
        • player verification and insertion
        • incremental updates using minimal API calls
        • dual‑database copying and live‑swap
        • recomputation housekeeping (e.g., trimming to last 10 matches)
    """

    def __init__(self, live_path="data/live.db", update_path="data/update.db"):
        # Paths for the two database copies
        self.live_path = live_path
        self.update_path = update_path

        # Open the current live DB; the update DB will be lazy‑created
        self.db = DatabaseHandler(live_path)
        self.api = RiotAPIClient()

        # Cached global attributes – cheap to compute once
        self.level_min = None
        self.level_max = None

    # ------------------------------------------------------------------ #
    # --- Utility / metadata functions ----------------------------------#
    # ------------------------------------------------------------------ #
    def compute_level_bounds(self):
        """Compute global min/max summoner level for normalization."""
        cur = self.db.conn.execute("SELECT MIN(last_scraped), MAX(last_scraped) FROM players")
        self.level_min, self.level_max = cur.fetchone()

    def log(self, msg: str):
        """Simple console logger with timestamps."""
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ------------------------------------------------------------------ #
    # --- Player management ---------------------------------------------#
    # ------------------------------------------------------------------ #
    async def verify_player(self, puuid: str):
        """
        Ensure a player exists in the database.

        If missing, attempts to scrape minimal player info via the Riot API.
        This keeps the database self‑healing even when new participants
        appear in matches during updates.
        """
        cur = self.db.conn.execute("SELECT 1 FROM players WHERE puuid=?", (puuid,))
        if cur.fetchone():
            return True  # already known

        async with aiohttp.ClientSession() as session:
            url = f"https://euw1.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
            data = await self.api._safe_get(session, url)
            if not data:
                self.log(f"Failed to verify new player {puuid}")
                return False

            self.db.conn.execute(
                "INSERT OR IGNORE INTO players(puuid, last_scraped) VALUES(?,?)",
                (puuid, time.time()),
            )
            self.db.conn.commit()
            self.log(f"🟢 Added new player {data.get('name','?')} ({puuid[:8]}...)")
            return True

    # ------------------------------------------------------------------ #
    # --- Core update logic ---------------------------------------------#
    # ------------------------------------------------------------------ #
    async def update_player(self, puuid: str):
        """
        Fetch matches newer than the player's last known match and slide window.

        Pseudologic:
            1. Retrieve last known timestamp for player.
            2. Get recent match IDs from Riot API (count=10).
            3. For each ID not already stored → fetch details and insert.
            4. Keep only 10 most recent entries.
        """
        async with aiohttp.ClientSession() as session:
            ids = await self.api.get_match_ids(session, puuid, count=10)
            if not ids:
                return 0

            new_count = 0
            for mid in ids:
                # Skip existing entries
                cur = self.db.conn.execute(
                    "SELECT 1 FROM player_match_stats WHERE puuid=? AND match_id=?",
                    (puuid, mid)
                )
                if cur.fetchone():
                    continue

                detail = await self.api.get_match_detail(session, mid)
                if not detail or "info" not in detail:
                    continue

                info = detail["info"]
                # Extract basic per‑player stats
                for p in info.get("participants", []):
                    if p["puuid"] == puuid:
                        self.db.conn.execute("""
                            INSERT OR REPLACE INTO player_match_stats
                                (puuid, match_id, timestamp, role, stats_json)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            puuid,
                            mid,
                            info.get("gameStartTimestamp", time.time()),
                            p.get("teamPosition"),
                            json.dumps({
                                "kills": p.get("kills",0),
                                "deaths": p.get("deaths",0),
                                "assists": p.get("assists",0),
                                "gold": p.get("goldEarned",0),
                                "damage": p.get("totalDamageDealtToChampions",0),
                                "vision": p.get("visionScore",0)
                            })
                        ))
                        self.db.conn.commit()
                        new_count += 1
                        break

            # Trim to 10 recent matches
            self.db.conn.execute("""
                DELETE FROM player_match_stats
                WHERE rowid NOT IN (
                    SELECT rowid FROM player_match_stats
                    WHERE puuid=? ORDER BY timestamp DESC LIMIT 10
                ) AND puuid=?;
            """, (puuid, puuid))
            self.db.conn.commit()
            self.log(f"Updated {puuid[:8]}…  (+{new_count} new)")
            return new_count

    async def update_all_players(self):
        """
        Iterate every player and run a minimal update.
        This is the standard periodic refresh operation.
        """
        cur = self.db.conn.execute("SELECT puuid FROM players")
        players = [r[0] for r in cur.fetchall()]
        for puuid in players:
            try:
                await self.update_player(puuid)
            except Exception as e:
                self.log(f"⚠️ {puuid[:8]}… skipped: {e}")

    # ------------------------------------------------------------------ #
    # --- Database copy / swap ------------------------------------------#
    # ------------------------------------------------------------------ #
    def copy_live_db(self):
        """Duplicate live database into update path safely."""
        if os.path.exists(self.update_path):
            os.remove(self.update_path)
        shutil.copy(self.live_path, self.update_path)
        self.log("Copied live → update DB")

    def promote_update_db(self):
        """Replace live DB with update DB after successful refresh."""
        if os.path.exists(self.live_path):
            os.remove(self.live_path)
        shutil.copy(self.update_path, self.live_path)
        self.log("Promoted update DB → live")

    async def build_from_scratch(self):
        """
        Complete rebuild (used only for first run or schema change).
        """
        self.log("🚧 Building database from scratch…")
        if os.path.exists(self.live_path):
            os.remove(self.live_path)
        self.db = DatabaseHandler(self.live_path)
        # Here we might re‑seed challengers or all ladders again.
        self.log("✅ Fresh database created.")

# End of MatchBase