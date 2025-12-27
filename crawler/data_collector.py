"""
crawler/data_collector.py
=========================

PlayerCollector  –  seeds the players table with all Challenger,
                    Grandmaster and Master ladder PUUIDs.

MatchCrawler      –  iterates players, fetches match IDs and details,
                    keeps ordered PUUIDs and continuous labels,
                    and prints ongoing progress.

This reproduces the behaviour of the test script
inside the organised architecture.
"""

import asyncio
import aiohttp
from .riot_api_client import RiotAPIClient
from .db_handler import DatabaseHandler

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

ROLES_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

def order_puuids_by_role(players):
    """Return 10 PUUIDs in fixed [blue roles, red roles] order."""
    blue, red = [], []
    for role in ROLES_ORDER:
        for p in players:
            if p["teamId"] == 100 and p["teamPosition"] == role:
                blue.append(p["puuid"])
            if p["teamId"] == 200 and p["teamPosition"] == role:
                red.append(p["puuid"])
    return blue + red

def compute_label(info):
    """
    Continuous label measuring blue/left‑team success.
    y = 0.55 × gold_ratio + 0.45 × win_flag
    """
    teams = info["teams"]
    win100 = teams[0]["win"]
    gold100 = sum(p["goldEarned"] for p in info["participants"] if p["teamId"] == 100)
    gold200 = sum(p["goldEarned"] for p in info["participants"] if p["teamId"] == 200)
    ratio = gold100 / (gold100 + gold200)
    win_flag = 1.0 if win100 else 0.0
    return 0.55 * ratio + 0.45 * win_flag


# --------------------------------------------------------------------------- #
# PlayerCollector – low‑volume seeding job
# --------------------------------------------------------------------------- #

class PlayerCollector:
    """Fetch Challenger + GM + Master players and save to DB."""

    def __init__(self, api: RiotAPIClient, db: DatabaseHandler):
        self.api = api
        self.db = db

    async def seed_players(self):
        """Populate player table with ladder PUUIDs."""
        async with aiohttp.ClientSession() as session:
            puuids = await self.api.get_all_tier_puuids(session)
            for p in puuids:
                self.db.insert_player(p)
        count = self.db.conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        print(f"✅  Seeded {count} players from ladders.")


# --------------------------------------------------------------------------- #
# MatchCrawler – high‑volume asynchronous crawl
# --------------------------------------------------------------------------- #

class MatchCrawler:
    """
    Loops through players and collects unique matches.

    For each new match:
        * fetch full details
        * compute label  (blue-side success)
        * order players  (TOP→SUP for both sides)
        * write to SQLite
        * print running progress
    """

    def __init__(self, api: RiotAPIClient, db: DatabaseHandler,
                 matches_per_player: int = 10, target_matches: int = 5000):
        self.api = api
        self.db = db
        self.mpp = matches_per_player
        self.target = target_matches

    async def run(self):
        async with aiohttp.ClientSession() as session:
            processed = self.db.match_count()
            while processed < self.target:
                players = self.db.player_batches(limit=5)
                if not players:
                    print("⚠️ No players left to scrape.")
                    break

                for puuid in players:
                    ids = await self.api.get_match_ids(session, puuid, count=self.mpp)
                    for mid in ids:
                        if self.db.match_exists(mid):
                            continue

                        match = await self.api.get_match_detail(session, mid)
                        if not match or "info" not in match:
                            continue
                        info = match["info"]

                        # keep only ranked solo queue
                        if info.get("queueId") != 420:
                            continue

                        ordered = order_puuids_by_role(info["participants"])
                        if len(ordered) != 10:
                            continue

                        label = compute_label(info)
                        self.db.insert_match(mid, info, ordered, label)
                        processed = self.db.match_count()

                        # add any new PUUIDs from this match
                        for pid in match["metadata"]["participants"]:
                            self.db.insert_player(pid, discovered=1)
                            self.db.conn.execute(
                                "UPDATE players SET in_match=1 WHERE puuid=?",
                                (pid,)
                            )
                        self.db.conn.commit()

                        if processed % 5 == 0:
                            print(f"🟢  {processed} / {self.target} matches stored.")

                    self.db.mark_scraped(puuid)

                # refresh loop condition
                processed = self.db.match_count()

            print(f"✅  Crawl completed: {processed} matches in database.")