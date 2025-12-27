"""
crawler/feature_builder.py
==========================

FeatureBuilder – builds player feature vectors match‑by‑match.

✓ uses only PUUID‑based endpoints
✓ adds champion‑mastery / challenges enrichment
✓ resumable via vector_complete
✓ caches match history & static data
✓ reports progress
"""

import asyncio, aiohttp, json, time, statistics
from pathlib import Path
from .riot_api_client import RiotAPIClient, HEADERS, PLATFORM_REGION
from .db_handler import DatabaseHandler
from .data_collector import compute_label

CACHE_DIR = Path("cache/player_matches")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Feature builder class
# --------------------------------------------------------------------------- #

class FeatureBuilder:
    def __init__(self, api: RiotAPIClient, db: DatabaseHandler):
        self.api, self.db = api, db

    # ------------------ utility ------------------------------------------------
    def tier_to_norm(self, tier):
        mapping = {
            None: 0.3, "IRON": 0.05, "BRONZE": 0.1, "SILVER": 0.2,
            "GOLD": 0.3, "PLATINUM": 0.4, "EMERALD": 0.5, "DIAMOND": 0.6,
            "MASTER": 0.75, "GRANDMASTER": 0.9, "CHALLENGER": 1.0
        }
        return mapping.get(tier.upper() if tier else None, 0.3)

    def next_unprocessed_match(self):
        cur = self.db.conn.execute(
            "SELECT match_id FROM matches "
            "WHERE vector_complete IS NULL OR vector_complete=0 LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None

    def mark_complete(self, mid):
        self.db.conn.execute(
            "UPDATE matches SET vector_complete=1 WHERE match_id=?", (mid,)
        )
        self.db.conn.commit()

    # ------------------ static enrichment -------------------------------------
    async def fetch_remote_statics(self, session, puuid):
        """Pull rank / mastery / challenges / level using only by‑puuid endpoints."""
        statics = {}

        # rank (league‑v4 by‑puuid)
        u = f"https://{PLATFORM_REGION}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
        ranks = await self.api._safe_get(session, u)
        if isinstance(ranks, list):
            for e in ranks:
                if e.get("queueType") == "RANKED_SOLO_5x5":
                    statics["tier"] = e.get("tier")
                    statics["rank"] = e.get("rank")
                    statics["league_points"] = e.get("leaguePoints")
                    break

        # summoner‑v4 (level / icon)
        u = f"https://{PLATFORM_REGION}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        summ = await self.api._safe_get(session, u)
        if summ:
            statics["summoner_level"] = summ.get("summonerLevel")
            statics["profile_icon"] = summ.get("profileIconId")

        # mastery total score
        u = f"https://{PLATFORM_REGION}.api.riotgames.com/lol/champion-mastery/v4/scores/by-puuid/{puuid}"
        score = await self.api._safe_get(session, u)
        if isinstance(score, (int, float)):
            statics["mastery_score"] = score

        # challenge total points
        u = f"https://{PLATFORM_REGION}.api.riotgames.com/lol/challenges/v1/player-data/{puuid}"
        ch = await self.api._safe_get(session, u)
        if ch:
            statics["challenge_points"] = ch.get("totalPoints", {}).get("current")

        statics["tier_norm"] = self.tier_to_norm(statics.get("tier"))
        return statics

    def get_cached_static(self, puuid):
        row = self.db.conn.execute(
            "SELECT static_json FROM player_features WHERE puuid=?", (puuid,)
        ).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except Exception:
                pass
        return None

    # ------------------ dynamic aggregation -----------------------------------
    def aggregate_history(self, matches_json, puuid):
        k, d, a, gpm, cspm, vis, dmg, w = [], [], [], [], [], [], [], []
        for m in matches_json:
            info = m.get("info", {})
            dur = max(1, info.get("gameDuration", 1) / 60)
            for p in info.get("participants", []):
                if p.get("puuid") == puuid:
                    k.append(p.get("kills", 0))
                    d.append(p.get("deaths", 0))
                    a.append(p.get("assists", 0))
                    gpm.append(p.get("goldEarned", 0) / dur)
                    cspm.append(
                        (p.get("totalMinionsKilled", 0)
                         + p.get("neutralMinionsKilled", 0)) / dur
                    )
                    vis.append(p.get("visionScore", 0))
                    dmg.append(p.get("totalDamageDealtToChampions", 0))
                    w.append(1 if p.get("win") else 0)
                    break
        if not k:
            return {}
        return {
            "kills_avg": statistics.mean(k),
            "deaths_avg": statistics.mean(d),
            "assists_avg": statistics.mean(a),
            "kda": (statistics.mean(k) + statistics.mean(a)) / max(1, statistics.mean(d)),
            "gold_per_min": statistics.mean(gpm),
            "cs_per_min": statistics.mean(cspm),
            "vision_score": statistics.mean(vis),
            "damage_to_champs": statistics.mean(dmg),
            "win_rate_recent": sum(w)/len(w),
        }

    async def fetch_history(self, session, puuid):
        cache_file = CACHE_DIR / f"{puuid}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass
        ids = await self.api.get_match_ids(session, puuid, count=10)
        out = []
        for mid in ids:
            m = await self.api.get_match_detail(session, mid)
            if m:
                out.append(m)
            await asyncio.sleep(0.25)
        cache_file.write_text(json.dumps(out))
        return out

    # ------------------ per‑match processing ----------------------------------
    async def process_match(self, mid):
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            match = await self.api.get_match_detail(session, mid)
            if not match or "info" not in match:
                print(f"[warn] match {mid} unavailable")
                return False

            info = match["info"]
            label = compute_label(info)

            for p in info.get("participants", []):
                puuid = p["puuid"]
                static_data = self.get_cached_static(puuid)
                if not static_data:
                    static_data = await self.fetch_remote_statics(session, puuid)
                    self.db.conn.execute(
                        "UPDATE players SET tier=?, last_scraped=? WHERE puuid=?",
                        (static_data.get("tier"), time.time(), puuid),
                    )
                    self.db.conn.commit()

                hist = await self.fetch_history(session, puuid)
                dynamic = self.aggregate_history(hist, puuid)
                dynamic["label"] = label
                dynamic["tier_norm"] = static_data.get("tier_norm", 0.3)

                self.db.conn.execute("""
                    INSERT OR REPLACE INTO player_features
                        (puuid, tier_norm, static_json, dynamic_json,
                         games_used, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    puuid,
                    static_data.get("tier_norm", 0.3),
                    json.dumps(static_data),
                    json.dumps(dynamic),
                    len(hist),
                    time.time(),
                ))
                self.db.conn.execute(
                    "UPDATE players SET has_features=1 WHERE puuid=?", (puuid,)
                )
                self.db.conn.commit()

            self.mark_complete(mid)
            print(f"✅ match {mid} | {len(info['participants'])} players updated")
            return True

    # ------------------ run loop ----------------------------------------------
    async def run(self, max_matches=None, report_interval=1):
        processed = 0
        while True:
            mid = self.next_unprocessed_match()
            if not mid:
                print("✅ all matches already vector_complete.")
                break

            ok = await self.process_match(mid)
            if not ok:
                print(f"⚠️ skipping {mid}")
                continue

            processed += 1
            if processed % report_interval == 0:
                done = self.db.conn.execute(
                    "SELECT COUNT(*) FROM matches WHERE vector_complete=1"
                ).fetchone()[0]
                total = self.db.conn.execute(
                    "SELECT COUNT(*) FROM matches"
                ).fetchone()[0]
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{total} matches complete")

            if max_matches and processed >= max_matches:
                print(f"⏸ stopped after {processed} match(es)")
                break


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

async def main():
    db = DatabaseHandler("data/matches.db")
    api = RiotAPIClient()
    fb = FeatureBuilder(api, db)
    await fb.run(max_matches=20, report_interval=5)

if __name__ == "__main__":
    asyncio.run(main())