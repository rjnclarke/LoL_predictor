import asyncio, aiohttp, os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {"X-Riot-Token": API_KEY}

PLATFORM_REGION = "euw1"
ROUTING_REGION  = "europe"

class RiotAPIClient:
    """Asynchronous Riot API wrapper with basic throttling and back‑off."""

    def __init__(self, sem_limit=5, cooldown=0.25):
        self.sem = asyncio.Semaphore(sem_limit)
        self.cooldown = cooldown

    async def _safe_get(self, session: aiohttp.ClientSession, url: str):
        """Perform GET with simple rate‑limit handling."""
        async with self.sem:
            async with session.get(url, headers=HEADERS) as r:
                if r.status == 429:  # rate‑limit
                    wait = int(r.headers.get("Retry-After", 2))
                    print(f"⚠️ 429 – waiting {wait}s")
                    await asyncio.sleep(wait)
                    return await self._safe_get(session, url)
                if r.status != 200:
                    print(f"[WARN] {r.status} → {url}")
                    return None
                data = await r.json()
                await asyncio.sleep(self.cooldown)
                return data

    async def get_ladder_puuids(self, session: aiohttp.ClientSession, tier: str):
        """Retrieve PUUIDs from a specific ladder tier."""
        url = (f"https://{PLATFORM_REGION}.api.riotgames.com/lol/league/v4/"
               f"{tier}leagues/by-queue/RANKED_SOLO_5x5")
        data = await self._safe_get(session, url)
        if not data:
            return []
        entries = data.get("entries", [])
        return [e["puuid"] for e in entries if "puuid" in e]

    async def get_all_tier_puuids(self, session):
        """Collect Challenger, Grandmaster and Master PUUIDs."""
        tiers = ["challenger", "grandmaster", "master"]
        all_puuids = []
        for tier in tiers:
            ps = await self.get_ladder_puuids(session, tier)
            print(f"➡️ {tier.title()} → {len(ps)}")
            all_puuids.extend(ps)
        # deduplicate while preserving order
        seen, ordered = set(), []
        for p in all_puuids:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        print(f"✅ Total seed PUUIDs: {len(ordered)}")
        return ordered

    async def get_match_ids(self, session, puuid: str, count: int = 5):
        """Latest match IDs for a given PUUID."""
        url = (f"https://{ROUTING_REGION}.api.riotgames.com/"
               f"lol/match/v5/matches/by-puuid/{puuid}/ids?count={count}")
        data = await self._safe_get(session, url)
        return data or []

    async def get_match_detail(self, session, match_id: str):
        """Full match detail JSON."""
        url = f"https://{ROUTING_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return await self._safe_get(session, url)