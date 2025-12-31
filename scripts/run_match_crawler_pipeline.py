import asyncio

from crawler.riot_api_client import RiotAPIClient
from crawler.db_handler import DatabaseHandler
from crawler.data_collector import PlayerCollector, MatchCrawler


async def main():
    db = DatabaseHandler("data/matches.db")
    api = RiotAPIClient()

    pc = PlayerCollector(api, db)
    await pc.seed_players()

    mc = MatchCrawler(api, db, matches_per_player=10, target_matches=5000)
    await mc.run()


if __name__ == "__main__":
    asyncio.run(main())
