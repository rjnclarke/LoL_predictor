import asyncio

from crawler.riot_api_client import RiotAPIClient
from crawler.db_handler import DatabaseHandler
from crawler.feature_builder import FeatureBuilder


async def main():
    db = DatabaseHandler("data/matches.db")
    api = RiotAPIClient()
    fb = FeatureBuilder(api, db)
    await fb.run()


if __name__ == "__main__":
    asyncio.run(main())
