import asyncio
import logging
from datetime import datetime, timedelta
from src.data_fetcher import UNVotingDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_loop():
    fetcher = UNVotingDataFetcher()

    # Try to fetch votes from the last 365 days
    # This should trigger the search, find some records, and hopefully parse them
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    logger.info(
        f"Testing fetch_recent_votes from {start_date.date()} to {end_date.date()}"
    )

    df = await fetcher.fetch_recent_votes(start_date, end_date)

    if not df.empty:
        logger.info(f"SUCCESS: Fetched {len(df)} votes!")
        logger.info(df.head())
        logger.info(f"Unique Resolutions: {df['resolution'].unique()}")
    else:
        logger.warning(
            "No votes found (this might be normal if no votes in range), but loop finished."
        )


if __name__ == "__main__":
    asyncio.run(verify_loop())
