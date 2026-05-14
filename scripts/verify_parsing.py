import logging
from src.data_fetcher import UNVotingDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_parsing():
    # Load the HTML we captured
    try:
        with open("debug_rendered.html", "r") as f:
            html = f.read()
    except FileNotFoundError:
        logger.error("debug_rendered.html not found. Run inspect_un_page.py first.")
        return

    fetcher = UNVotingDataFetcher()
    record = fetcher.parse_vote_record(
        html, "https://digitallibrary.un.org/record/3959039?ln=en"
    )

    if record:
        logger.info("SUCCESS: Parsed record!")
        logger.info(f"Title: {record['title']}")
        logger.info(f"Date: {record['date']}")
        logger.info(f"Resolution: {record['resolution']}")
        logger.info(f"Votes found: {len(record['votes'])}")

        # Show sample votes
        logger.info("Sample votes:")
        for vote in record["votes"][:5]:
            logger.info(f"  {vote['country_name']}: {vote['vote']}")

        # Check for Afghanistan
        afghanistan = next(
            (v for v in record["votes"] if "AFGHANISTAN" in v["country_name"]), None
        )
        if afghanistan:
            logger.info(f"Afghanistan vote: {afghanistan}")
        else:
            logger.warning("Afghanistan vote NOT found in parsed results.")

    else:
        logger.error("FAILED: Could not parse record.")


if __name__ == "__main__":
    verify_parsing()
