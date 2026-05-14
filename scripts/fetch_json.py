import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_json(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        }
        logger.info(f"Fetching {url}...")
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        data = response.json()

        # Save to file
        with open("un_record.json", "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved full JSON to un_record.json")

        # Check for "Afghanistan" again in the full dump
        text_dump = json.dumps(data)
        if "Afghanistan" in text_dump:
            logger.info("SUCCESS: 'Afghanistan' found in JSON!")
        else:
            logger.warning("WARNING: 'Afghanistan' NOT found in JSON.")

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    target_url = "https://digitallibrary.un.org/record/3959039?ln=en&of=recjson"
    fetch_json(target_url)
