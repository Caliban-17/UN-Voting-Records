"""
Data Fetcher Module for UN Voting Records
Fetches and updates voting data from UN Digital Library via web scraping using Playwright
"""

import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional, Dict
import logging
from datetime import datetime, timedelta
import time
from pathlib import Path
from playwright.async_api import async_playwright

from src.config import (
    UN_DIGITAL_LIBRARY_URL,
    SCRAPING_DELAY_SECONDS,
    MAX_RETRIES,
    DATA_DIR,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)


class UNVotingDataFetcher:
    """
    Fetches UN voting records from the UN Digital Library.
    Uses Playwright to handle dynamic content.
    """

    def __init__(self):
        """Initialize the data fetcher."""
        self.base_url = UN_DIGITAL_LIBRARY_URL
        self.delay = SCRAPING_DELAY_SECONDS
        self.max_retries = MAX_RETRIES
        # Playwright context options
        self.browser_args = {
            "headless": True,
            "args": ["--no-sandbox", "--disable-setuid-sandbox"],
        }
        self.context_args = {
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

    async def fetch_page_dynamic(self, url: str, retry_count: int = 0) -> Optional[str]:
        """
        Fetch a page using Playwright, waiting for dynamic content.

        Args:
            url: URL to fetch
            retry_count: Current retry attempt

        Returns:
            HTML content or None if failed
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(**self.browser_args)
                context = await browser.new_context(**self.context_args)
                page = await context.new_page()

                logger.info(f"Navigating to {url}...")
                await page.goto(url, timeout=60000)

                # Wait for network idle to ensure content is loaded
                try:
                    await page.wait_for_load_state("networkidle", timeout=30000)
                except Exception:
                    logger.warning(
                        "Timeout waiting for networkidle, proceeding with current content"
                    )

                # Get content
                content = await page.content()
                await browser.close()

                return content

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            if retry_count < self.max_retries:
                wait_time = (retry_count + 1) * self.delay * 2
                logger.info(
                    f"Retrying... ({retry_count + 1}/{self.max_retries}) in {wait_time}s"
                )
                await asyncio.sleep(wait_time)
                return await self.fetch_page_dynamic(url, retry_count + 1)
            return None

    def parse_vote_record(self, html: str, record_url: str) -> Optional[Dict]:
        """
        Parse voting record data from HTML.
        Handles the text-based vote format found in UN Digital Library.

        Format example: "Y AFGHANISTAN<br> Y ALBANIA<br> A ALGERIA"

        Args:
            html: HTML content
            record_url: URL of the record

        Returns:
            Dictionary with vote data or None
        """
        try:
            soup = BeautifulSoup(html, "lxml")

            record = {
                "undl_link": record_url,
                "date": None,
                "session": None,
                "title": None,
                "resolution": None,
                "votes": [],  # List of {country_code, country_name, vote}
            }

            # 1. Parse Metadata
            # Look for metadata rows
            metadata_rows = soup.find_all("div", class_="metadata-row")

            for row in metadata_rows:
                title_span = row.find("span", class_="title")
                value_span = row.find("span", class_="value")

                if not title_span or not value_span:
                    continue

                title = title_span.get_text(strip=True).lower()
                value = value_span.get_text(strip=True)

                if "vote date" in title:
                    record["date"] = value
                elif "resolution" in title:
                    record["resolution"] = value
                elif "title" in title:  # Sometimes title is just the main header
                    record["title"] = value

            # Fallback for title if not in metadata
            if not record["title"]:
                h1 = soup.find("h1")  # Usually the title is h1
                if h1:
                    record["title"] = h1.get_text(strip=True)

            # 2. Parse Votes
            # The votes are in a span class="value" that contains "AFGHANISTAN"
            # We search for the specific span containing country names

            vote_span = None
            # Heuristic: Find span with class 'value' that contains 'AFGHANISTAN'
            candidates = soup.find_all("span", class_="value")
            for span in candidates:
                if "AFGHANISTAN" in span.get_text():
                    vote_span = span
                    break

            if vote_span:
                # The content is separated by <br> tags
                # We need to process the HTML to preserve line breaks or just split by text
                # get_text(separator='|') is useful here
                text_content = vote_span.get_text(separator="|")
                entries = text_content.split("|")

                for entry in entries:
                    entry = entry.strip()
                    if not entry:
                        continue

                    # Format: "Y COUNTRY NAME" or "A COUNTRY NAME" or "N COUNTRY NAME"
                    # Sometimes there might be extra spaces
                    parts = entry.split(" ", 1)
                    if len(parts) < 2:
                        continue

                    vote_code = parts[0].strip()
                    country_name = parts[1].strip()

                    # Validate vote code (Y, N, A, etc.)
                    # We map it to our internal code if needed, or keep as is
                    # VOTE_ENCODING keys are 'Y', 'N', 'A', etc.

                    if vote_code in ["Y", "N", "A", "X"]:  # Basic validation
                        record["votes"].append(
                            {
                                "country_code": country_name,  # We use name as code initially, will need mapping later
                                "country_name": country_name,
                                "vote": vote_code,
                            }
                        )

            if not record["votes"]:
                logger.warning(f"No votes found in {record_url}")
                return None

            return record

        except Exception as e:
            logger.error(f"Error parsing vote record: {e}")
            return None

    async def fetch_recent_votes(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch recent voting records from UN Digital Library.
        Iterates through search results until it finds records older than start_date.

        Args:
            start_date: Start date for fetching
            end_date: End date for fetching

        Returns:
            DataFrame with new voting records
        """
        logger.info(f"Fetching votes from {start_date.date()} to {end_date.date()}")

        all_records = []
        jrec = 1
        records_per_page = 50
        keep_fetching = True

        # Base search URL parameters
        # cc=Voting Data, sf=year (sort by year), so=d (descending), rg=50 (records per page)
        # fct__2=General Assembly (filter by body)
        base_search_url = (
            f"{self.base_url}/search?ln=en&cc=Voting+Data&p=&f=&rm=&sf=year&so=d"
            f"&rg={records_per_page}&c=Voting+Data&c=&of=hb&fti=0&fct__2=General+Assembly"
        )

        while keep_fetching:
            search_url = f"{base_search_url}&jrec={jrec}"
            logger.info(f"Fetching search results page: {search_url}")

            html = await self.fetch_page_dynamic(search_url)
            if not html:
                logger.warning("Failed to fetch search page, stopping.")
                break

            soup = BeautifulSoup(html, "lxml")

            # Find record links
            # Links are like /record/12345?ln=en
            # We look for unique record IDs to avoid duplicates
            record_links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/record/" in href and "ln=en" in href:
                    # Normalize URL
                    if not href.startswith("http"):
                        href = (
                            f"{self.base_url}{href}"
                            if href.startswith("/")
                            else f"{self.base_url}/{href}"
                        )
                    record_links.add(href)

            if not record_links:
                logger.info("No more records found on this page.")
                break

            logger.info(f"Found {len(record_links)} records on page starting at {jrec}")

            # Fetch each record
            for record_url in record_links:
                # Check if we already have this record (optimization)
                # For now, we fetch everything and filter by date

                record_html = await self.fetch_page_dynamic(record_url)
                if not record_html:
                    continue

                record = self.parse_vote_record(record_html, record_url)
                if not record or not record["date"]:
                    continue

                # Parse date
                try:
                    # Date format in UN DL is usually YYYY-MM-DD
                    record_date = datetime.strptime(record["date"], "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Could not parse date: {record['date']}")
                    continue

                # Check date range
                if record_date > end_date:
                    continue  # Too new (unlikely if sorted desc, but possible)

                if record_date < start_date:
                    logger.info(
                        f"Found record from {record_date.date()} which is older than start date {start_date.date()}. Stopping."
                    )
                    keep_fetching = False
                    break

                # Add to list
                all_records.append(record)
                logger.info(f"Added record: {record['title']} ({record['date']})")

            # Next page
            jrec += records_per_page

            # Safety break to prevent infinite loops during testing
            if jrec > 500:  # Limit to 10 pages for safety
                logger.warning("Reached safety limit of 500 records. Stopping.")
                break

        # Convert to DataFrame
        if not all_records:
            logger.warning("No new voting records found")
            return pd.DataFrame()

        # Flatten vote data
        rows = []
        for record in all_records:
            # Extract RCID from URL if possible (e.g. record/12345)
            rcid = record["undl_link"].split("/record/")[1].split("?")[0]

            for vote in record["votes"]:
                rows.append(
                    {
                        "rcid": rcid,  # Using rcid instead of undl_id to match config
                        "country_code": vote["country_code"],
                        "ms_name": vote["country_name"],
                        "vote": vote["vote"],
                        "date": record["date"],
                        "session": record.get("session"),
                        "title": record.get("title"),
                        "resolution": record.get("resolution"),
                        "undl_link": record["undl_link"],
                        "year": datetime.strptime(record["date"], "%Y-%m-%d").year,
                    }
                )

        df = pd.DataFrame(rows)
        logger.info(
            f"Fetched {len(df)} total vote entries from {len(all_records)} records"
        )

        return df

    def merge_with_local_data(
        self, new_df: pd.DataFrame, existing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge new and existing rows, deduplicating on a single source-of-truth id.

        DANGER: the historical CSV uses ``undl_id`` for the record id and
        ``ms_code`` for the country code; the fetcher writes ``rcid`` and
        ``country_code``. Naively deduplicating on the fetcher's names
        causes the entire historical dataset (where those columns are NaN)
        to collapse to a single row — this used to silently destroy data.

        Defenses, in order:

        1. Refuse to run with empty new data — never shrink the file by
           accident.
        2. Normalize: copy ``rcid`` → ``undl_id`` and ``country_code`` →
           ``ms_code`` on the new rows so the two sources share keys.
        3. Use ``("undl_id", "ms_code")`` as the dedup key — never one that
           contains only-NaN values.
        4. Hard sanity check: refuse to return a frame smaller than the
           existing input. The script's caller logs the delta either way,
           but we never overwrite the CSV if rows would be lost.
        """
        if new_df is None or new_df.empty:
            logger.info("Nothing to merge — new_df empty; returning existing.")
            return existing_df

        new_df = new_df.copy()
        # Make the new rows speak the historical CSV's column names so
        # dedup actually works.
        if "rcid" in new_df.columns and "undl_id" not in new_df.columns:
            new_df["undl_id"] = new_df["rcid"]
        if "country_code" in new_df.columns and "ms_code" not in new_df.columns:
            new_df["ms_code"] = new_df["country_code"]

        logger.info(
            "Merging new (%d rows) with existing (%d rows).",
            len(new_df), len(existing_df) if existing_df is not None else 0,
        )

        combined = pd.concat(
            [existing_df, new_df] if existing_df is not None else [new_df],
            ignore_index=True,
        )

        # Pick a key both DataFrames actually have values for. undl_id is the
        # canonical UN Digital Library record id and is non-null in both.
        key_priority = (
            ["undl_id", "ms_code"],
            ["rcid", "country_code"],
            ["date", "ms_code", "resolution"],
            ["date", "country_code", "resolution"],
        )
        chosen_key: list[str] | None = None
        for cols in key_priority:
            if all(c in combined.columns for c in cols):
                # Both sources must populate at least one row each on this key.
                if combined[cols].dropna().shape[0] >= max(1, len(existing_df) // 2):
                    chosen_key = cols
                    break
        if chosen_key is None:
            raise RuntimeError(
                "merge_with_local_data: no viable dedup key found — refusing "
                "to drop_duplicates and risk silent data loss."
            )

        combined = combined.drop_duplicates(subset=chosen_key, keep="last")
        logger.info(
            "Deduped on %s → %d rows (Δ %+d).",
            chosen_key, len(combined), len(combined) - (len(existing_df) if existing_df is not None else 0),
        )

        # Final defensive check — caller will compare and write only if this
        # passes, but we refuse to be the source of silent shrinkage.
        existing_rows = len(existing_df) if existing_df is not None else 0
        if existing_rows > 0 and len(combined) < existing_rows:
            raise RuntimeError(
                f"merge_with_local_data: result has {len(combined)} rows but "
                f"existing had {existing_rows}. Refusing to return — would lose "
                f"{existing_rows - len(combined)} rows."
            )

        return combined

    def save_updated_data(
        self, df: pd.DataFrame, filename: str = "un_votes_updated.csv"
    ):
        """
        Save updated dataset to file.

        Args:
            df: DataFrame to save
            filename: Output filename
        """
        filepath = DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved updated data to {filepath}")

        # Also save as parquet for faster loading
        parquet_path = CACHE_DIR / filename.replace(".csv", ".parquet")
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved parquet cache to {parquet_path}")


async def fetch_and_update(
    days_back: int = 30, existing_data_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Convenience function to fetch recent votes and update local data.
    """
    fetcher = UNVotingDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    new_df = await fetcher.fetch_recent_votes(start_date, end_date)

    # Load existing data logic...
    # (Simplified for now)
    return new_df


def schedule_daily_update():
    """
    Schedule automatic daily updates.
    """
    import schedule
    from src.config import AUTO_UPDATE_CRON, ENABLE_AUTO_UPDATE

    if not ENABLE_AUTO_UPDATE:
        logger.info("Automatic updates disabled")
        return

    async def update_task():
        logger.info("Running scheduled update...")
        try:
            await fetch_and_update(days_back=7)
            logger.info("Scheduled update completed successfully")
        except Exception as e:
            logger.error(f"Scheduled update failed: {e}")

    time_part = AUTO_UPDATE_CRON.split()[1]
    schedule.every().day.at(f"{time_part}:00").do(lambda: asyncio.run(update_task()))

    logger.info(f"Scheduled daily updates at {time_part}:00")

    while True:
        schedule.run_pending()
        time.sleep(60)
