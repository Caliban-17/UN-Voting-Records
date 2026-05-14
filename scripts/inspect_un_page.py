from playwright.sync_api import sync_playwright
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_with_playwright(url):
    with sync_playwright() as p:
        logger.info("Launching browser...")
        browser = p.chromium.launch(headless=True)
        # Create context with user agent
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        logger.info(f"Navigating to {url}...")
        page.goto(url, timeout=60000)

        # Wait for potential dynamic content
        logger.info("Waiting for page load...")
        page.wait_for_load_state("networkidle")

        # Take a screenshot for debugging
        page.screenshot(path="debug_screenshot.png")
        logger.info("Saved debug_screenshot.png")

        # Dump full HTML
        html = page.content()
        with open("debug_rendered.html", "w") as f:
            f.write(html)
        logger.info("Saved debug_rendered.html")

        # Check for "Afghanistan"
        if "Afghanistan" in html:
            logger.info("SUCCESS: 'Afghanistan' found in rendered HTML!")

            # Try to find the container
            # Look for elements containing "Afghanistan"
            elements = page.get_by_text("Afghanistan").all()
            logger.info(f"Found {len(elements)} elements with 'Afghanistan'")

            for i, el in enumerate(elements):
                try:
                    tag = el.evaluate("el => el.tagName")
                    parent_tag = el.evaluate("el => el.parentElement.tagName")
                    parent_class = el.evaluate("el => el.parentElement.className")
                    logger.info(
                        f"Element {i}: <{tag}> inside <{parent_tag} class='{parent_class}'>"
                    )
                except Exception as e:
                    logger.warning(f"Error inspecting element {i}: {e}")

        else:
            logger.warning("WARNING: 'Afghanistan' NOT found even after rendering.")

        browser.close()


if __name__ == "__main__":
    # Search for "voting data" or specific resolution to see result list structure
    # Trying a broad search for "voting" in the "Voting Data" collection if possible, or just general search.
    # Base URL: https://digitallibrary.un.org/
    # We will try to navigate to search results.

    target_url = "https://digitallibrary.un.org/search?ln=en&cc=Voting+Data&p=&f=&rm=&ln=en&sf=year&so=d&rg=50&c=Voting+Data&c=&of=hb&fti=0&fct__2=General+Assembly"
    # This URL attempts to:
    # cc=Voting Data (Collection)
    # sf=year (Sort by year)
    # so=d (Sort descending)
    # rg=50 (50 results per page)
    # fct__2=General Assembly (Filter by body)

    inspect_with_playwright(target_url)
