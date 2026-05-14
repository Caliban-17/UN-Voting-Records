import os
import logging
from src.data_processing import _load_and_preprocess_data_impl
from src.config import UN_VOTES_CSV_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_real_data():
    """Validate that the real dataset loads and processes correctly."""
    logger.info(f"Validating real data from: {UN_VOTES_CSV_PATH}")

    if not os.path.exists(UN_VOTES_CSV_PATH):
        logger.error(f"File not found: {UN_VOTES_CSV_PATH}")
        return

    try:
        # Load data (bypass cache to ensure fresh load)
        df, issues = _load_and_preprocess_data_impl(
            str(UN_VOTES_CSV_PATH), use_cache=False
        )

        logger.info("Data loaded successfully!")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Unique Years: {sorted(df['year'].unique())}")
        logger.info(f"Unique Countries: {df['country_identifier'].nunique()}")
        logger.info(f"Unique Issues: {len(issues)}")

        # Check for critical columns
        required = ["rcid", "country_identifier", "vote", "year"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
        else:
            logger.info("All critical columns present.")

        # Check vote values
        logger.info(f"Vote value counts:\n{df['vote'].value_counts(dropna=False)}")

        # Check for NaNs in critical fields
        nan_votes = df["vote"].isna().sum()
        logger.info(f"NaN votes: {nan_votes} ({nan_votes/len(df)*100:.2f}%)")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    validate_real_data()
