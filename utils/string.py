import datetime
import pytz
from utils.logging_config import logger

# Add this function to convert timestamps
def convert_to_rfc3339(timestamp_str: str) -> str | None:
    if not timestamp_str:
        return None
    try:
        # Parse the timestamp string
        dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S%z")
        # Convert to RFC3339 format
        return dt.astimezone(pytz.UTC).isoformat()
    except ValueError:
        logger.warning(f"Could not parse timestamp {timestamp_str}")
        return None