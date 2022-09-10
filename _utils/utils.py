from datetime import datetime
import pytz
from dateutil.parser import parse as timeparse

def timestamp_to_local(epoch_ts, timezone):
    """Converts UNIX timestamp to local datetime object"""
    return datetime.fromtimestamp(epoch_ts, pytz.timezone(timezone))

def timestr_to_timestamp(time_string:str, timezone:str):
    timestamp = pytz.timezone(timezone).localize(timeparse(time_string))
    return int(timestamp.timestamp())