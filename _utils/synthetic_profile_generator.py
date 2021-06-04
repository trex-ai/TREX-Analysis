import math
from scipy import signal
import numpy as np
from datetime import datetime
import pytz
from dateutil.parser import parse as timeparse
import matplotlib.pyplot as plt
import dataset

def timestamp_to_local(epoch_ts, timezone):
    """Converts UNIX timestamp to local datetime object"""
    return datetime.fromtimestamp(epoch_ts, pytz.timezone(timezone))

def timestr_to_timestamp(time_string:str, timezone:str):
    timestamp = pytz.timezone(timezone).localize(timeparse(time_string))
    return int(timestamp.timestamp())

def ts_and_duration(start_datetime_str, end_datetime_str, timezone):
    start_datetime = pytz.timezone(timezone).localize(timeparse(start_datetime_str))
    end_datetime = pytz.timezone(timezone).localize(timeparse(end_datetime_str))
    start_timestamp = int(start_datetime.timestamp())
    end_timestamp = int(end_datetime.timestamp())
    duration_minutes = int((end_timestamp - start_timestamp) / 60)
    timestamps = tuple(range(start_timestamp, end_timestamp, 60))
    return timestamps, duration_minutes

def generate_flat_profile(start_datetime_str, end_datetime_str, timezone, peak_power):
    # peak power in Watts
    timestamps, duration_minutes = ts_and_duration(start_datetime_str, end_datetime_str, timezone)
    power_profile = peak_power * np.ones(duration_minutes)
    energy_profile = (60/3600) * power_profile
    return timestamps, energy_profile

def generate_cosine_profile(start_datetime_str, end_datetime_str, timezone, peak_power, time_offset=0):
    # peak power in Watts
    # time_offset in minutes, defaults to 0
    # start_datetime = pytz.timezone(timezone).localize(timeparse(start_datetime_str))
    # end_datetime = pytz.timezone(timezone).localize(timeparse(end_datetime_str))
    # start_timestamp = start_datetime.timestamp()
    # end_timestamp = end_datetime.timestamp()
    # duration_minutes = int((end_timestamp - start_timestamp) / 60)
    # timestamps = np.linspace(start_timestamp, end_timestamp, duration_minutes)

    timestamps, duration_minutes = ts_and_duration(start_datetime_str, end_datetime_str, timezone)
    x = np.linspace(0, duration_minutes, duration_minutes, endpoint=False)
    power_profile = (peak_power/2) * np.cos((2 * np.pi/1440) * (x + time_offset)) + (peak_power/2)
    energy_profile = (60/3600) * power_profile
    return timestamps, energy_profile


def generate_square_profile(start_datetime_str, end_datetime_str, timezone, peak_power, time_offset=0):
    # peak power in Watts
    # time_offset in minutes, defaults to 0
    # start_datetime = pytz.timezone(timezone).localize(timeparse(start_datetime_str))
    # end_datetime = pytz.timezone(timezone).localize(timeparse(end_datetime_str))
    # start_timestamp = start_datetime.timestamp()
    # end_timestamp = end_datetime.timestamp()
    # duration_minutes = int((end_timestamp - start_timestamp) / 60)
    # timestamps = np.linspace(start_timestamp, end_timestamp, duration_minutes)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.square.html
    # +ve=load, -ve=generation

    timestamps, duration_minutes = ts_and_duration(start_datetime_str, end_datetime_str, timezone)
    x = np.linspace(0, duration_minutes, duration_minutes, endpoint=False)
    # power_profile = (peak_power/2) * np.cos((2 * np.pi/1440) * (x + time_offset)) + (peak_power/2)
    power_profile = peak_power * signal.square(2 * np.pi / 1440 * (x + time_offset))
    energy_profile = (60/3600) * power_profile
    return timestamps, energy_profile

def write_to_db(timestamps, energy_profile, db_str:str, profile_name:str):
    energy_profile_grid = np.array([i if i > 0 else i for i in energy_profile])
    energy_profile_solar = np.array([0 if i > 0 else -i for i in energy_profile])
    profile = [{
        'tstamp': int(timestamps[idx]),
        'grid': energy_profile_grid[idx],
        # 'solar': energy_profile_generation[idx],
        'solar+': energy_profile_solar[idx]
        } for idx in range(len(timestamps))]

    db = dataset.connect(db_str)
    db.create_table(profile_name, primary_id='tstamp')
    db[profile_name].insert_many(profile)

start_time = '2010-01-01 0:0:0'
end_time = '2030-01-01 0:0:0'
timezone = 'America/Vancouver'
# time_offset = 0
# time_offset = int(1440/2)
# timestamps, energy_profile = generate_cosine_profile(start_time, end_time, timezone, 1000, time_offset=int(1440/2))
# timestamps, energy_profile = generate_flat_profile(start_time, end_time, timezone, 1000)
timestamps, energy_profile = generate_square_profile(start_time, end_time, timezone, 1000, int(1440/2))
# plt.plot(timestamps, energy_profile)
# plt.show()
write_to_db(timestamps, energy_profile,
            'postgresql://postgres:postgres@localhost/profiles',
            'test_profile_1kw_square+720')



