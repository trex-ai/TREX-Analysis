# import math
# from scipy import signal
import numpy as np
from datetime import datetime
import pytz
from dateutil.parser import parse as timeparse
import matplotlib.pyplot as plt

import sqlalchemy
from sqlalchemy import create_engine, MetaData, Column, func, insert
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy_utils import database_exists, create_database, drop_database
# import dataset

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


def generate_square_profile(start_datetime_str, end_datetime_str, timezone, peak_power, period:int=1440, time_offset:int=0):
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
    
    half_period = int(period/2)
    power_profile_one = list(np.repeat([-1, 1], half_period))
    power_profile = power_profile_one * int(duration_minutes/len(power_profile_one))
    
    # power_profile = (peak_power/2) * np.cos((2 * np.pi/1440) * (x + time_offset)) + (peak_power/2)
    # power_profile = peak_power * signal.square(2 * np.pi / period * (x + time_offset))
    energy_profile = peak_power * (60/3600) * np.array(power_profile)
    if time_offset:
        energy_profile = np.roll(energy_profile, time_offset)
    return timestamps, energy_profile


def create_profile_table(engine, profile_name:str):
    table = sqlalchemy.Table(
        profile_name,
        MetaData(),
        Column('time', sqlalchemy.Integer, primary_key=True),
        Column('generation', sqlalchemy.Float),
        Column('consumption', sqlalchemy.Float)
    )
    if not database_exists(engine.url):
        create_database(engine.url)
    table.create(engine, checkfirst=True)
    return table


def write_to_db(timestamps, energy_profile, db_str:str, profile_name:str):
    energy_profile_grid = np.array([i if i > 0 else i for i in energy_profile])
    energy_profile_solar = np.array([0 if i > 0 else -i for i in energy_profile])
    profile = [{
        'time': int(timestamps[idx]),
        # 'grid': float(energy_profile_grid[idx]),
        # 'solar': energy_profile_generation[idx],
        # 'solar+': float(energy_profile_solar[idx]),
        'generation': float(max(energy_profile_grid[idx], 0.0)),
        'consumption': float(-min(energy_profile_grid[idx], 0.0))
        } for idx in range(len(timestamps))]
    # return profile
    # db = dataset.connect(db_str)
    # db.create_table(profile_name, primary_id='time')
    engine = create_engine(db_str)
    table = create_profile_table(engine, profile_name)
    # db[profile_name].upsert_many(profile, list(profile[0].keys()))
    with Session(engine) as session:
        session.execute(insert(table), profile)
        session.commit()

start_time = '2010-01-01 0:0:0'
end_time = '2030-01-01 0:0:0'
timezone = 'America/Vancouver'
# time_offset = 0
# time_offset = int(1440/2)
# timestamps, energy_profile = generate_cosine_profile(start_time, end_time, timezone, 1000, time_offset=int(1440/2))
# timestamps, energy_profile = generate_flat_profile(start_time, end_time, timezone, 1000)
timestamps, energy_profile = generate_square_profile(start_time, end_time, timezone, 1000, int(2), int(2/2))
# profile = write_to_db(timestamps, energy_profile,
#             'postgresql://postgres:postgres@localhost/profiles',
#             'test_profile_1kw_square_p2+1')
# plt.plot(timestamps, energy_profile)
# plt.show()
write_to_db(timestamps, energy_profile,
            'postgresql+psycopg://postgres:postgres@localhost/profiles',
            'test_profile_1kw_constant')
print('done')


