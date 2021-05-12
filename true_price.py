import dataset
import ast
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import numpy as np

generation = 100
market_id = 'training'
timezone = 'America/Vancouver'
participant = ''
db = dataset.connect('postgresql://postgres:postgres@localhost/TB1-no-bess-100g')

def timestamp_to_local(epoch_ts, timezone):
    """Converts UNIX timestamp to local datetime object"""
    return datetime.fromtimestamp(epoch_ts, pytz.timezone(timezone))


def market_table_filter(generation:int, simulation_type:str):
    prefix = '_'.join((str(generation), simulation_type, 'market'))
    return prefix

market = market_table_filter(99, 'training')


total_quantity = np.zeros(24)
price_x_quantity = np.zeros(24)

for transaction in db[market]:
    if transaction['energy_source'] == 'grid':
        continue

    time_consumption = transaction['time_consumption']
    hour = timestamp_to_local(time_consumption, timezone).hour
    quantity = transaction['quantity']
    settlement_price = transaction['settlement_price']

    total_quantity[hour] += quantity
    price_x_quantity[hour] += settlement_price * quantity



hours = np.arange(24)
true_price = np.divide(price_x_quantity, total_quantity)
plt.scatter(hours, true_price=1)

plt.show()