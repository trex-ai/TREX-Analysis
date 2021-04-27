import dataset
import ast
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import numpy as np


market_id = 'training'
timezone = 'America/Vancouver'
# sim_name = 'TB2-synthetic-balanced-30g'
# sim_name = 'TB1-no-bess-100g'



def timestamp_to_local(epoch_ts, timezone):
    """Converts UNIX timestamp to local datetime object"""
    return datetime.fromtimestamp(epoch_ts, pytz.timezone(timezone))


def metrics_table_filter(tables:list, generation:int, simulation_type:str):
    prefix = '_'.join((str(generation), simulation_type, 'metrics'))
    # print(prefix)
    return [table for table in tables if table.startswith(prefix)]



def get_action_price_qty_time_slice(action, time_window:tuple, resolution):
    data = {}
    for agent in tb:
        for timestep in db[agent]:
            actions_dict = timestep['actions_dict']
            if action in actions_dict:
                timestamps = list(actions_dict[action].keys())
                for timestamp in timestamps:
                    timestamp_end = ast.literal_eval(timestamp)[1]
                    hour = timestamp_to_local(timestamp_end, timezone).hour
                    if hour not in range(*time_window):
                        continue
                    quantity = actions_dict[action][timestamp]['quantity']
                    price = round(actions_dict[action][timestamp]['price']*100, resolution)

                    if price not in data:
                        data[price] = quantity
                    else:
                        data[price] += quantity
    return np.array([(k, v) for k, v in data.items()])


def market_table_filter(generation:int, simulation_type:str):
    prefix = '_'.join((str(generation), simulation_type, 'market'))
    return prefix

def get_settlement_price_qty_time_slice(time_window:tuple, resolution):
    data = {}
    for transaction in db[market]:
        if transaction['seller_id'] == transaction['buyer_id']:
            continue

        if transaction['seller_id'] == 'grid' or transaction['buyer_id'] == 'grid':
            continue

        time_consumption = transaction['time_consumption']
        hour = timestamp_to_local(time_consumption, timezone).hour
        if hour not in range(*time_window):
            continue
        quantity = transaction['quantity']

        for sk in ('settlement_price', 'settlement_price_buy', 'settlement_price_sell'):
            if sk not in transaction:
                continue
            settlement_price = round(transaction[sk]*100, resolution)
            if settlement_price not in data:
                data[settlement_price] = quantity
            else:
                data[settlement_price] += quantity
    return np.array([(k, v) for k, v in data.items()])

import plotly.graph_objects as go

# sim_name = 'TB3-sma2-10s-5d'
# db = dataset.connect('postgresql://postgres:postgres@localhost/'+sim_name)
# tb = metrics_table_filter(db.tables, generation, 'training')
# market = market_table_filter(generation, 'training')
generation = 50
time_window = (0, 24) #first inclusive, last non-inclusive

# 1s10d -> expect higher bids
# 10s1d -> expect lower asks

fig = go.Figure()
for d in ((1, 10), (5, 5), (10, 1)):

    # d = (10, 1)
    # range(1, 10, 2)
    #     sim_name = 'TB3-ute2-rns2-' + str(d[0]) + 's-' + str(d[1]) + 'd'
    config = str(d[0]) + 's-' + str(d[1]) + 'd'
    sim_name = 'TB5-ute3-pamax-' + config
    # sim_name = 'TB3-ute2-pamax-' + config
    # sim_name = 'TB1-ute-pamax-' + config
    db = dataset.connect('postgresql://postgres:postgres@localhost/' + sim_name)
    tb = metrics_table_filter(db.tables, generation, 'training')
    market = market_table_filter(generation, 'training')

    sup = []
    dem = []
    sett = []

    for gen in range(max(0, generation-10), generation+1):
        tb = metrics_table_filter(db.tables, gen, 'training')
        market = market_table_filter(gen, 'training')
        sup.append(get_action_price_qty_time_slice('asks', time_window, 2))
        dem.append(get_action_price_qty_time_slice('bids', time_window, 2))

    sup = np.concatenate(sup, axis=0)
    dem = np.concatenate(dem, axis=0)
    fig.add_trace(go.Scatter(x=sup[:, 0], y=sup[:, 1], mode='markers', name=config + 'asks'))
    fig.add_trace(go.Scatter(x=dem[:, 0], y=dem[:, 1], mode='markers', name=config + 'bids'))

fig.show()
