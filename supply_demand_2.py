import dataset
import ast
from datetime import datetime
import pytz
import numpy as np
import math


timezone = 'America/Vancouver'

def timestamp_to_local(epoch_ts, timezone):
    """Converts UNIX timestamp to local datetime object"""
    return datetime.fromtimestamp(epoch_ts, pytz.timezone(timezone))


def metrics_table_filter(tables:list, generation:int, simulation_type:str):
    prefix = '_'.join((str(generation), simulation_type, 'metrics'))
    # print(prefix)
    return [table for table in tables if table.startswith(prefix)]

def market_table_filter(generation:int, simulation_type:str):
    prefix = '_'.join((str(generation), simulation_type, 'market'))
    return prefix

def get_action_price_qty(action, resolution):
    data = {}
    for agent in tb:
        for timestep in db[agent]:
            actions_dict = timestep['actions_dict']
            if action in actions_dict:
                timestamps = list(actions_dict[action].keys())
                for timestamp in timestamps:
                #     timestamp_end = ast.literal_eval(timestamp)[1]
                #     hour = timestamp_to_local(timestamp_end, timezone).hour
                #     if hour not in range(*time_window):
                #         continue
                    quantity = actions_dict[action][timestamp]['quantity'] / 1000
                    price = round(actions_dict[action][timestamp]['price']*100, resolution)
                    if price not in data:
                        data[price] = quantity
                    else:
                        data[price] += quantity
    return data


def get_settlement_price_qty(market, resolution):
    data = {}
    for transaction in db[market]:
        if transaction['seller_id'] == transaction['buyer_id']:
            continue

        if transaction['seller_id'] == 'grid' or transaction['buyer_id'] == 'grid':
            continue

        # time_consumption = transaction['time_consumption']
        # convert from Wh to kWh
        quantity = transaction['quantity'] / 1000
        for sk in ('settlement_price', 'settlement_price_buy', 'settlement_price_sell'):
            if sk not in transaction:
                continue

            # if sk in ('settlement_price_buy', 'settlement_price_sell'):
            #     quantity /= 2

            price = round(transaction[sk] * 100, resolution)
            if price not in data:
                data[price] = quantity
            else:
                data[price] += quantity
    return data


def update_dict1_values(d1, d2):
    for k in d2:
        if k not in d1:
            d1[k] = d2[k]
        else:
            d1[k] += d2[k]
    return d1

def supply_demand_to_readable(supply, demand):
    n_supply = str(supply / min(supply, demand))
    n_demand = str(demand / min(supply, demand))
    sd_ratio = str(n_supply) + ':' + str(n_demand)
    if supply > demand:
        return 'Excess Supply (supply:demand = ' + sd_ratio + ')'
    elif supply < demand:
        return 'Excess Demand (supply:demand = ' + sd_ratio + ')'
    return 'Equal Supply and Demand'

# import plotly.graph_objects as go
import matplotlib.pyplot as plt
market_id = 'training'
episode = 100
episodes = 10
# time_window = (0, 24) #first inclusive, last non-inclusive

# 1s10d -> expect higher bids
# 5s5d -> expect bids and asks to be around the middle
# 10s1d -> expect lower asks

# fig = go.Figure()
sup_dem_scales = [(1, 10), (10, 1), (5, 5)]
# sup_dem_scales = [(10, 15), (15, 10), (1, 10), (10, 1), (5, 5)]
fig_y = 1.5 * len(sup_dem_scales)
# sup_dem_scales = [(1, 10), (10, 1)]
fig, axs = plt.subplots(len(sup_dem_scales), 1, sharex='col', figsize=(6, fig_y))

# sim_prefix = 'ae2021-ute-qbandit-'
# sim_prefix = 'ae2021-ute2-qbandit-'
sim_prefix = 'ae2021-ute3-qbandit-'
# sim_prefix = 'ae2021-ute3-qbandit-'
# sim_prefix = 'ae2021-ute3-qbandit-anneal-exp-'
# sim_prefix = 'ae2021-ute3-qbandit-anneal-lr-exp-'
# for episode in [25, 50, 100, 150, 199]:
for index in range(len(sup_dem_scales)):
    supply = sup_dem_scales[index][0]
    demand = sup_dem_scales[index][1]
    config = str(supply) + 's-' + str(demand) + 'd'
    sim_name = sim_prefix + config

    db = dataset.connect('postgresql://postgres:postgres@localhost/' + sim_name)

    asks = {}
    bids = {}
    setts = {}

    for eps in range(max(0, episode-episodes), episode+1):
        tb = metrics_table_filter(db.tables, eps, market_id)
        market = market_table_filter(eps, market_id)
        asks = update_dict1_values(asks, get_action_price_qty('asks', 2))
        bids = update_dict1_values(bids, get_action_price_qty('bids', 2))
        setts = update_dict1_values(setts, get_settlement_price_qty(market, 2))

    asks = np.array([(k, v) for k, v in asks.items()])
    asks = asks[np.argsort(asks[:, 0])]
    asks[:, 1] /= max(asks[:, 1])

    bids = np.array([(k, v) for k, v in bids.items()])
    bids = bids[np.argsort(bids[:, 0])]
    bids[:, 1] /= max(bids[:, 1])

    setts = np.array([(k, v) for k, v in setts.items()])
    setts = setts[np.argsort(setts[:, 0])]
    setts[:, 1] /= max(setts[:, 1])
    # asks[:, 1] /= (episodes * d[0])
    # bids[:, 1] /= (episodes * d[1])
    # size = 15
    size = 15
    alpha = 1
    axs[index].scatter(x=asks[:, 0], y=asks[:, 1], s=size, marker='.', alpha=alpha, label='asks', color='dodgerblue')
    axs[index].scatter(x=bids[:, 0], y=bids[:, 1], s=size, marker='.', alpha=alpha, label='bids', color='darkred')
    axs[index].scatter(x=setts[:, 0], y=setts[:, 1], s=size, marker='.', alpha=alpha, label='settlements', color='hotpink')
    axs[index].title.set_text(supply_demand_to_readable(supply, demand))
    # axs[index].title.set_text('Supply Demand Ratio ' + str(supply) + ':' + str(demand))
    axs[index].grid()

    # fig.add_trace(go.Scatter(x=asks[:, 0], y=asks[:, 1], mode='markers', name=config + ' asks'))
    # fig.add_trace(go.Scatter(x=bids[:, 0], y=bids[:, 1], mode='markers', name=config + ' bids'))
    # fig.add_trace(go.Scatter(x=setts[:, 0], y=setts[:, 1], mode='markers', name=config + ' settlements'))
axs[0].legend(loc="upper left")
axs[-1].set_xlabel("Price (\u00a2/kWh)")
axs[int(np.floor(len(axs)/2))].set_ylabel("Normalized Quantity")

plt.tight_layout()
plt.savefig(sim_prefix + 'plot.pdf')
fig.show()

