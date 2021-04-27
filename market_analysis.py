from _utils import extract, extract2
from _utils import utils
import dataset
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import seaborn as sns

# def sanity_check(profile_ts):
#     profiles_sorted = list(profile_ts.keys())
#     profiles_sorted.sort()
#     #sanity checks. Check min and max (avg) 1 min load
#     for profile in profiles_sorted:
#         print(profile,
#               sum(profile_ts[profile]['generation']),
#               sum(profile_ts[profile]['consumption']),
#               round(sum(profile_ts[profile]['profit_e']), 2),
#               round(sum(profile_ts[profile]['profit_f']), 2),
#               round(sum(profile_ts[profile]['cost_e']), 2),
#               round(sum(profile_ts[profile]['cost_f']), 2),
#               round(sum(profile_ts[profile]['net_income']), 2),
#               min(profile_ts[profile]['power']),
#               max(profile_ts[profile]['power']))
#
# def sanity_compare(profile_ts_1, profile_ts_2):
#     profiles_sorted = list(profile_ts_1.keys())
#     profiles_sorted.sort()
#
#     for profile in profiles_sorted:
#         print(profile,
#               sum(profile_ts_2[profile]['generation']) - sum(profile_ts_1[profile]['generation']),
#               sum(profile_ts_2[profile]['consumption']) - sum(profile_ts_1[profile]['consumption']),
#               round(sum(profile_ts_2[profile]['profit_e']) - sum(profile_ts_1[profile]['profit_e']), 2),
#               round(sum(profile_ts_2[profile]['profit_f']) - sum(profile_ts_1[profile]['profit_f']), 2),
#               round(sum(profile_ts_2[profile]['cost_e']) - sum(profile_ts_1[profile]['cost_e']), 2),
#               round(sum(profile_ts_2[profile]['cost_f']) - sum(profile_ts_1[profile]['cost_f']), 2),
#               round(sum(profile_ts_2[profile]['net_income']) - sum(profile_ts_1[profile]['net_income']), 2),
#               round(min(profile_ts_2[profile]['power']) - min(profile_ts_1[profile]['power']), 2),
#               round(max(profile_ts_2[profile]['power']) - max(profile_ts_1[profile]['power']), 2))

# def load_profiles_to_csv(profile_ts, output_path):
#     import os
#     import numpy as np
#     if not os.path.exists(output_path):
#         os.mkdir(output_path)
#
#     profiles_sorted = list(profile_ts.keys())
#     profiles_sorted.sort()
#     for profile in profiles_sorted:
#         filename = output_path+str(profiles_sorted.index(profile))+'.csv'
#         np.savetxt(filename, profile_ts[profile]['power'])

# Create plot of revenue relative to net metering scenario
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

# def compare_profiles(profile_1, profile_2, what):
#     fig = go.Figure()
#     nodes = [node for node in profile_1]
#     nodes.sort()
#
#     for node in nodes:
#         name = node
#         color = '#3D9970'
#         if node == 'grid':
#             continue
#             name = 'grid'
#             color = '#FF4136'
#         # x = profile_1[node]['x_f']
#         # print(profile_1[node])
#         x = profile_1[node]['x_t']
#         x_min = min(x)
#         x = [(x_i - x_min)/(60*60*24) for x_i in x]
#         net_income_advantage = np.subtract(profile_2[node]['net_income'], profile_1[node]['net_income'])
#         fig.add_trace(go.Scatter(
#             x=x,
#             # y=net_income_advantage,
#             y=np.cumsum(net_income_advantage),
#             name=name,
#             #stackgroup='participant'
#             # line=dict(color=color)
#         ))
#     fig.update_layout(
#         title_text='Cumulative Net Income Advantage',
#         title_x=0.5,
#         yaxis_title='Income Advantage ($)',
#         xaxis_title='Day',
#     )
#     # fig.write_image("D:/_simulations/asdasdasd.svg")
#     fig.show()

# def plot_profile(profile):
#     fig = go.Figure()
#     # x = np.linspace(0, 365, 365 * 1440)
#     for node in profile:
#         name = 'participant'
#         color = '#3D9970'
#         if node == 'grid':
#             continue
#             # name = 'grid'
#             # color = '#FF4136'
#
#         net_power = profile[node]['power']
#         x = profile[node]['x_e']
#         # net_consumption = profile[node]['consumption']
#         fig.add_trace(go.Scatter(
#             x=x,
#             y=net_power,
#             name=name,
#             line=dict(color=color),
#             stackgroup='one'
#         ))
#     fig.update_layout(
#         title_text='AAAAAAA',
#         title_x=0.5,
#         yaxis_title='Power? (kW)',
#         xaxis_title='Day',
#     )
#     # fig.write_image("D:/_simulations/figs/58_day_26_gen_overfit-4-powahp.pdf")
#     fig.show()

# def plot_price(prices, source):
#     fig = go.Figure()
#     x = prices['x_e']
#     type = 'physical'
#
#     x_t = []
#     price_t = []
#     for x_i in x:
#         if x_i not in prices[source][type]:
#             continue
#
#         if not prices[source][type][x_i][0]:
#             continue
#
#         x_t.append(x_i)
#         price_t.append(prices[source][type][x_i][0]/ prices[source][type][x_i][1]) # why do you divide by the 1st element?
#
#     x_min = min(x)
#     x_t = [(x_i - x_min) / (60 * 60 * 24) for x_i in x] #converts timestamps into days rather than seconds
#
#     fig.add_trace(go.Scatter(
#         x=x_t,
#         y=price_t,
#         name=source,
#         mode='markers'))
#     fig.update_layout(
#         title_text='Weighted Average PV Settlement Price' + type,
#         title_x=0.5,
#         yaxis_title='Price ($)',
#         xaxis_title='Day',
#     )
#     # fig.write_image("D:/_simulations/figs/58_day_26_gen_overfit-4-pvp.pdf")
#     fig.show()
#
# def net_profit(profile, x_t):
#     fig, ax = plt.subplots(1, figsize=(10, 6))
#     # fig = go.Figure()
#     nodes = [node for node in profile]
#     nodes.sort()
#
#     x_min = min(x_t)
#     x = [(x_i - x_min) / 1440 / 60 for x_i in x_t]
#     for node in nodes:
#     #     if node == 'grid':
#     #         continue
#         net_profit = np.subtract(profile[node][2, :], profile[node][3, :])
#
#         ax.scatter(x, np.divide(net_profit, 1000), s=0.01)
#     ax.set_xlabel('Day')
#     ax.set_ylabel('Net Profit ($)')
#     ax.set_title('Net Profit')
#     # fig.write_image("D:/_simulations/asdasdasd.svg")
#     plt.show()



def net_profit_advantage(profile_1, profile_2, alt_node_names):
    df = pd.DataFrame()

    for node_name in alt_node_names:
        node = alt_node_names[node_name]
        if node == 'grid':
            continue

        net_profit_1 = profile_1[node]['profit'] - profile_1[node]['cost']
        net_profit_2 = profile_2[node]['profit'] -  profile_2[node]['cost']
        net_profit_advantage = net_profit_1 - net_profit_2
        df[node_name] = net_profit_advantage.cumsum()

    df['x_tick'] = np.linspace(0, 30, df.shape[0])
    ax = df.plot(x='x_tick')
    ax.set_xlabel('Day')
    ax.set_ylabel('Income Advantage ($)')
    ax.set_title('Cumulative Net Income Advantage')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig("D:/_simulations/income_advantage_bess_2_way3.pdf")

# def net_unit_settlement_price(profile, x_t, yes_no):
#     fig, ax = plt.subplots(1, figsize=(10, 6))
#     # fig.suptitle('Example Of Scatterplot')
#
#     nodes = [node for node in profile]
#     nodes.sort()
#     x_min = min(x_t)
#     x = [(x_i - x_min) / 1440 / 60 for x_i in x_t]
#
#     for node in nodes:
#         # unit_profit_t = np.divide(profile[node][2, :], profile[node][0, :])
#         unit_cost_t = np.divide(profile[node][3, :], profile[node][1, :])
#         # ax.scatter(x, unit_profit_t, s=0.01, c='r')
#         ax.scatter(x, unit_cost_t, s=0.01, c='r')
#
#     ax.set_xlabel('Day')
#     ax.set_ylabel('$/kWh')
#     ax.set_title('Unit energy price ' + yes_no + ' self consumption')
#     plt.grid()
#     plt.show()

def settlement_price(transactions, offset, save_path:str=''):

    x_tick = transactions['time_consumption'] - offset
    transactions['x_tick'] = x_tick / 3600
    # np.linspace(0, 24, transactions.shape[0])

    # df = pd.DataFrame(transactions).set_index('time_consumption').sort_index()

    CB91_Amber = '#F5B14C'
    CB91_Green = '#47DBCD'
    CB91_Purple = '#9D2EC5'
    palette = {'grid': CB91_Amber,
               'bess': CB91_Purple,
               'solar': CB91_Green}

    plt.figure(figsize=(8, 4))
    ax=sns.scatterplot(x='x_tick', y='settlement_price', data=transactions,
                       hue='energy_source',
                       palette=palette,
                       linewidth=0, s=3)

    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel('Hour')
    ax.set_ylabel('Settlement Price ($/kWh)')
    ax.set_title('Scatterplot of Settlement Prices of a Typical Day')
    ax.legend(loc='lower left')
    ax.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def energy_profile(transactions):

    transactions['x_tick'] = np.linspace(0, 24, transactions.shape[0])

    CB91_Amber = '#F5B14C'
    CB91_Green = '#47DBCD'
    CB91_Purple = '#9D2EC5'
    palette = {'grid': CB91_Amber,
               'bess': CB91_Purple,
               'solar': CB91_Green}

    plt.figure(figsize=(8, 4))
    ax=sns.lineplot(x='x_tick', y='settlement_price', data=transactions,
                       hue='energy_source',
                       palette=palette,
                       linewidth=0, s=3)

    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel('Hour')
    ax.set_ylabel('Settlement Price ($)')
    ax.set_title('Scatterplot of Settlement Prices of a Typical Day')
    ax.legend(loc='lower left')
    ax.grid()
    plt.tight_layout()

    plt.savefig("D:/_simulations/settlement_price_bess_2_way_2.pdf")
    plt.show()


def extract_all_summary(db, transactions, profile_list, time_interval):
    output = {}
    for profile in profile_list:
        data = extract2.transaction_summary(db, transactions, profile, time_interval)
        output[profile] = data
    return output




# baseline_market_p, baseline_market, baseline_prices = extract.from_transactions(db_str, '0_baseline_market')
# csp_market_p, csp_market, csp_prices = extract.from_transactions(db_str, '5_csp_market')
# plot_price(csp_prices, 'solar')
# compare_profiles(baseline_market, csp_market, 'net_income')
#

# alt_node_names = {
#     'R5': 'egauge19821',
#     'R6': 'egauge16616',
#     'R7': 'egauge17073',
#     'R8': 'egauge13974',
#     'R9': 'egauge28684',
#     'R10': 'egauge18158',
#     'R11': 'egauge23534',
#     'R12': 'egauge22271',
#     'R13': 'egauge15623',
#     'R14': 'egauge16608'
# }
#
# list(alt_node_names.values())

# baseline = extract_all_summary(db, baseline_transactions, list(alt_node_names.values()), (start_ts, end_ts))
# training = extract_all_summary(db, training_transactions, list(alt_node_names.values()), (start_ts, end_ts))


# x_t, baseline = extract.physical_transactions(db_str, '0_baseline_market')
# # x_t2, baseline2 = extract.physical_transactions(db_str2, '0_baseline_market')
# _, csp_market = extract.physical_transactions(db_str, '100_training_market')
# _, csp_market_nosc = extract.physical_transactions(db_str, '100_training_market', exclude_self_consumption=True)
# net_profit(csp_market, x_t)
# net_profit(baseline, x_t)
# net_profit_advantage(training, baseline, alt_node_names)
# net_unit_settlement_price(csp_market, x_t, 'incl.')
# net_unit_settlement_price(csp_market_nosc, x_t, 'excl.')

# all_ratios = {'generation': [],
#               'day': [],
#               'grid': [],
#               'self': [],
#               'community': []}
#
# for generation in range(0, 20):
#     table_name = str(generation) + '_training_market'
#     training_transactions = db[table_name].table
#
#     for day in range(1, 31):
#         print(generation, day)
#
#         ratios = {'grid': 0,
#                   'community': 0,
#                   'self': 0}
#         start_ts2 = utils.timestr_to_timestamp('2018-07-' + str(day) +' 0:0:0', 'America/Vancouver')
#         end_ts2 = utils.timestr_to_timestamp('2018-07-' + str(day) +' 23:59:0', 'America/Vancouver')
#         transactions = extract2.transactions(db, training_transactions, (start_ts2, end_ts2), exclude_self_consumption=False)
#
#         for row in transactions.itertuples(index=True, name='id'):
#             if row.buyer_id == 'grid' or row.seller_id == 'grid':
#                 ratios['grid'] += row.quantity
#                 # ratios['grid'] += 1
#             elif row.buyer_id == row.seller_id:
#                 ratios['self'] += row.quantity
#                 # ratios['self'] += 1
#             else:
#                 ratios['community'] += row.quantity
#                 # ratios['community'] += 1
#
#         total = sum(ratios.values())
#         for a in ratios:
#             ratios[a] /= total
#             # ratios[a] = round(ratios[a], 4)
#
#         all_ratios['generation'].append(generation)
#         all_ratios['day'].append(day)
#         for source in ratios:
#             all_ratios[source].append(ratios[source])
#
#     # print(row.quantity, row.energy_source)
# df = pd.DataFrame.from_dict(all_ratios, orient='index').transpose()
# df.set_index('generation', drop=True, inplace=True)
# df2 = df.drop(columns='day')
# df_0 = df[df.generation == 0]
# df_100 = df[df.generation == 100]

# print(df_0.mean())
# print(df_100.mean())
# # heatmap1_data = pd.pivot_table(df, values='community',
# #                      index=['day'],
# #                      columns='generation')
#
# plt.figure(figsize=(8, 4))
# ax = sns.lineplot(data=df2)
# ax.set_xlim(0, 19)
# ax.xaxis.set_major_locator(MultipleLocator(2))
# ax.xaxis.set_minor_locator(MultipleLocator(1))
# ax.set_xlabel('Generation')
# ax.set_ylabel('Ratio')
# ax.set_title('Energy Transaction Ratio Across Generations')
# ax.legend(loc='upper right')
# ax.grid()
# plt.tight_layout()
# plt.savefig("D:/_simulations/tr_bess_1_way.pdf")
# ax = sns.lineplot(x="generation", y="community", data=df)
# plt.show()
# print(ratios)
db_str = 'postgresql://postgres:postgres@stargate/ojape_no_bess_1'
# db_str = 'postgresql://postgres:postgres@stargate/ojape_bess_1way_4'
# db_str = 'postgresql://postgres:postgres@stargate/ojape_bess_2way_2'
# db_str = 'postgresql://postgres:postgres@stargate/ojape_bess_2way_3'
# start_ts = utils.timestr_to_timestamp("2018-07-1 0:0:0", "America/Vancouver")
# end_ts = start_ts + 30 * 60 * 1440

db = dataset.connect(db_str)
baseline_transactions = db['0_baseline_market'].table
training_transactions = db['100_training_market'].table

start_ts2 = utils.timestr_to_timestamp('2018-07-17 0:0:0', 'America/Vancouver')
end_ts2 = utils.timestr_to_timestamp('2018-07-17 23:59:0', 'America/Vancouver')

# print(start_ts2, end_ts2)
transactions = extract2.transactions(db, training_transactions, (start_ts2, end_ts2))
# transactions.set_index('time_purchase', drop=True, inplace=True)
# t2 = transactions.drop(columns=['id', 'time_creation', 'seller_id', 'buyer_id', 'energy_source', 'fee_ask', 'fee_bid'])
# t2['pc'] = t2['settlement_price'] * t2['quantity']
# ax = sns.lineplot(x='time_purchase', y='pc', data=t2)
settlement_price(transactions, offset=start_ts2, save_path='D:/_simulations/settlement_price_no_bess.pdf')
# settlement_price(transactions, offset=start_ts2)

# st = extract2.self_consumption(db, training_transactions, (start_ts2, end_ts2))

# df = pd.DataFrame(transactions).set_index('time_consumption').sort_index()
# settlement_price(st, offset=start_ts2)

# import matplotlib.pyplot as plt
# plt.plot(baseline['grid'][0, :])
# plt.plot(csp_market['eGauge13830']['generation'])
# plt.plot(csp_market['eGauge13836']['generation'])
# plt.plot(csp_market['eGauge14795']['generation'])
# plt.plot(csp_market['grid']['generation'])
# plt.show()tra

# grid_con = []
# for time in csp_market['grid']['x_t']:
#     if time in csp_market_p['grid']['consumption']:
#         grid_con.append(csp_market_p['grid']['consumption'][time])
#     else:
#         grid_con.append(0)