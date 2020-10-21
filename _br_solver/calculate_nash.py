# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _extractor.extractor import Extractor
from sqlalchemy import create_engine
import numpy as np
import copy

from _utils.market_simulation import match
from _utils.rewards_proxy import Reward
# import _utils.market_simulation

# Dirty solution to be abe to import from TREX-Core,
# could be improved with os by walking one up and then adding TREX-Core
import sys
# ----------------------------------------------------------------------------------------------------------------------
# tests
# test settlement process by attempting to verify what we should also have in the DB
# the test is passed if we get the market simulation accuretely represents the market
# FOR NOW this probably does not account for self-consumption!!!

def _check_config(config):
    if 'data' not in config.keys():
        print('missing key <data> in experiment config file')

    if 'study' not in config['data'].keys():
        print('missing sub-key <study>')

    if 'start_datetime' not in config['data']['study'].keys():
        print('missing one of the sub^2-keys <start_datetime> or <days>')

def _sim_market(participants:dict, learning_agent_id:str, timestep:int=None):
    learning_agent = participants[learning_agent_id]
    # opponents = copy.deepcopy(participants)
    # opponents.pop(learning_agent_id, None)
    open = {}
    learning_agent_times_delivery = []
    market_sim_df = []
    if timestep is None:
        timesteps = range(len(learning_agent['metrics']['actions_dict']))
    else:
        timesteps = [timestep]

    for idx in timesteps:
        for participant_id in participants:
            agent_actions = participants[participant_id]['metrics']['actions_dict'][idx]

            for action in ('bids', 'asks'):
                if action in agent_actions:
                    for time_delivery in agent_actions[action]:
                        if time_delivery not in open:
                            open[time_delivery] = {}
                        if action not in open[time_delivery]:
                            open[time_delivery][action] = []

                        aa = agent_actions[action][time_delivery]
                        aa['participant_id'] = participant_id
                        open[time_delivery][action].append(aa)
                        if participant_id == learning_agent_id:
                            learning_agent_times_delivery.append(time_delivery)

    for t_d in learning_agent_times_delivery:
        if 'bids' in open[t_d] and 'asks' in open[t_d]:
            market_sim_df.extend(match(open[t_d]['bids'], open[t_d]['asks'], 'solar', t_d))

    return pd.DataFrame(market_sim_df)

def _get_market_records_for_agent(participant:str, market_df):
    # get stuff from the market df, filter out grid and self-consumption
    sucessfull_bids_log = market_df[market_df['buyer_id'] == participant]
    sucessfull_bids_log = sucessfull_bids_log[sucessfull_bids_log['seller_id'] != 'grid']
    sucessfull_bids_log = sucessfull_bids_log[sucessfull_bids_log['seller_id'] != participant]
    # sucessfull_bids_log = sucessfull_bids_log[sucessfull_bids_log['energy_source'] == 'solar']

    sucessfull_asks_log = market_df[market_df['seller_id'] == participant]
    sucessfull_asks_log = sucessfull_asks_log[sucessfull_asks_log['buyer_id'] != 'grid']
    sucessfull_asks_log = sucessfull_asks_log[sucessfull_asks_log['buyer_id'] != participant]
    # sucessfull_asks_log = sucessfull_asks_log[sucessfull_asks_log['energy_source'] == 'solar']

    return sucessfull_bids_log, sucessfull_asks_log

def _compare_records(market_sim_df, market_db_df):

    if market_sim_df.shape[0] != market_db_df.shape[0]:
        print('market dataframe num_entries inconsistent, failed test')
        return False

    if np.sum(market_sim_df['quantity']) != np.sum(market_db_df['quantity']):
        print('cumulative quantities not equivalent, failed test')
        return False

    if market_sim_df.shape[0] and market_db_df.shape[0] != 0:
        if np.median(market_sim_df['settlement_price']) != np.median(market_db_df['settlement_price']):
            print('median price not equivalent, failed test')
            return False

        if np.mean(market_sim_df['settlement_price']) != np.mean(market_db_df['settlement_price']):
            print('mean price not equivalent, failed test')
            return False

    print('passed tests')
    return True

def _test_settlement_process(participants:dict, learning_agent_id:str, market_df):

    market_sim_df = _sim_market(participants, learning_agent_id)

    sim_bids, sim_asks = _get_market_records_for_agent(learning_agent_id, market_sim_df)
    db_bids, db_asks = _get_market_records_for_agent(learning_agent_id, market_df)
    print(sim_bids.columns, db_bids.columns)
    print('testing for bids equivalence')
    bids_identical = _compare_records(sim_bids, db_bids)

    print('testing for asks equivalence')
    asks_identical = _compare_records(sim_asks, db_asks)

    if bids_identical and asks_identical:
        print('passed market equivalence test')
        return True
    else:
        print('failed market equivalence test')
        return False
# ----------------------------------------------------------------------------------------------------------------------
def _map_market_to_ledger(market_df_ts, learning_agent):

    quantity = market_df_ts['quantity']
    price = market_df_ts['settlement_price']
    source = market_df_ts['energy_source']
    if (market_df_ts['seller_id'] == learning_agent) & (market_df_ts['buyer_id'] != 'grid'):
        action = 'ask'
    elif (market_df_ts['buyer_id'] == learning_agent) & (market_df_ts['buyer_id'] != 'grid'):
        action = 'bid'
    else:
        action = None
    ledger_entry = (action, quantity, price, source)
    return ledger_entry

def get_G(participants_dict, reward_fun, learning_agent, grid_transaction_dummy:tuple):
    market_df = _sim_market(participants=participants_dict, learning_agent_id=learning_agent)
    learning_agent_market_interactions = market_df[(market_df['seller_id'] == learning_agent) & (market_df['buyer_id'] != 'grid') | (market_df['buyer_id'] == learning_agent) & (market_df['seller_id'] != 'grid')]

    market_ledger = []
    for index in range(learning_agent_market_interactions.shape[0]):
        timeslice = learning_agent_market_interactions.iloc[index]
        market_ledger.append(_map_market_to_ledger(timeslice, learning_agent))

    rewards = []
    for index in range(len(market_ledger)):
        r = reward_fun.calculate(market_transactions=[market_ledger[index]], grid_transactions=grid_transaction_dummy)
        rewards.append(r)

    return sum(rewards)
# ----------------------------------------------------------------------------------------------------------------------
# general stuff we need for this to work
def _get_tables(engine):
    find_profile_names = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
    """
    table = pd.read_sql_query(find_profile_names, engine) # get tables

    return [element[0] for element in table.values.tolist()] #unpack, because list of lists

def _add_metrics_to_participants(participants_dict):
    for participant in participants_dict.keys():
        participant_dict = participants_dict[participant]

        if 'track_metrics' not in participant_dict['trader']:
            print('no <track_metrics> in participant dict!')

        if participant_dict['trader']['track_metrics'] == True:
            participants_dict[participant]['metrics'] = extractor.from_metrics(start_gen, sim_type, participant)

    return participants_dict
# ----------------------------------------------------------------------------------------------------------------------
# Value iteration shit
def _query_market_get_reward_for_one_tuple(participants_dict, ts, learning_participant):
    market_df = _sim_market(participants=participants_dict, learning_agent_id=learning_participant, timestep=ts)

    market_ledger = []
    for index in range(market_df.shape[0]):
        timeslice = market_df.iloc[index]
        market_ledger.append(_map_market_to_ledger(timeslice, learning_participant))
    rewards = []
    for index in range(len(market_ledger)):
        rewards.append(reward_fun.calculate(market_transactions=[market_ledger[index]], grid_transactions=grid_transaction_dummy))

    return sum(rewards)

# sweep over all applicable actions for the learning participant and bootstrap Q
def _get_bootstrap_Q(participants_dict, ts, learning_participant):
    # find Q* previous ts
    Q_max = {}
    steps = len(participants_dict[learning_participant]['Q']) #gets the max timestamp
    for action_type in participants_dict[learning_participant]['Q'][ts]:
        # ask for max ts key first
        # check if key exists, if not error warning
        # fetch Q_max
        if ts +1 == steps:
            print('hit end')
            Q_max[action_type] = 0.0
        else:
            Q_max[action_type] = max(participants_dict[learning_participant]['Q'][ts+1][action_type])

    return Q_max

def _Q_sweep(participants_dict, ts, learning_participant):
    Q_bootstrap = _get_bootstrap_Q(participants_dict, ts, learning_participant)

    # ToDo: FOR NOW, we are only doing action-wise
    # might be better to combine those somehow
    actions = participants_dict[learning_participant]['Q_space'].keys()

    if ('bid_price' in actions) & ('bid_quantity' and 'ask_price' and 'ask_quantity' not in actions):

        ns_load = participants_dict[learning_participant]['metrics']['next_settle_load'][ts]
        ns_gen = participants_dict[learning_participant]['metrics']['next_settle_generation'][ts]
        ns_net_load = ns_load - ns_gen

        if ns_net_load > 0:
            if 'bids' not in participants_dict[learning_participant]['metrics']['actions_dict'][ts]:
                print('found missing bids, autocomplete not implemented yet')
                #ToDo: autocomplete bids if and when necessary

            key = list(participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'].keys())[0]

            bid_Q = []

            for idx in range(len(participants_dict[learning_participant]['Q_space']['bid_price'])):

                bid_price = participants_dict[learning_participant]['Q_space']['bid_price'][idx]
                participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'][key]['price'] = bid_price
                r = _query_market_get_reward_for_one_tuple(participants_dict, ts, learning_participant)

                bid_Q.append(r + Q_bootstrap['bid_price'])

            return {'bid_price': bid_Q}

    return participants_dict

# perform value iteration for one participant
def _Q_iterate_participant(participants_dict, learning_participant):
    timesteps = len(participants_dict[learning_participant]['Q'])

    # ToDo: iterate overa all applicable actions
    # ToDo: modify the actions dict according to greedy policy
    Q_array = []
    for ts in reversed(range(timesteps)):
        bidQ = _Q_sweep(participants_dict, ts, learning_participant)
        Q_array.append(bidQ)
    return Q_array

# u[date greedy strategy based on Q values
def _update_to_new_policy(participant_dict):
    new_policy = []

    for ts in range(len(participant_dict['metrics']['actions_dict'])):
        entry = {}
        for action  in participant_dict['metrics']['actions_dict'][ts].keys():
            entry[action] = {}

            if action == 'bids':
                old_policy = participants_dict['egauge19821']['metrics']['actions_dict'][ts]['bids']
                ts_key = list(participant_dict['metrics']['actions_dict'][ts]['bids'].keys())[0]
                old_policy = old_policy[ts_key]


                if 'bid_price' in participant_dict['Q_space']:
                    best_action_index = np.argmax(participant_dict['Q'][ts]['bid_price'])
                    best_price = participant_dict['Q_space']['bid_price'][best_action_index]
                    old_policy['price'] = best_price
                entry[action][ts_key]= old_policy
        new_policy.append(entry)
    return new_policy


# perform value iteration for all participants where applicable one full time
def _full_Q_iteration(participants_dict):
    for participant in participants_dict:
        if 'learning' in participants_dict[participant]['trader']:
            if participants_dict[participant]['trader']['learning'] == True:
                Q_updated_participant = _Q_iterate_participant(participants_dict, learning_participant=participant)
                del participants_dict[participant]['Q']
                participants_dict[participant]['Q'] = Q_updated_participant
                participants_dict[participant]['metrics']['actions_dict'] = _update_to_new_policy(participants_dict[participant])

    return participants_dict

def _fetch_Qs(participants_dict):
    for participant in participants_dict:
        if 'learning' in participants_dict[participant]['trader']:
            if participants_dict[participant]['trader']['learning'] == True:
                participants_old_Q = {}
                participants_old_Q[participant] = participants_dict[participant]['Q']

    return participants_old_Q

def _add_Q_and_Qspace(participants_dict:dict, timesteps:int=10, prices:tuple=(0,1), action_space_separation:int=10):
    for participant in participants_dict:
        if 'learning' in participants_dict[participant]['trader']:
            if participants_dict[participant]['trader']['learning'] == True:
                participants_dict[participant]['Q_space'] = {}
                participants_dict[participant]['Q_space']['bid_price'] = np.linspace(start=prices[0], stop=prices[1],
                                                                                     num=action_space_separation)

                Q_dict = {'bid_price': [0] * action_space_separation}
                participants_dict[participant]['Q'] = []
                for ts in range(len(timesteps)):
                    participants_dict[participant]['Q'].append(Q_dict)
    return participants_dict
# ----------------------------------------------------------------------------------------------------------------------
# Metrics stuff for temination
def Wasserstein(x:list, y:list):
    x_cumsum = np.cumsum(x)
    y_cumsum = np.cumsum(y)
    mass_delta = sum(abs(x_cumsum-y_cumsum))
    return mass_delta

def _calculate_Q_metrics(old_Q, new_Q):
    metrics={}
    for participant in old_Q:
        if participant in new_Q:
            metrics[participant] = {}

            for action in old_Q[participant][0]:
                if action in new_Q[participant][0]:
                    metrics[participant][action] = []

            for x, y in zip(old_Q[participant], new_Q[participant]):
                for action in x:
                    if action in y:
                        metrics[participant][action].append(Wasserstein(x[action], y[action]))

            for action in metrics[participant]:
                metrics[participant][action] = sum(metrics[participant][action])
                print(metrics[participant][action])

def _get_delta_G(participants_dict, old_G=None, reward_fun=None):
    new_G = {}
    for participant in participants_dict:
        if 'learning' in participants_dict[participant]['trader']:
            if participants_dict[participant]['trader']['learning'] == True:
                new_G[participant] = get_G(participants_dict,
                                           reward_fun=reward_fun,
                                           learning_agent=participant,
                                           grid_transaction_dummy=grid_transaction_dummy)
    print('G:', new_G)

    if old_G is None:
        print(new_G)
        return None, new_G
    else:
        delta_G = {}
        print(new_G, old_G)
        for participant in new_G:
            if participant in old_G:
                delta_G[participant] = new_G[participant] - old_G[participant]
        print(delta_G)
        return delta_G, old_G
# def _fetch_reward_function(participants_dict):
#     #ToDo: once config has reward type, automatically adjust!
#     sys.path.insert(0, 'E:/TREX-Core/')
#     from _agent._rewards.economic_advantage import Reward
#     reward = Reward()
#     return reward
# ----------------------------------------------------------------------------------------------------------------------
# the actual code
# Get the data
    # acess a simulation database
    # import the profiles of all participants

db_path1 = 'postgresql://postgres:postgres@stargate/remote_agent_test_np'
table_list = _get_tables(create_engine(db_path1))

agent_id = 'egauge19821'
sim_type = 'training'
start_gen = 0
action_space_separation = 10

extractor = Extractor(db_path1)
exp_config = extractor.extract_config()
_check_config(exp_config)

len_sim_steps = 1 + exp_config['data']['study']['days']*24*60

min_price = exp_config['data']['market']['grid']['price']
max_price = min_price * (1 + exp_config['data']['market']['grid']['fee_ratio'])
grid_transaction_dummy = (None, max_price, None, min_price)
Q_matrix = np.zeros(shape=[action_space_separation])

participants_dict = exp_config['data']['participants']
participants_dict = _add_metrics_to_participants(participants_dict)

# Test the core components of the algorithm for consistency with the sim
# no point in doing all this if its not consistent
market_df = extractor.from_market(start_gen, sim_type)
# _test_settlement_process(participants_dict, agent_id, market_df)

reward_fun = Reward()
_, old_G = _get_delta_G(participants_dict, old_G=None, reward_fun=reward_fun)

# ToDo: add other functionalities than simply bid-price, but lets test with this first!
t_start = min(market_df['time_creation'])
t_end = max(market_df['time_creation'])
timesteps = list(range(t_start, t_end+60, 60))
participants_dict = _add_Q_and_Qspace(participants_dict=participants_dict,
                                      timesteps=timesteps,
                                      prices=(min_price, max_price),
                                      action_space_separation=action_space_separation)

old_Qs = _fetch_Qs(participants_dict)
participants_dict = _full_Q_iteration(participants_dict)
new_Qs = _fetch_Qs(participants_dict)


_calculate_Q_metrics(old_Q=old_Qs, new_Q=new_Qs)
delta_G, new_G = _get_delta_G(participants_dict, old_G=old_G, reward_fun=reward_fun)


# participants_dict[participant]['metrics']['actions_dict'] for participant in participants_dict.keys()
# is a pandas df including a dict tho, so we can assume its ordered
# BE AWARE THIS IS A CRUCIAL ASSUMPTION!
# actions = {
#     'bess': {
#         time_interval: scheduled_qty
#     },
#     'bids': {
#         time_interval: {
#             'quantity': qty,
#             'source': source,
#             'price': dollar_per_kWh
#         }
#     },
#     'asks': {
#         time_interval: {
#             'quantity': qty,
#             'source': source,
#             'price': dollar_per_kWh?
#         }
#     }
# }







