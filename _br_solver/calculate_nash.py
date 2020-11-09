# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _extractor.extractor import Extractor
from sqlalchemy import create_engine
import numpy as np
import copy

from _utils.market_simulation import sim_market
from _utils.rewards_proxy import NetProfit_Reward as Reward
from _utils.metrics import Metrics_manager
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ToDo: next steps
# ToDo: 1) adjust reward to full value reward:
#       - create reward calculation
#       - figure out remaining net load to/from grid
#       - write a test to confirm
# ToDO: 2) try battery sim for the baselines sim
#       - a) only bid prices, quantity = net-load + whatever's missing in the battery
#           - write a battery logic that does discharge when grid-load > 0
#       - b) bid pricees and battery action (charge, discharge, etc), quantity = net-load + max(charge, 0)
#       --> G_a < G_b, we'll have an estimmate of time increasee due to battery
#
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# tests
# config small probe
def _check_config(config):
    if 'data' not in config.keys():
        print('missing key <data> in experiment config file')

    if 'study' not in config['data'].keys():
        print('missing sub-key <study>')

    if 'start_datetime' not in config['data']['study'].keys():
        print('missing one of the sub^2-keys <start_datetime> or <days>')

# market equality test, the goal is to have the simulated market for the imported participants equal the market database records
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

    market_sim_df = sim_market(participants, learning_agent_id)

    sim_bids, sim_asks = _get_market_records_for_agent(learning_agent_id, market_sim_df)
    db_bids, db_asks = _get_market_records_for_agent(learning_agent_id, market_df)
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
# ----------------------------------------------------------------------------------------------------------------------
# all the stuff we need to calculate returns
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

def _extract_grid_transactions(market_ledger, metrics, grid_transaction_dummy, ts):
    sucessfull_bids = sum([sett[1] for sett in market_ledger if sett[0] == 'bid'])
    sucessfull_asks = sum([sett[1] for sett in market_ledger if sett[0] == 'ask'])
    net_influx = metrics['next_settle_generation'][ts] + sucessfull_bids
    net_outflux = metrics['next_settle_load'][ts] + sucessfull_asks 
    grid_load = net_outflux - net_influx

    return (max(0, grid_load), grid_transaction_dummy[1], max(0, -grid_load), grid_transaction_dummy[3])

def get_G(participants_dict, reward_fun, learning_agent, grid_transaction_dummy:tuple):
    # learning_agent_market_interactions = market_df[
    #     (market_df['seller_id'] == learning_agent) & (market_df['buyer_id'] != 'grid') | (
    #             market_df['buyer_id'] == learning_agent) & (market_df['seller_id'] != 'grid')]
    rewards = []
    #ToDo: change the way we index here?
    print(participants_dict[learning_agent]['metrics']['reward'].shape[0])
    for ts in range(participants_dict[learning_agent]['metrics']['reward'].shape[0]):
        r = _query_market_get_reward_for_one_tuple(participants_dict=participants_dict,
                                                   ts=ts,
                                                   learning_participant=learning_agent,
                                                   reward_fun=reward_fun,
                                                   grid_transaction_dummy=grid_transaction_dummy)
        rewards.append(r)
    return sum(rewards)

# ----------------------------------------------------------------------------------------------------------------------
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

def _add_metrics_to_participants(participants_dict, extractor, start_gen, sim_type):
    for participant in participants_dict.keys():
        participant_dict = participants_dict[participant]

        if 'track_metrics' not in participant_dict['trader']:
            print('no <track_metrics> in participant dict!')

        if participant_dict['trader']['track_metrics'] == True:
            participants_dict[participant]['metrics'] = extractor.from_metrics(start_gen, sim_type, participant)

    return participants_dict

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Value iteration stuff
# get rewards for one action tuple

def _query_market_get_reward_for_one_tuple(participants_dict, ts, learning_participant, reward_fun, grid_transaction_dummy):
    market_df = sim_market(participants=participants_dict, learning_agent_id=learning_participant, timestep=ts)
    market_ledger = []
    for index in range(market_df.shape[0]):
        settlement = market_df.iloc[index]
        # print(settlement)
        market_ledger.append(_map_market_to_ledger(settlement, learning_participant))

    grid_transactions = _extract_grid_transactions(market_ledger=market_ledger,
                                                    metrics=participants_dict[learning_participant]['metrics'],
                                                    grid_transaction_dummy=grid_transaction_dummy,
                                                    ts=ts)

    rewards = reward_fun.calculate(market_transactions=market_ledger, grid_transactions=grid_transactions)
    # print('market', market_ledger)
    # print('grid', grid_transactions)
    # print('r', rewards)
    # print('metered_r', participants_dict[learning_participant]['metrics']['reward'][ts])
    return rewards

# get the bootstrap Q
def _get_bootstrap_Q(Q_array, learning_participant_dict, ts):
    # find Q* previous ts
    Q_max = {}
    steps = len(learning_participant_dict['Q']) #gets the max timestamp
    for action_type in learning_participant_dict['Q'][ts]:
        # ask for max ts key first
        # check if key exists, if not error warning
        # fetch Q_max
        if ts +1 == steps:
            print('hit end')
            Q_max[action_type] = 0.0
        else:
            Q_max[action_type] = max(Q_array[ts+1][action_type])

    return Q_max

# sweep actions, collect updated Q's
# returns one Q_value dictionary
# ToDo: start implementing other action combos instead of only bid-prices
def _action_sweep(participants_dict, ts, learning_participant, bootstrap_Q, reward_fun, grid_transaction_dummy):
    # might be better to combine those somehow
    actions = participants_dict[learning_participant]['Q_space'].keys()


    ns_load = participants_dict[learning_participant]['metrics']['next_settle_load'][ts]
    ns_gen = participants_dict[learning_participant]['metrics']['next_settle_generation'][ts]
    ns_net_load = ns_load - ns_gen


    if 'bids' not in participants_dict[learning_participant]['metrics']['actions_dict'][ts]:
        print('found missing bids, autocomplete not implemented yet')
        #ToDo: autocomplete bids if and when necessary

    key = list(participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'].keys())[0]


    bid_price_Q = []
    if 'bid_price' in actions:
        for idx in range(len(participants_dict[learning_participant]['Q_space']['bid_price'])):

            bid_price = participants_dict[learning_participant]['Q_space']['bid_price'][idx]
            participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'][key]['price'] = bid_price
            r = _query_market_get_reward_for_one_tuple(participants_dict, ts, learning_participant, reward_fun, grid_transaction_dummy)
            bid_price_Q.append(r + bootstrap_Q['bid_price'])

    bid_quant_Q = []
    if 'bid_quantity' in actions:
        for idx in range(len(participants_dict[learning_participant]['Q_space']['bid_quantity'])):

            bid_quant = participants_dict[learning_participant]['Q_space']['bid_quantity'][idx]
            participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'][key]['quantity'] = bid_quant
            r = _query_market_get_reward_for_one_tuple(participants_dict, ts, learning_participant, reward_fun,
                                                       grid_transaction_dummy)
            bid_quant_Q.append(r + bootstrap_Q['bid_quantity'])


    if 'bid_price' in actions and 'bid_quantity' in actions:
        return {'bid_price': bid_price_Q,
                'bid_quantity': bid_quant_Q}

    elif 'bid_price' in actions and 'bid_quantity' not in actions:
        return {'bid_price': bid_price_Q}

    elif 'bid_price' not in actions and 'bid_quantity' in actions:
        return {'bid_quantity': bid_quant_Q}


# perform Q value iteration for one participant,
# returns updated list of Q_value dictionaries
def _Q_iterate_participant(participants_dict, learning_participant, reward_fun, grid_transaction_dummy):
    timesteps = len(participants_dict[learning_participant]['Q'])
    _dict = participants_dict.copy()

    Q_array = [np.nan]*(timesteps)
    for ts in reversed(range(timesteps)):
        bootstrap_Q = _get_bootstrap_Q(Q_array, participants_dict[learning_participant], ts)
        bidQ = _action_sweep(participants_dict, ts, learning_participant, bootstrap_Q, reward_fun, grid_transaction_dummy)
        Q_array[ts] = bidQ
    return Q_array

# update greedy strategy based on Q_values for a learning participant
# returns a behavior dictionary
def _update_to_new_policy(participants_dict, learning_participant=None):
    new_policy = []
    learner_dict = participants_dict[learning_participant]
    for ts in range(len(learner_dict['metrics']['actions_dict'])):
        entry = {}
        for action in learner_dict['metrics']['actions_dict'][ts].keys():
            entry[action] = {}

            if action == 'bids':
                old_policy = learner_dict['metrics']['actions_dict'][ts]['bids']
                ts_key = list(learner_dict['metrics']['actions_dict'][ts]['bids'].keys())[0]
                old_policy = old_policy[ts_key]


                if 'bid_price' in learner_dict['Q_space']:
                    best_action_index = np.argmax(learner_dict['Q'][ts]['bid_price'])
                    best_price = learner_dict['Q_space']['bid_price'][best_action_index]
                    old_policy['price'] = best_price

                if 'bid_quantity' in learner_dict['Q_space']:
                    best_action_index = np.argmax(learner_dict['Q'][ts]['bid_quantity'])
                    best_price = learner_dict['Q_space']['bid_quantity'][best_action_index]
                    old_policy['quantity'] = best_price

                entry[action][ts_key]= old_policy
        new_policy.append(entry)
    return new_policy

# perform value iteration for all participants where applicable one full time
# returns the full participants dict
def _full_Q_iteration(participants_dict, reward_fun, grid_transaction_dummy):
    for participant in participants_dict:
        if 'learning' in participants_dict[participant]['trader']:
            if participants_dict[participant]['trader']['learning'] == True:
                Q_updated_participant = _Q_iterate_participant(participants_dict, learning_participant=participant, reward_fun=reward_fun, grid_transaction_dummy=grid_transaction_dummy)
                del participants_dict[participant]['Q']
                participants_dict[participant]['Q'] = Q_updated_participant
                participants_dict[participant]['metrics']['actions_dict'] = _update_to_new_policy(participants_dict, learning_participant=participant)

    return participants_dict

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# helper to fetch Q
# returns a Q_dictionary, dict[participant][Q] = Q
def _fetch_Qs(participants_dict):
    for participant in participants_dict:
        if 'learning' in participants_dict[participant]['trader']:
            participants_old_Q = {}
            if participants_dict[participant]['trader']['learning'] == True:
                participants_old_Q[participant] = participants_dict[participant]['Q']

    return participants_old_Q


# calculate return G
# returns a G dictionary dict[participant] = G
def _calculate_G(participants_dict, reward_fun, grid_transaction_dummy):
    new_G = {}
    for participant in participants_dict:
        if 'learning' in participants_dict[participant]['trader']:
            if participants_dict[participant]['trader']['learning'] == True:
                new_G[participant] = get_G(participants_dict,
                                           reward_fun=reward_fun,
                                           learning_agent=participant,
                                           grid_transaction_dummy=grid_transaction_dummy)
    return new_G

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# helper functions for setup of value iteration
# add Q values and Qspace to the participants dictionary
def _add_Q_and_Qspace(participants_dict:dict, timesteps:int=10, prices:tuple=(0,1), action_space_separation:int=10):
    for participant in participants_dict:
        if 'learning' in participants_dict[participant]['trader']:
            if participants_dict[participant]['trader']['learning'] == True:
                participants_dict[participant]['Q_space'] = {}
                participants_dict[participant]['Q_space']['bid_price'] = np.linspace(start=prices[0], stop=prices[1],
                                                                                     num=action_space_separation)

                participants_dict[participant]['Q_space']['bid_quantity'] = np.linspace(start=0, stop=10,
                                                                                     num=action_space_separation)

                Q_dict = {'bid_price': [0.0] * action_space_separation,
                          'bid_quantity': [0.0] * action_space_separation
                          }
                participants_dict[participant]['Q'] = []
                for ts in range(len(timesteps)):
                    participants_dict[participant]['Q'].append(Q_dict)
    return participants_dict

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# actual nash solver
# beware, contains a while loop!
#ToDo: keep working and making more flexible
def _test_G(calculated_Gs, participants_dict):
    for participant in participants_dict:
        if 'learning' in participants_dict[participant]['trader']:
            if participants_dict[participant]['trader']['learning'] == True:
                calculated_G = calculated_Gs[participant]
                metered_G = sum(participants_dict[participant]['metrics']['reward'][1:])
                print(calculated_G, metered_G)


def solve_for_nash(sim_db_path='postgresql://postgres:postgres@stargate/remote_agent_test_np',
                    agent_id = 'egauge19821',
                    sim_type = 'training',
                    start_gen = 0,
                    action_space_separation = 10,
                   ):

    extractor = Extractor(sim_db_path)
    exp_config = extractor.extract_config()
    _check_config(exp_config)

    participants_dict = exp_config['data']['participants']
    participants_dict = _add_metrics_to_participants(participants_dict, extractor, start_gen, sim_type)

    market_df = extractor.from_market(start_gen, sim_type)
    _test_settlement_process(participants_dict, agent_id, market_df)

    reward_fun = Reward()
    min_price = exp_config['data']['market']['grid']['price']
    max_price = min_price * (1 + exp_config['data']['market']['grid']['fee_ratio'])
    grid_transaction_dummy = (None, max_price, None, min_price)
    initial_G = _calculate_G(participants_dict, reward_fun=reward_fun, grid_transaction_dummy=grid_transaction_dummy)
    _test_G(initial_G, participants_dict)

    timesteps = list(range(min(market_df['time_creation']), max(market_df['time_creation'])+60, 60))

    participants_dict = _add_Q_and_Qspace(participants_dict=participants_dict,
                                          timesteps=timesteps,
                                          prices=(min_price, max_price),
                                          action_space_separation=action_space_separation)


    initial_Qs = _fetch_Qs(participants_dict)
    metrics = Metrics_manager(initial_Qs=initial_Qs, calculate_Q_metrics=True,
                              initial_G=initial_G, calculate_G_metrics=True)

    summed_delta = np.inf
    summed_wasser = np.inf

    while summed_delta > 1e-3 or summed_wasser > 1.0:

        participants_dict = _full_Q_iteration(participants_dict, reward_fun, grid_transaction_dummy)

        metrics_history = metrics.calculate_metrics(new_Qs=_fetch_Qs(participants_dict),
                                                    new_G=_calculate_G(participants_dict, reward_fun, grid_transaction_dummy),
                                                    return_metrics_history=True)

        summed_delta = 0
        for participant in metrics_history:
            summed_delta += abs(metrics_history[participant]['delta_G'][-1])

        summed_wasser = 0
        for participant in metrics_history:
            for action in metrics_history[participant]:
                if action == 'bid_price' or action == 'bid_quantity' or action == 'ask_price' or action == 'ask_quantity':
                    summed_wasser += metrics_history[participant][action]['Wasserstein'][-1]

        print(metrics_history)



if __name__ == "__main__":
    solve_for_nash(sim_db_path='postgresql://postgres:postgres@stargate/remote_agent_test_np',
                    agent_id = 'egauge19821',
                    sim_type = 'training',
                    start_gen = 0,
                    action_space_separation = 10,)



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







