# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _extractor.extractor import Extractor
from sqlalchemy import create_engine
import numpy as np
import copy

from _utils.market_simulation import _get_settlement, match
# import _utils.market_simulation

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

def _sim_market(participants:dict, learning_agent_id:str):
    learning_agent = participants[learning_agent_id]
    # opponents = copy.deepcopy(participants)
    # opponents.pop(learning_agent_id, None)
    open = {}
    learning_agent_times_delivery = []
    market_sim_df = []

    for idx in range(len(learning_agent['metrics']['actions_dict'])):
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

min_price = exp_config['data']['market']['grid']['price']
max_price = min_price * (1 + exp_config['data']['market']['grid']['fee_ratio'])
action_space = np.linspace(start=min_price, stop=max_price, num=action_space_separation)

participants_dict = exp_config['data']['participants']
participants_dict = _add_metrics_to_participants(participants_dict)

market_df = extractor.from_market(start_gen, sim_type)

_test_settlement_process(participants_dict, agent_id, market_df)


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








# Import the necessities to construct the MDP:
    # Import market settlement mechanism
        # ToDo: Figure this out, how do we import this from the old code?
    # Import the rewards
        # ToDo: Figure this out, how do we use rewards here??
# Construct an MPD
    # construct an MDP for each of the participants
    # import an action space for those participants
        # that action space must be discrete for now!
    # load the participants action history into the MDP

# until convergence
    # assessed by:
        # changes in return of each participant
        # changes in the policy of each participant (Wasserstein) --> how do we deal with multiple pi_*
        # an iteration ceiling

    # for each LEARNING participant
        # (partially?) determine best response for participant wrt other participants
            # for no-EES sim:
                # backward pass of Value iteration
                # immediate convergence
                # construct pi_*
                    # if pi_* is multiple, randomly choose from best

    # do some metrics and display (see assessment criteria)

# --> we should now have a system that is converged to a Nash Equilibrium

# Save this as a separate table into the sim database (we know which one it is anyways)


