# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _extractor.extractor import Extractor
from sqlalchemy import create_engine
import numpy as np
from operator import itemgetter

def _get_settlement(opponent_actions_ts, learner_action):
    # Learner_action:
    # {'timestamp': ts,
    #  'action': 'bids', (or 'asks')
    #  'price': number,
    #  'quantity': number}
    # step 1: extract all opponent actions for timestep
    best_settle = []
    for action_type in learner_action.keys():
        settle_ts = list(learner_action[action_type].keys())[0]
    # decide what to do based on net load and allowable actions
    # if net load > 0 and allowed to bid, then find best settle for bids
    # if net load < 0 and allowed to ask, then find best settle for asks
    # compile settlements in a format usable by reward func

    # reward calc for market transaction is a list of tuples
    # each tuple follows the following format:
    # (action ('bid' or 'ask'), quantity, price, source))
    # best settle for market should follow this format

    if 'bids' in learner_action.keys():
        # filter out opponent actions by asks
        # sort price from lowest to highest (ascending order)
        oa_unsorted = [action['asks'] for action in opponent_actions_ts if 'asks' in action.keys()]
        oa_sorted = sorted([action[settle_ts] for action in oa_unsorted if settle_ts in action.keys()],
                           key=itemgetter('price'), reverse=False)


        # iterate over the sorted opponent actions and create settlements until bid quantity is fully satisfied
        for opponent_action in oa_sorted:

            if learner_action['bids'][settle_ts]['quantity'] <= 0:
                break

            # create settlement tuple and decrease remaining quantity:
            # transaction only happens if abs(action price) >= abs(opponent price)
            if learner_action['bids'][settle_ts]['price'] < opponent_action['price']:
                break

            # settlement price is still the average of bid and ask prices
            settle_price = (learner_action['bids'][settle_ts]['price'] + opponent_action['price']) / 2
            settle_qty = min(learner_action['bids'][settle_ts]['quantity'], opponent_action['quantity'])
            best_settle.append(('bid', settle_qty, settle_price, opponent_action['source']))
            learner_action['bids'][settle_ts]['quantity'] -= settle_qty

    if 'asks' in learner_action.keys():
        # filter out opponent actions by positive quantities
        # sort price from highest to lowest (descending order)
        oa_unsorted = [action['bids'] for action in opponent_actions_ts if 'bids' in action.keys()]
        oa_sorted = sorted([action[settle_ts] for action in oa_unsorted if settle_ts in action.keys()],
                           key=itemgetter('price'), reverse=False)


        # iterate over the sorted opponent actions and create settlements until ask quantity is fully satisfied
        for opponent_action in oa_sorted:
            if learner_action['asks'][settle_ts]['quantity'] <= 0:
                break

            # create settlement tuple and decrease remaining quantity:
            # transaction only happens if abs(action price) <= abs(opponent price)
            if learner_action['asks'][settle_ts]['price'] > opponent_action['price']:
                break

            settle_price = (learner_action['asks'][settle_ts]['price'] + opponent_action['price']) / 2
            settle_qty = min(learner_action['asks'][settle_ts]['quantity'], opponent_action['quantity'])
            best_settle.append(('asks', settle_qty, settle_price, opponent_action['source']))
            learner_action['asks'][settle_ts]['quantity'] -= settle_qty

    return best_settle

def _check_config(config):
    if 'data' not in config.keys():
        print('missing key <data> in experiment config file')

    if 'study' not in config['data'].keys():
        print('missing sub-key <study>')

    if 'start_datetime' not in config['data']['study'].keys():
        print('missing one of the sub^2-keys <start_datetime> or <days>')

def get_tables(engine):
    find_profile_names = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
    """
    table = pd.read_sql_query(find_profile_names, engine) # get tables

    return [element[0] for element in table.values.tolist()] #unpack, because list of lists

# Get the data
    # acess a simulation database
    # import the profiles of all participants

db_path1 = 'postgresql://postgres:postgres@stargate/remote_agent_test_np'
table_list = get_tables(create_engine(db_path1))

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
#ToDo: Do not forget to add the check for each trader if they're learning!
for participant in participants_dict.keys():
    participant_dict = participants_dict[participant]

    if 'track_metrics' not in participant_dict['trader']:
        print('no <track_metrics> in participant dict!')

    if participant_dict['trader']['track_metrics'] == True:
        participants_dict[participant]['metrics'] = extractor.from_metrics(start_gen, sim_type, participant)

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
for participant in participants_dict.keys():
    for timestep in range(len(participants_dict[participant]['metrics']['actions_dict'])):
        opponents_actions = [participants_dict[opponent]['metrics']['actions_dict'][timestep] for opponent in participants_dict.keys() if opponent != participant]
        learner_action = participants_dict[participant]['metrics']['actions_dict'][timestep]

        settlement = _get_settlement(opponent_actions_ts=opponents_actions,
                                     learner_action=learner_action)

        print(settlement)



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


