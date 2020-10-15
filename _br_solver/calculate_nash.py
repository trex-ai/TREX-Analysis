# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _extractor.extractor import Extractor
from sqlalchemy import create_engine
import numpy as np

from _br_solver.market_simulation import _get_settlement


def _check_config(config):
    if 'data' not in config.keys():
        print('missing key <data> in experiment config file')

    if 'study' not in config['data'].keys():
        print('missing sub-key <study>')

    if 'start_datetime' not in config['data']['study'].keys():
        print('missing one of the sub^2-keys <start_datetime> or <days>')

def _get_tables(engine):
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


