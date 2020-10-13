# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _extractor.extractor import Extractor
from sqlalchemy import create_engine


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

extractor = Extractor(db_path1)
agent_market_df = extractor.from_metrics(start_gen, sim_type, agent_id)

# engine = create_engine('postgresql://postgres:postgres@stargate/profiles')


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

    # for each participant
        # (partially?) determine best response for participant wrt other participants
            # for no-EES sim:
                # backward pass of Value iteration
                # immediate convergence
                # construct pi_*
                    # if pi_* is multiple, randomly choose from best

    # do some metrics and display (see assessment criteria)

# --> we should now have a system that is converged to a Nash Equilibrium

# Save this as a separate table into the sim database (we know which one it is anyways)


