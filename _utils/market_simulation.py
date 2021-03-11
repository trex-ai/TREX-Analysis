from operator import itemgetter
import pandas as pd
import numpy as np
import itertools

# pretend market settlement
# simulated market for participants, giving back learning agent's settlements, optionally for a specific timestamp
def sim_market(participants:dict, learning_agent_id:str, row:int=None):
    learning_agent = participants[learning_agent_id]
    # opponents = copy.deepcopy(participants)
    # opponents.pop(learning_agent_id, None)
    open = {}
    learning_agent_times_delivery = []
    market_sim_df = []
    if row == None:
        row = range(len(learning_agent['metrics']['actions_dict']))
    else:
        row = [row]

    for idx in row:
        for participant_id in participants:
            agent_actions = participants[participant_id]['metrics']['actions_dict'][idx]
            for action in ('bids', 'asks'):
                if action in agent_actions:
                    # print(open)
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
    # print(open)
    for t_d in learning_agent_times_delivery:
        if 'bids' in open[t_d] and 'asks' in open[t_d]:
            market_sim_df.extend(match(open[t_d]['bids'], open[t_d]['asks'], 'solar', t_d))

    return pd.DataFrame(market_sim_df)

def match(bids, asks, source_type, time_delivery):
    settled = []

    bids = sorted([bid for bid in bids if (bid['quantity'] > 0 and bid['source'] == source_type)], key=itemgetter('price'), reverse=True)
    asks = sorted([ask for ask in asks if (ask['quantity'] > 0 and ask['source'] == source_type)], key=itemgetter('price'), reverse=False)

    for bid, ask, in itertools.product(bids, asks):
        if ask['price'] > bid['price']:
            continue

        if bid['participant_id'] == ask['participant_id']:
            continue

        if bid['source'] != ask['source']:
            continue

        if bid['quantity'] <= 0 or ask['quantity'] <= 0:
            continue

        # Settle highest price bids with lowest price asks
        settle_record = settle(bid, ask, time_delivery)
        if settle_record:
            settled.append(settle_record)

    return settled

def settle(bid: dict, ask: dict, time_delivery: tuple, settlement_price=None):
    if bid['source'] == 'grid' and ask['source'] == 'grid':
        return

    # only proceed to settle if settlement quantity is positive
    quantity = min(bid['quantity'], ask['quantity'])
    if quantity <= 0:
        return

    if not settlement_price:
        settlement_price = (ask['price'] + bid['price']) / 2

    record = {
        'quantity': quantity,
        'seller_id': ask['participant_id'],
        'buyer_id': bid['participant_id'],
        'energy_source': ask['source'],
        'settlement_price': settlement_price,
        'time_delivery': time_delivery
    }
    return record

# testing stuff
# market equality test, the goal is to have the simulated market for the imported participants equal the market database records
def _get_market_records_for_agent(participant:str,
                                  market_df):
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
def _compare_records(market_sim_df, #compares an extracted market_df to a simulated market_df
                     market_db_df):

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

def _test_settlement_process(participants_dict:dict,    #the imported metrics from the database
                             learning_agent_id:str,     # the agent to check for, one should be enough
                             market_df):                # the extracted market dataframe

    market_sim_df = sim_market(participants_dict, learning_agent_id)

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
def _map_market_to_ledger(market_df_ts, # one timemslice of the market dataframe
                          learning_agent):

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