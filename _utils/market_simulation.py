from operator import itemgetter
import pandas as pd
import itertools

# pretend market settlement
# simulated market for participants, giving back learning agent's settlements, optionally for a specific timestamp
def sim_market(participants:dict, learning_agent_id:str, timestep:int=None):
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