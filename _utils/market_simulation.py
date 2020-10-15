from operator import itemgetter
import itertools

# pretend market settlement
# takes a single time-slice of actions for participants, separated into 'learner' and opponents
# performs a settlement
# returns the settlement for the learner
def _get_settlement(opponent_actions_ts, learner_action):
    if not opponent_actions_ts:
        print('oa', opponent_actions_ts, 'la', learner_action)
    # actions:
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
                continue

            # create settlement tuple and decrease remaining quantity:
            # transaction only happens if abs(action price) >= abs(opponent price)
            if learner_action['bids'][settle_ts]['price'] < opponent_action['price']:
                continue

            # settlement price is still the average of bid and ask prices
            settle_price = (learner_action['bids'][settle_ts]['price'] + opponent_action['price']) / 2
            settle_qty = min(learner_action['bids'][settle_ts]['quantity'], opponent_action['quantity'])
            best_settle.append(('bid', settle_qty, settle_price, opponent_action['source'], settle_ts))
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
                continue

            # create settlement tuple and decrease remaining quantity:
            # transaction only happens if abs(action price) <= abs(opponent price)
            if learner_action['asks'][settle_ts]['price'] > opponent_action['price']:
                continue

            settle_price = (learner_action['asks'][settle_ts]['price'] + opponent_action['price']) / 2
            settle_qty = min(learner_action['asks'][settle_ts]['quantity'], opponent_action['quantity'])
            best_settle.append(('ask', settle_qty, settle_price, opponent_action['source'], settle_ts))
            learner_action['asks'][settle_ts]['quantity'] -= settle_qty

    return best_settle

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