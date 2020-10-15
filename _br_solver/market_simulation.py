from operator import itemgetter

# pretend market settlement
# takes a single time-slice of actions for participants, separated into 'learner' and opponents
# performs a settlement
# returns the settlement for the learner
def _get_settlement(opponent_actions_ts, learner_action):
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
