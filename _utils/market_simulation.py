from operator import itemgetter
import itertools

# pretend market settlement
# takes a single time-slice of actions for participants, separated into 'learner' and opponents
# performs a settlement
# returns the settlement for the learner
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