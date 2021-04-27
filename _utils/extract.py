import dataset
from sqlalchemy import and_, or_
import pandas as pd

def extract_trades_data(db, table, participant_id, time_consumption_interval:tuple):
    statement = table.select().where(and_(and_(table.c.time_consumption >= time_consumption_interval[0],
                                               table.c.time_consumption <= time_consumption_interval[1]),
                                          or_(table.c.buyer_id == participant_id,
                                              table.c.seller_id == participant_id)))

    transactions = db.query(statement)
    trades = {}
    for transaction in transactions:
        if (transaction['seller_id'], transaction['buyer_id']) not in trades:
                trades[(transaction['seller_id'], transaction['buyer_id'])] = {'settlement_price': transaction['settlement_price'],
                         'quantity': transaction['quantity']}
        else:
            prev_quantity = trades[(transaction['seller_id'], transaction['buyer_id'])]['quantity']
            prev_price = trades[(transaction['seller_id'], transaction['buyer_id'])]['settlement_price']
            added_quantity = transaction['quantity']
            added_price = transaction['settlement_price']

            cum_quantity = prev_quantity + added_quantity
            avg_price = (prev_price * prev_quantity + added_quantity * added_price)/cum_quantity
            trades[(transaction['seller_id'], transaction['buyer_id'])]['quantity'] = cum_quantity
            trades[(transaction['seller_id'], transaction['buyer_id'])]['settlement_price'] = avg_price

    return trades

def transaction_summary(db, table, participant_id, time_consumption_interval:tuple):
    """

    """
    statement = table.select().where(and_(and_(table.c.time_consumption >= time_consumption_interval[0],
                                               table.c.time_consumption <= time_consumption_interval[1]),
                                          or_(table.c.buyer_id == participant_id,
                                              table.c.seller_id == participant_id)))

    transactions = db.query(statement)
    results = {}

    for transaction in transactions:
        time_consumption = transaction['time_consumption']
        if not time_consumption:
            # only track physical transactions for now
            continue
        if time_consumption not in results:
            results[time_consumption] = {
                'generation_total': 0,
                'generation_to_community': 0,
                'consumption_total': 0,
                'consumption_from_community': 0,
                'self_consumption': 0,
                'profit': 0,
                'cost': 0
            }
        summary = results[time_consumption]
        seller = transaction['seller_id']
        buyer = transaction['buyer_id']

        quantity = transaction['quantity'] / 1000 #convert from Wh to kWh
        # source = transaction['energy_source']
        price = transaction['settlement_price'] #price is in $/kWh

        if buyer == seller:
            # self consumption
            summary['generation_total'] += quantity
            summary['consumption_total'] += quantity
            summary['self_consumption'] += quantity

        elif participant_id == buyer:
            summary['consumption_total'] += quantity
            summary['cost'] += price * quantity
            if seller != 'grid':
                summary['consumption_from_community'] += quantity
        else:
            summary['generation_total'] += quantity
            summary['profit'] += price * quantity
            if buyer != 'grid':
                summary['generation_to_community'] += quantity

    return pd.DataFrame.from_dict(results, orient='index').sort_index()

def transactions(db, table, time_consumption_interval:tuple):
    """

    """
    # if exclude_self_consumption:
    #     statement = table.select().where(and_(table.c.time_consumption >= time_consumption_interval[0],
    #                                           table.c.time_consumption <= time_consumption_interval[1],
    #                                           table.c.settlement_price != 0))
    # else:
    statement = table.select().where(and_(table.c.time_consumption >= time_consumption_interval[0],
                                          table.c.time_consumption <= time_consumption_interval[1]))
    transactions = db.query(statement)
    # df = pd.DataFrame(transactions).set_index('time_consumption').sort_index()
    # df.set_index('time_consumption')
    # df
    return pd.DataFrame(transactions)

def self_consumption(db, table, time_consumption_interval:tuple):
    """

    """
    # if exclude_self_consumption:
    #     statement = table.select().where(and_(table.c.time_consumption >= time_consumption_interval[0],
    #                                           table.c.time_consumption <= time_consumption_interval[1],
    #                                           table.c.settlement_price != 0))
    # else:
    statement = table.select().where(and_(table.c.time_consumption >= time_consumption_interval[0],
                                          table.c.time_consumption <= time_consumption_interval[1],
                                          table.c.settlement_price == 0))
    transactions = db.query(statement)
    # df = pd.DataFrame(transactions).set_index('time_consumption').sort_index()
    # df.set_index('time_consumption')
    # df
    return pd.DataFrame(transactions)