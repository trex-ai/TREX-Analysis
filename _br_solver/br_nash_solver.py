# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _extractor.extractor import Extractor
import numpy as np
import copy

from _utils.market_simulation import sim_market, _test_settlement_process, _map_market_to_ledger
from _utils.rewards_proxy import NetProfit_Reward as Reward
from _utils.metrics import Metrics_manager
from _utils.Bess import Storage
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# general stuff we need for this to work
# def _get_tables(engine):
#     find_profile_names = """
#     SELECT table_name
#     FROM information_schema.tables
#     WHERE table_schema = 'public'
#     ORDER BY table_name
#     """
#     table = pd.read_sql_query(find_profile_names, engine) # get tables
#
#     return [element[0] for element in table.values.tolist()] #unpack, because list of lists

def _add_metrics_to_participants(participants_dict, extractor, start_gen, sim_type):
    for participant in participants_dict.keys():
        participant_dict = participants_dict[participant]

        if 'track_metrics' not in participant_dict['trader']:
            print('no <track_metrics> in participant dict!')

        if participant_dict['trader']['track_metrics'] == True:
            participants_dict[participant]['metrics'] = extractor.from_metrics(start_gen, sim_type, participant)

    return participants_dict

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Value iteration stuff
# get rewards for one action tuple

# get the bootstrap Q


# sweep actions, collect updated Q's
# returns one Q_value dictionary
# ToDo: start implementing other action combos instead of only bid-prices


# perform Q value iteration for one participant,
# returns updated list of Q_value dictionaries


# update greedy strategy based on Q_values for a learning participant
# returns a behavior dictionary

# perform value iteration for all participants where applicable one full time
# returns the full participants dict

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# actual nash solver
# beware, contains a while loop!
#ToDo: keep working and making more flexible


class BR_solver(object):
    def __init__(self,
                 sim_db_path='postgresql://postgres:postgres@stargate/remote_agent_test_np',
                 agent_id='egauge19821',
                 sim_type='training',
                 start_gen=0,
                 ):

        extractor = Extractor(sim_db_path)
        exp_config = extractor.extract_config()
        if 'data' not in exp_config.keys():
            print('missing key <data> in experiment config file')
        if 'study' not in exp_config['data'].keys():
            print('missing sub-key <study>')
        if 'start_datetime' not in exp_config['data']['study'].keys():
            print('missing one of the sub^2-keys <start_datetime> or <days>')

        self.participants_dict = exp_config['data']['participants']
        self.participants_dict = _add_metrics_to_participants(self.participants_dict, extractor, start_gen, sim_type)

        market_df = extractor.from_market(start_gen, sim_type)
        _test_settlement_process(self.participants_dict, agent_id, market_df)

        self.reward_fun = Reward()
        min_price = exp_config['data']['market']['grid']['price']
        max_price = min_price * (1 + exp_config['data']['market']['grid']['fee_ratio'])
        self.prices = (min_price, max_price)
        self.grid_transaction_dummy = (None, max_price, None, min_price)
        self._test_G()

        self.timesteps = list(range(min(market_df['time_creation']), max(market_df['time_creation']) + 60, 60))

        self.EES = Storage()


    def _test_G(self):
        calculated_Gs = self._calculate_G()
        for participant in self.participants_dict:
            if 'learning' in self.participants_dict[participant]['trader']:
                if self.participants_dict[participant]['trader']['learning'] == True:
                    calculated_G = calculated_Gs[participant]
                    metered_G = sum(self.participants_dict[participant]['metrics']['reward'][1:])
                    print(calculated_G, metered_G)

    def _fetch_Qs(self):
        for participant in self.participants_dict:
            if 'learning' in self.participants_dict[participant]['trader']:
                participants_old_Q = {}
                if self.participants_dict[participant]['trader']['learning'] == True:
                    participants_old_Q[participant] = self.participants_dict[participant]['Q']

        return participants_old_Q

    def _calculate_G(self):
        G = {}
        for participant in self.participants_dict:
            if 'learning' in self.participants_dict[participant]['trader']:
                if self.participants_dict[participant]['trader']['learning'] == True:
                        rewards = []
                        for ts in range(self.participants_dict[participant]['metrics']['reward'].shape[0]):
                            r, _ = self._query_market_get_reward_for_one_tuple(ts=ts, learning_participant=participant)
                            rewards.append(r)
                        G[participant] = sum(rewards)

        return G

    def _extract_grid_transactions(self, market_ledger, learning_participant, grid_transaction_dummy, ts, battery=0.0):
        sucessfull_bids = sum([sett[1] for sett in market_ledger if sett[0] == 'bid'])
        sucessfull_asks = sum([sett[1] for sett in market_ledger if sett[0] == 'ask'])

        net_influx = self.participants_dict[learning_participant]['metrics']['next_settle_generation'][ts] + sucessfull_bids
        net_outflux = self.participants_dict[learning_participant]['metrics']['next_settle_load'][ts] + sucessfull_asks

        grid_load = net_outflux - net_influx + battery

        return (max(0, grid_load), grid_transaction_dummy[1], max(0, -grid_load), grid_transaction_dummy[3])

    def _query_market_get_reward_for_one_tuple(self, ts, learning_participant):
        market_df = sim_market(participants=self.participants_dict, learning_agent_id=learning_participant, timestep=ts)
        market_ledger = []
        quant = 0
        for index in range(market_df.shape[0]):
            settlement = market_df.iloc[index]
            quant = settlement['quantity']
            market_ledger.append(_map_market_to_ledger(settlement, learning_participant))

        # key = list(participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'].keys())[0]
        # bat_action = participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery']
        if 'battery' in self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]:
            if ts == 0:
                battery_soc_previous = 0
            else:
                bat_key = \
                list(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts - 1]['battery'].keys())[
                    0]
                battery_soc_previous = self.participants_dict[learning_participant]['metrics']['actions_dict'][ts - 1]['battery'][bat_key]['SoC']
            bat_key = \
            list(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'].keys())[0]
            battery_amt = self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'][bat_key]['action']

            battery_amt, battery_soc_previous = self.EES.simulate_activity(battery_soc_previous,
                                       energy_activity=battery_amt)

        else:
            battery_amt = 0

        grid_transactions = self._extract_grid_transactions(market_ledger=market_ledger,
                                                       learning_participant=learning_participant,
                                                       grid_transaction_dummy=self.grid_transaction_dummy,
                                                       ts=ts,
                                                       battery=battery_amt)

        rewards = self.reward_fun.calculate(market_transactions=market_ledger, grid_transactions=grid_transactions)
        # print('market', market_ledger)
        # print('grid', grid_transactions)
        # print('r', rewards)
        # print('metered_r', participants_dict[learning_participant]['metrics']['reward'][ts])
        return rewards, quant

    def define_action_space(self,
                            action_space_separation_prices=10,
                            action_space_separation_quantity=30,
                            action_space_separation_bat=20,
                            ):

        for participant in self.participants_dict:
            if 'learning' in self.participants_dict[participant]['trader']:
                if self.participants_dict[participant]['trader']['learning'] == True:
                    Q_dict = {}
                    space_shape = []
                    self.participants_dict[participant]['Q_space'] = {}
                    if action_space_separation_prices is not None:
                        self.participants_dict[participant]['Q_space']['bid_price'] = np.linspace(start=self.prices[0],
                                                                                             stop=self.prices[1],
                                                                                             num=action_space_separation_prices)
                        space_shape.append(action_space_separation_prices)

                    if action_space_separation_quantity is not None:
                        self.participants_dict[participant]['Q_space']['bid_quantity'] = np.linspace(start=0,
                                                                                                stop=action_space_separation_quantity - 1,
                                                                                                num=action_space_separation_quantity,
                                                                                                dtype=int)
                        space_shape.append(action_space_separation_quantity)

                    if action_space_separation_bat is not None:
                        self.participants_dict[participant]['Q_space']['battery'] = np.linspace(
                            start=-int(action_space_separation_bat / 2),
                            stop=int(action_space_separation_bat / 2),
                            num=action_space_separation_bat,
                            dtype=int)
                        space_shape.append(action_space_separation_bat)

                        for ts in range(len(self.participants_dict[participant]['metrics']['actions_dict'])):
                            key = list(self.participants_dict[participant]['metrics']['actions_dict'][ts]['bids'].keys())[0]
                            self.participants_dict[participant]['metrics']['actions_dict'][ts]['battery'] = {}
                            self.participants_dict[participant]['metrics']['actions_dict'][ts]['battery'][key] = {'SoC': 0.0,
                                                                                                             'action': 0.0}
                            self.participants_dict[participant]['metrics']['actions_dict'][ts]['asks'] = {}
                            self.participants_dict[participant]['metrics']['actions_dict'][ts]['asks'][key] = {
                                'price': 0.0,
                                'quantity': 0.0}


                    Q_dict['bids+bat'] = np.zeros(shape=space_shape)
                    self.participants_dict[participant]['Q'] = []
                    for ts in range(len(self.timesteps)):
                        self.participants_dict[participant]['Q'].append(Q_dict)

    # Value iteration stuff
    def _get_bootstrap_Q(self, Q_array, learning_participant, ts):
        # find Q* previous ts
        Q_max = {}
        steps = len(self.participants_dict[learning_participant]['Q'])  # gets the max timestamp
        for action_type in self.participants_dict[learning_participant]['Q'][ts]:
            # ask for max ts key first
            # check if key exists, if not error warning
            # fetch Q_max
            if ts + 1 == steps:
                print('hit end')
                Q_max[action_type] = 0.0
            else:
                Q_max[action_type] = np.amax(self.participants_dict[learning_participant]['Q'][ts + 1][action_type])

        return Q_max

    def _full_Value_iteration(self):
        for participant in self.participants_dict:
            if 'learning' in self.participants_dict[participant]['trader']:
                if self.participants_dict[participant]['trader']['learning'] == True:
                    # Q-iterate
                    _dict = self.participants_dict.copy()

                    Q_array = [np.nan] * (self.timesteps)
                    counter = 0
                    percent_counter = 0
                    for ts in reversed(range(self.timesteps)):
                        bootstrap_Q = self._get_bootstrap_Q(Q_array, participant, ts)
                        self._bootstrap_one_action_sweep(ts, participant, bootstrap_Q)

                        if 100 * counter / self.timesteps > percent_counter:
                            print(100 * counter / self.timesteps, '% done')
                            percent_counter += 5
                        counter += 1

                        self._update_to_greedy_policy(learning_participant=participant, ts=ts)

    def _update_to_greedy_policy(self, learning_participant=None, ts=None):

        for action in self.participants_dict[learning_participant]['metrics']['actions_dict'][ts].keys():

            best_action_index = np.unravel_index(
                np.argmax(self.participants_dict[learning_participant]['Q'][ts]['bids+bat']),
                self.participants_dict[learning_participant]['Q'][ts]['bids+bat'].shape)

            if action =='battery':
                ts_key = \
                list(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'].keys())[0]

                if 'battery' in self.participants_dict[learning_participant]['Q_space']:
                    best_battery_action = self.participants_dict[learning_participant]['Q_space']['battery'][best_action_index[2]]
                    self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'][ts_key]['action'] = best_battery_action

            if action == 'bids':
                ts_key = list(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'].keys())[0]

                if 'bid_price' in self.participants_dict[learning_participant]['Q_space']:
                    best_bid_price = self.participants_dict[learning_participant]['Q_space']['bid_price'][best_action_index[0]]
                    self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'][ts_key]['price'] = best_bid_price

                if 'bid_quantity' in self.participants_dict[learning_participant]['Q_space']:

                    best_bid_price = self.participants_dict[learning_participant]['Q_space']['bid_quantity'][
                        best_action_index[1]]
                    self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'][ts_key][
                        'quantity'] = best_bid_price

    def _bootstrap_one_action_sweep(self, ts, learning_participant, bootstrap_Q):

        if 'bids' not in self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]:
            print('found missing bids, autocomplete not implemented yet')

        price_key = list(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'].keys())[0]
        # print(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'])
        bat_key = list(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'].keys())[0]

        # ns_net_load = self.participants_dict[learning_participant]['metrics']['next_settle_load'][ts] - self.participants_dict[learning_participant]['metrics']['next_settle_generation'][ts]

        for price_idx in range(len(self.participants_dict[learning_participant]['Q_space']['bid_price'])):

            bid_price = self.participants_dict[learning_participant]['Q_space']['bid_price'][price_idx]
            self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'][price_key]['price'] = bid_price

            for quant_idx in range(len(self.participants_dict[learning_participant]['Q_space']['bid_quantity'])):
                bid_quantity = self.participants_dict[learning_participant]['Q_space']['bid_quantity'][quant_idx]
                self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'][price_key][
                    'quantity'] = bid_quantity

                for bat_idx in range(len(self.participants_dict[learning_participant]['Q_space']['battery'])):
                    bat_action = self.participants_dict[learning_participant]['Q_space']['battery'][bat_idx]
                    self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'][bat_key][
                        'action'] = bat_action
                    r, quant = self._query_market_get_reward_for_one_tuple(ts=ts, learning_participant=learning_participant)

                    self.participants_dict[learning_participant]['Q'][ts]['bids+bat'][price_idx, quant_idx, bat_idx] = r + bootstrap_Q['bids+bat']

    # MCTS stuff
    def get_market_ledger(self, ts, learning_participant):
        market_df = sim_market(participants=self.participants_dict, learning_agent_id=learning_participant, timestep=ts)
        market_ledger = []
        quant = 0
        # print(market_df)
        for index in range(market_df.shape[0]):
            settlement = market_df.iloc[index]
            quant += settlement['quantity']
            market_ledger.append(_map_market_to_ledger(settlement, learning_participant))
        return market_ledger, quant

    def rollout_from_ts(self, ts_start, learning_participant):
        rollout_rewards = []
        for ts in range(ts_start, self.timesteps):
            r_ts = self._maximize_self_consumption(ts, learning_participant)
            rollout_rewards.append(r_ts)
        return rollout_rewards


    def _maximize_self_consumption(self, ts, learning_participant):
        #net load
        ns_net_load = self.participants_dict[learning_participant]['metrics']['next_settle_load'][ts] - \
                      self.participants_dict[learning_participant]['metrics']['next_settle_generation'][ts]

        if ts == 0:
            battery_soc_previous = 0
        else:
            past_bat_key = list(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts - 1]['battery'].keys())[0]
            battery_soc_previous = self.participants_dict[learning_participant]['metrics']['actions_dict'][ts - 1]['battery'][past_bat_key]['SoC']
            # print(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts - 1]['battery'][past_bat_key])

        bat_key = list(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'].keys())[0]

        # battery_amt = self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'][bat_key]['action']

        charge_cap, discharge_cap = self.EES.get_discharge_charge_cap(battery_soc_previous)

        market_key = list(self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'].keys())[0]
        if ns_net_load >= 0:
            # bid for net load and whatever we need to charge battery
            self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'][market_key][
                'price'] = 0.12 # ToDo: some smart herustic for this?
            self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'][market_key][
                'quantity'] = ns_net_load + charge_cap
            # print(self.participants_dict[learning_participant]['metrics'].keys())
            # ['timestamp', 'actions_dict', 'ask_price', 'ask_quantity', 'ask_source',
            #        'bid_price', 'bid_quantity', 'bid_source', 'next_settle_load',
            #        'next_settle_generation', 'reward']

            market_ledger, quantity = self.get_market_ledger(ts, learning_participant)
            # if market_ledger != []:
            #     print(market_ledger)
            #     quants = sum([settlement[1] for settlement in market_ledger])
            #     avg_price = sum([settlement[1]*settlement[2] for settlement in market_ledger])/quants
            #     print(avg_price, quants)
            #     self.participants_dict[learning_participant]['metrics']['bid_price'][ts] = avg_price
            #     self.participants_dict[learning_participant]['metrics']['bid_quantity'][ts] = quants
            # set the battery target
            # if quant > ns_net_load then we can charge
            # elif quant < ns_net_load then we discharge
            battery_target = quantity - ns_net_load
            battery_amt, battery_soc = self.EES.simulate_activity(battery_soc_previous, energy_activity=battery_target)
            self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'][bat_key]['action'] = battery_amt
            self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'][bat_key]['SoC'] = battery_soc

        else: # ns_net_load < 0
            # lets see about the battery first
            battery_target = max(ns_net_load, charge_cap)
            battery_amt, battery_soc = self.EES.simulate_activity(battery_soc_previous, energy_activity=0)
            self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'][bat_key]['SoC'] = battery_soc
            ns_net_load = ns_net_load + battery_amt #ToDo: check if signs are the proper way around here...

            self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery'][bat_key][
                'action'] = battery_amt
            self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['asks'][market_key][
                'price'] = 0.12 # ToDo: some smart herustic for this?
            self.participants_dict[learning_participant]['metrics']['actions_dict'][ts]['asks'][market_key][
                'quantity'] = ns_net_load
            market_ledger, quantity = self.get_market_ledger(ts, learning_participant)
            # if market_ledger != []:
            #     print(market_ledger)
            #     quants = sum([settlement[1] for settlement in market_ledger])
            #     avg_price = sum([settlement[1]*settlement[2] for settlement in market_ledger])/quants
            #     print(avg_price, quants)
            #     self.participants_dict[learning_participant]['metrics']['bid_price'][ts] = avg_price
            #     self.participants_dict[learning_participant]['metrics']['bid_quantity'][ts] = quants

        grid_transactions = self._extract_grid_transactions(market_ledger=market_ledger,
                                                            learning_participant=learning_participant,
                                                            grid_transaction_dummy=self.grid_transaction_dummy,
                                                            ts=ts,
                                                            battery=battery_amt)

        rewards = self.reward_fun.calculate(market_transactions=market_ledger, grid_transactions=grid_transactions)

        # if battery_amt != 0:
        #     print('battery:', battery_amt, battery_soc)
        #     print('grid:', grid_transactions, ns_net_load)
        #     print('quantity: ', quantity)
        #     print('reward: ', rewards)
        return rewards

    def solve_for_nash(self, steps=None):
        if steps is not None:
            self.timesteps = steps
        else:
            self.timesteps = len(self.timesteps) #ToDo: this is dirty as fuck, better clean this up soon!

        metrics = Metrics_manager(initial_Qs=self._fetch_Qs(), calculate_Q_metrics=True,
                                  initial_G=self._calculate_G(), calculate_G_metrics=True)

        summed_delta = np.inf
        summed_wasser = np.inf

        while summed_delta > 0 or summed_wasser > 1.0:


            # self._full_Value_iteration()
            for participant in self.participants_dict:
                if 'learning' in self.participants_dict[participant]['trader']:
                    if self.participants_dict[participant]['trader']['learning'] == True:
                        r_ = self.rollout_from_ts(ts_start=0, learning_participant=participant)
                    print(sum(r_))
            metrics_history = metrics.calculate_metrics(new_Qs=self._fetch_Qs(),
                                                        new_G=self._calculate_G(),
                                                        return_metrics_history=True)

            summed_delta = 0
            for participant in metrics_history:
                summed_delta += abs(metrics_history[participant]['delta_G'][-1])

            summed_wasser = 0
            for participant in metrics_history:
                for action in metrics_history[participant]:
                    if action == 'bid_price' or action == 'bid_quantity' or action == 'ask_price' or action == 'ask_quantity':
                        summed_wasser += metrics_history[participant][action]['Wasserstein'][-1]

            print(metrics_history)

# ToDo: reprogram policy to try battery and net load bids first -.-

if __name__ == "__main__":
    solver = BR_solver(sim_db_path='postgresql://postgres:postgres@stargate/remote_agent_test_np',
                    agent_id = 'egauge19821',
                    sim_type = 'training',
                    start_gen = 0,)
    solver.define_action_space(action_space_separation_prices = 10,
                   action_space_separation_quantity=15,
                   action_space_separation_bat=14)
    solver.solve_for_nash()



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







