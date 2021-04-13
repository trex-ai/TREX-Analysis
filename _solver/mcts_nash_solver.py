# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _extractor.extractor import Extractor
import numpy as np
import copy
import itertools
import copy
from _utils.market_simulation_3 import sim_market, _test_settlement_process, _map_market_to_ledger
from _utils.rewards_proxy import NetProfit_Reward as Reward
from _utils.metrics import Metrics_manager
from _utils.Bess import Storage

import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
def _add_metrics_to_participants(participants_dict, extractor, start_gen, sim_type):
    for participant in participants_dict.keys():
        participant_dict = participants_dict[participant]

        if 'track_metrics' not in participant_dict['trader']:
            print('no <track_metrics> in participant dict!')

        if participant_dict['trader']['track_metrics'] == True:
            participants_dict[participant]['metrics'] = extractor.from_metrics(start_gen, sim_type, participant)

    return participants_dict

# ToDo (low priority): Add a process to tune parameters and balances MCTS's UBC weighting factor c with the magnitude of reward
# for now we assume that r_average =ca= c*sqrt(log(num_actions)/1), which means c = r_average / sqrt(log(num_actions)/1)


# Solver object that houses:
# - the data extraction from the database, to create the game tree that MCTS is supposed to solve
# - MCTS with UCB + random rollouts
# - implemented capabilities:
#       - variable load and generation
#       - Action space: quantized sell and buy prices
#       - Learning domains: each agent can ATM only sell or buy

# ToDo I: Implement &test batteries
    # ToDo 1a: Debug & reimplement battery
    # ToDo 1b: Implement a single player load-shift test to verify MCTS can learn battery
    # ToDo 2b: Implement a multi player test with several sellers and buyers with battery (so each has sell/buy + battery as actions)
# ToDo II: implement selling AND buying
# ToDo III: Run large scale experiment?

class Solver(object):
    def __init__(self,
                 sim_db_path='postgresql://postgres:postgres@stargate/remote_agent_test_np',
                 sim_type='training',
                 start_gen=0, # the generation we import the game-tree from, shouldn't make a difference but might be useful down the line
                 test_scenario='fixed', #None, 'variable', 'fixed', 'battery'
                 test_length = 5, #in steps
                 test_participants = 8,
                 ):

        # get the data to construct the game_tree, the game tree will be for now compressed in self.participants_dict
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

        # optional tests for markets to see if we get the same as the simulation did
        # market_df = extractor.from_market(start_gen, sim_type)
        # _test_settlement_process(self.participants_dict, agent_id, market_df)

        # set up reward and action space limits
        self.reward_fun = Reward()
        min_price = exp_config['data']['market']['grid']['price']
        max_price = min_price * (1 + exp_config['data']['market']['grid']['fee_ratio'])
        self.prices_max_min = (max_price, min_price)
        # ToDO: battery import & setup here?
        self.EES = Storage()

        # update the game tree to the test scenario
        if test_scenario is not None:
            self._create_test_participants(test_length,
                                           test_scenario=test_scenario,
                                           test_participants=test_participants) #teest scenarios go over 24 steps at the moment

    # helper for external / inspection usage
    def return_participant_dict(self):
        return self.participants_dict

    # creates specific game trees for test scenarios (manipulates load, generation, preselected actions, etc...)
    def _create_test_participants(self, test_length, test_scenario, test_participants):

        # for our tests we only want 2 participants, one learning one not learning
        if len(self.participants_dict) > test_participants:
            for unnecessary_participant in list(self.participants_dict.keys)[test_participants:]:
                del self.participants_dict[unnecessary_participant]
        elif len(self.participants_dict) < test_participants:
            # we'll need to generate new pseudo participants on the fly
            num_missing_participants = test_participants - len(self.participants_dict)
            template_participant = list(self.participants_dict.keys())[0]
            for _ in range(num_missing_participants):
                new_participant = template_participant + '_pseudo_' + str(_)
                new_participant_dict = copy.deepcopy(self.participants_dict[template_participant])
                self.participants_dict[new_participant]  = new_participant_dict


        for participant in self.participants_dict:

            self.participants_dict[participant]['trader']['learning'] = True
            print(participant,
                  'learning set to:',
                  self.participants_dict[participant]['trader']['learning'])

            self.participants_dict[participant]['metrics'] = self.participants_dict[participant]['metrics'][:test_length]

        #ToDo: optional if if we want to have other types of profiles!
        self._set_up_flat_pseudo_profiles(test_length=test_length, test_scenario=test_scenario)

        print(self.participants_dict[participant]['metrics'].columns)

        for participant in self.participants_dict:
            self.evaluate_current_policy(participant, do_print=True)

    # change profiles to match desired test
    def _set_up_flat_pseudo_profiles(self, test_length, test_scenario):
        print('building test_buy_fixed test scenario fake data')
        participants = list(self.participants_dict.keys())
        num_participants = len(participants)

        for index in range(num_participants):


            if test_scenario == 'fixed':
                # set up half of the participants as sources, half as sinks
                load = [20]*test_length if index < num_participants/2 else [0]*test_length
                gen = [20]*test_length if index >= num_participants/2 else [0]*test_length

                if index < num_participants/2:
                    self._set_up_initial_actions(participants[index],
                                                 action_type='bids',
                                                 scenario='fixed')
                else:
                    self._set_up_initial_actions(participants[index],
                                                 action_type='asks',
                                                 scenario='fixed')

            elif test_scenario == 'variable':
                load = []
                gen = []
                for i in range(test_length):
                    load_ts = np.random.choice([5, 10, 15, 20])
                    load.append(load_ts)
                    gen_ts = np.random.choice([5, 10, 15, 20])
                    gen.append(gen_ts)

                if index < num_participants/2:
                    self._set_up_initial_actions(participants[index],
                                                 action_type='bids',
                                                 scenario='variable')
                else:
                    self._set_up_initial_actions(participants[index],
                                                 action_type='asks',
                                                 scenario='variable')

            elif test_scenario == 'battery':  # set up one player temporal shift check
                load = []
                gen = []
                for ts in range(test_length):
                    if ts < test_length / 2:  # first we have over generation
                        gen.append(20)
                        load.append(0)
                    else:  # then we have under generation
                        gen.append(0)
                        load.append(20)

                self._set_up_initial_actions(participants[index],
                                             action_type='asks',
                                             scenario=test_scenario)



            self.participants_dict[participants[index]]['metrics']['next_settle_load'] = load
            self.participants_dict[participants[index]]['metrics']['next_settle_generation'] = gen



    # change actions to match desired test
    def _set_up_initial_actions(self,
                                participant,
                                action_type='bids', # 'asks',
                                scenario='fixed', #or 'variable'
                                ):

        # generate arbitrary timestamps for settlements / actions dict:
        # this is necessary FOR NOW to use the same market settlement process that the large T-REX simulation usees ... sigh
        # ts_actions_list = []
        # for action in self.participants_dict[participant]['metrics']['actions_dict']:
        #     key = list(action[list(action.keys())[0]].keys())[0]
        #     ts_actions_list.append(key)
        ts_actions_list = []
        for i in range(len(self.participants_dict[participant]['metrics'])):
            ts = self.participants_dict[participant]['metrics']['timestamp'][i]
            key = (ts + 60, ts + 2*60)
            key = str(key)
            ts_actions_list.append(key)

        for idx in range(len(self.participants_dict[participant]['metrics'])):
            # ToDo: can't do bids and asks yet!
            actions_dict = {action_type: {}}
            if scenario == 'fixed':
                quantity = 20
                price = 0.10
            elif scenario == 'variable':
                #ToDo: make this variable
                quantity = np.random.choice([10, 20, 30])
                price = np.random.choice([0.05, 0.1, 0.15])

            actions_dict[action_type] = {ts_actions_list[idx]:
                                        {'quantity': quantity,
                                         'source': 'solar', #ToDo: this might be changing in the future!
                                         'price': price,
                                         'participant_id': participant
                                         },
                                    }

            self.participants_dict[participant]['metrics'].at[idx, 'actions_dict'] = actions_dict

    # calculate G for every participant and return an equally formatted dictionary, can be tested with _test_G
    def _calculate_G(self):
        G = {}
        for participant in self.participants_dict:
            if 'learning' in self.participants_dict[participant]['trader']:
                if self.participants_dict[participant]['trader']['learning'] == True:
                    rewards = []
                    for row in range(self.participants_dict[participant]['metrics']['reward'].shape[0]):
                        r, _, __ = self._query_market_get_reward_for_one_tuple(row=row,
                                                                               learning_participant=participant)
                        rewards.append(r)
                    G[participant] = sum(rewards)
        return G

    # tests return and checks if our reward and setup achieves the same return as the simulation, to debug _calculate_G compare the raw database import
    def _test_G(self):
        calculated_Gs = self._calculate_G()
        for participant in self.participants_dict:
            if 'learning' in self.participants_dict[participant]['trader']:
                if self.participants_dict[participant]['trader']['learning'] == True:
                    calculated_G = calculated_Gs[participant]
                    metered_G = sum(self.participants_dict[participant]['metrics']['reward'][1:])
                    if calculated_G == metered_G:
                        print(participant, 'sucessfully passed Return check. return: ', metered_G)
                    else:
                        print(participant, 'FAILED Return check, please into cause of offset. return: ', metered_G)

    # get the market settlement for one specific row for one specific agent from self.participants_dict
    def _query_market_get_reward_for_one_tuple(self, row, learning_participant,
                                               do_print=False,  # idiot debug flag
                                               ):

        # get the market ledger
        market_df = sim_market(participants=self.participants_dict, learning_agent_id=learning_participant, row=row)
        market_ledger = []
        quant = 0
        for index in range(market_df.shape[0]):
            settlement = market_df.iloc[index]
            quant = settlement['quantity']
            entry = _map_market_to_ledger(settlement, learning_participant, do_print)
            if entry is not None:
                market_ledger.append(entry)

        # ToDo: reimplement battery evaluation here, this might have worked half a year ago, lol
        # key = list(participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'].keys())[0]
        # bat_action = participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery']
        battery_amt = 0
        # battery_amt, battery_soc_previous = self.EES.simulate_activity(battery_soc_previous,
        #                                energy_activity=battery_amt)

        # calculate the resulting grid transactions
        grid_transactions = self._extract_grid_transactions(market_ledger=market_ledger,
                                                            learning_participant=learning_participant,
                                                            ts=row,
                                                            battery=battery_amt)

        # then calculate the reward function
        rewards, avg_prices = self.reward_fun.calculate(market_transactions=market_ledger,
                                                        grid_transactions=grid_transactions)

        # if do_print:
        # print('market', market_ledger)
        # print('grid', grid_transactions)
        # print('r', rewards)
        # print('metered_r', participants_dict[learning_participant]['metrics']['reward'][ts])
        return rewards, quant, avg_prices

    # helper for _query_market_get_reward_for_one_tuple, to see what we get or put into grid
    def _extract_grid_transactions(self, market_ledger, learning_participant, ts, battery=0.0):
        sucessfull_bids = sum([sett[1] for sett in market_ledger if sett[0] == 'bid'])
        sucessfull_asks = sum([sett[1] for sett in market_ledger if sett[0] == 'ask'])

        net_influx = self.participants_dict[learning_participant]['metrics']['next_settle_generation'][
                         ts] - sucessfull_asks
        net_outflux = self.participants_dict[learning_participant]['metrics']['next_settle_load'][ts] - sucessfull_bids

        grid_load = net_outflux - net_influx + battery

        return (max(0, grid_load), self.prices_max_min[0], max(0, -grid_load), self.prices_max_min[1])

    # evaluate current policy of a participant inside a game tree and collects some metrics
    def evaluate_current_policy(self, participant, do_print=True):
        G = 0
        quant_cum = 0
        avg_prices = {}
        #
        for row in range(len(self.participants_dict[participant]['metrics']['actions_dict'])):

            r, quant, avg_price_row = self._query_market_get_reward_for_one_tuple(row, participant, True)

            for cathegory in avg_price_row:
                if cathegory not in avg_prices:
                    avg_prices[cathegory] = [avg_price_row[cathegory]]
                else:
                    avg_prices[cathegory].append(avg_price_row[cathegory])

            G += r
            quant_cum += quant
        for cathegory in avg_prices:
            avg_prices[cathegory] = np.nanmean(avg_prices[cathegory])

        if do_print:
            print('Policy of agent ', participant, ' achieves the following return: ', G)
            print('settled quantity is: ', quant_cum)
            print('avg prices: ', avg_prices)
        return G, quant_cum, avg_prices

    # run MCTS for every agent in the game tree...
    def MA_MCTS(self,
                epochs = 20,
                max_it_per_epoch=5000,
                action_space = [8,8], # action space might be better as a dict?
                c_adjustment=1):
        log = {}

        game_trees = {}
        s_0s = {}
        action_spaces = {}

        for participant in self.participants_dict:
            log[participant] = {'G': [],
                                'quant': []}

        for epoch in range(epochs):
            for participant in self.participants_dict:
                print('MCTS epoch', epoch, 'for', participant)
                game_trees[participant], s_0s[participant], action_spaces[participant] = self.MCTS(max_it_per_epoch, action_space, c_adjustment, learner=participant)
                # print('Return is now', G_participant)

            log = self._update_policies_and_evaluate(game_trees, s_0s, action_spaces, log)

        self._plot_log(log)

        return log

    # single agent MCTS, might not be functional anymore!
    def SA_MCTS(self,
                max_it_per_epoch=1000,
                action_space = [5,1], # action space might be better as a dict?
                c_adjustment=1):
        log = {}


        game_tree, s_0, action_space = self.MCTS(max_it_per_epoch, action_space, c_adjustment)
        # print('Return is now', G_participant)

        self._update_policy_from_tree(self.learner, game_tree, s_0, action_space)
        for action in self.participants_dict[self.learner]['metrics']['actions_dict']:
            print(action['bids'])
        self.evaluate_current_policy(participant=self.learner, do_print=True)

        self._plot_log(log)

        return log

    # for MCTS, determine a suitable c rate for UCB
    def _estimate_c_ubc(self, learner, num_actions, c_adjustment):
        G, _, __ = self.evaluate_current_policy(self.learner, do_print=False)
        r_avg = G/len(self.participants_dict[self.learner]['metrics'])
        c = np.abs(r_avg)/ np.sqrt(np.log(num_actions)/(c_adjustment*num_actions))
        # if do_print:
            # print('ubc factor is', c)
        return c

    # one single pass of MCTS for one  learner
    def MCTS(self, max_it=500,
             action_space=[5, 1], # action space might be better as a dict?
             c_adjustment=1,
             learner=None):

        # designate the target agent
        if learner is None:
            self.learner = list(self.participants_dict.keys())[0]
        else:
            self.learner = learner
        self.actions = {
                        'price': np.linspace(self.prices_max_min[1], self.prices_max_min[0], action_space[0]),
                        'quantity': np.linspace(0, 30, action_space[1])
                   }
        self.shape_action_space = []
        for action_dimension in self.actions:
            self.shape_action_space.append(len(self.actions[action_dimension]))

        # determine the size of the action space, I am sure this can be done better
        # ToDo: more elegant way of determining this pls
        num_individual_entries = 1
        for dimension in self.shape_action_space:
            num_individual_entries = num_individual_entries*dimension
        action_space =  np.arange(num_individual_entries).tolist()

        self.c_ucb = self._estimate_c_ubc(self.learner, num_individual_entries, c_adjustment)

        # determine start and build the actual game tree
        start = self.participants_dict[self.learner]['metrics']['timestamp'][0] #first state of the cropped data piece
        self.time_end = max(self.participants_dict[self.learner]['metrics']['timestamp'])
        s_0 = self.encode_states(start)
        game_tree = {}
        game_tree[s_0] = {'N': 0}
        # We need a data structure to store the 'game tree'
        # A dict with a hashed list l as key; l = [ts, a1, ...an]
        # entries in the dicts are:
        # n: number of visits
        # V: current estimated value of state
        # a: a dict with action tuples as keys
            # each of those is a subdict with key
            # r: reward for transition from s -a-> s'
            # s_next: next  state
            # n: number of times this action was taken

        # the actual MCTS part
        it = 0
        while it < max_it:
            game_tree = self._one_MCT_rollout_and_backup(game_tree, s_0, action_space)
            it += 1

        return game_tree, s_0, action_space

    # this update the policy from game tree and evaluate the policy
    # ToDO: better name, seems conflicting with _update_policy_from_tree
    def _update_policies_and_evaluate(self, game_trees, s_0s, action_spaces, measurment_dict):

        for participant in self.participants_dict:
            # establish the best policy and test
            game_tree = game_trees[participant]
            s_0 = s_0s[participant]
            action_space = action_spaces[participant]
            self._update_policy_from_tree(participant, game_tree, s_0, action_space)

        for participant in self.participants_dict:
            G, quant, avg_prices = self.evaluate_current_policy(participant=participant, do_print=True)
            if 'G' not in measurment_dict[participant]:
                measurment_dict[participant]['G'] = [G]
            else:
                measurment_dict[participant]['G'].append(G)

            if 'quant' not in measurment_dict[participant]:
                measurment_dict[participant]['quant'] = [quant]
            else:
                measurment_dict[participant]['quant'].append(quant)

            if 'avg_prices' not in measurment_dict[participant]:
                measurment_dict[participant]['avg_prices'] = {}
                for cathegory in avg_prices:
                    measurment_dict[participant]['avg_prices'][cathegory] = [avg_prices[cathegory]]
            else:
                for cathegory in avg_prices:

                    if cathegory not in measurment_dict[participant]['avg_prices']:
                        measurment_dict[participant]['avg_prices'][cathegory] = [avg_prices[cathegory]]
                    else:
                        measurment_dict[participant]['avg_prices'][cathegory].append(avg_prices[cathegory])


        return measurment_dict

    # update the policy from the game tree
    def _update_policy_from_tree(self, participant, game_tree, s_0, action_space):
        # the idea is to follow a greedy policy from S_0 as long as we can and then switch over to the default rollout policy
        finished = False
        s_now = s_0
        while not finished:

            if s_now in game_tree:
                if 'a' in game_tree[s_now]: #meaning we know the Q values, --> pick greedily
                    actions = list(game_tree[s_now]['a'].keys())
                    # print(actions)
                    Q = []
                    for a in actions:
                        r = game_tree[s_now]['a'][a]['r']
                        if r is None:
                            r = -np.inf
                        s_next = game_tree[s_now]['a'][a]['s_next']
                        ts, _ = self.decode_states(s_next)
                        if ts > self.time_end:
                            finished = True
                        if ts >= self.time_end:
                            V = 0
                        else:
                            V = game_tree[s_next]['V']
                        # print(V, game_tree[state]['a'][a], self.time_end)
                        Q.append(V + r)
                    index = np.random.choice(np.where(Q == np.max(Q))[0])
                    a_state = actions[index]

                else: #well, use the rollout policy then
                    _, s_next, a_state, finished = self.one_default_step(s_now, action_space=action_space)

            else: #well, use the rollout policy then
                _, s_next, a_state, finished = self.one_default_step(s_now, action_space=action_space)

            timestamp, _ = self.decode_states(s_now)
            s_now = s_next

            if timestamp <= self.time_end:

                row = self.participants_dict[participant]['metrics'].index[
                    self.participants_dict[participant]['metrics']['timestamp'] == timestamp]
                row = row[0]
                bid_or_ask = [action for action in
                              self.participants_dict[participant]['metrics'].at[row, 'actions_dict']]
                actions = self.decode_actions(a_state, timestamp, bid_or_ask[0], do_print=True)

                self.participants_dict[participant]['metrics'].at[row, 'actions_dict'] = actions

    # one MCTS rollout
    def _one_MCT_rollout_and_backup(self, game_tree, s_0, action_space):
        s_now = s_0
        trajectory = []
        finished = False

        # we're traversing the tree till we hit bottom
        while not finished:
            trajectory.append(s_now)
            game_tree, s_now, finished = self._one_MCTS_step(game_tree, s_now, action_space)

        game_tree = self.bootstrap_values(trajectory, game_tree)

        return game_tree

    # backprop of values
    def bootstrap_values(self, trajectory, game_tree):
        # now we backpropagate the value up the tree:
        if len(trajectory) >=2: # meaning there is actually something to backprop
            trajectory = trajectory[:-1] #discard last state as this is unrolled and we backprop one state deep anyways
            if len(trajectory) > 1:
                trajectory.reverse()

            for idx in range(len(trajectory)):
                s_now = trajectory[idx]
                # get all possible followup states
                Q = []
                actions = list(game_tree[s_now]['a'].keys())

                # now we wanna see the Q values for the transition from s -allpossible-> s' and take the max value there

                for a in actions:
                    s_next = game_tree[s_now]['a'][a]['s_next']
                    r = game_tree[s_now]['a'][a]['r']
                    V_s = game_tree[s_next]['V']

                    V_s = V_s if V_s is not None else -np.inf  # to make sure we ignore transitions we havent sampled yet
                    r = r if r is not None else -np.inf  # to make sure we ignore transitions we havent sampled yet
                    Q.append(V_s + r)

                game_tree[s_now]['V'] = np.amax(Q)

        return game_tree

    # decode states, placeholder function for more complex states
    def decode_states(self, s):
        timestamp = s[0]
        return timestamp, None

    # same as decode, but backwards...^^
    def encode_states(self, ts):
        # for now we only encode  time
        s = (ts, None)
        return s
    # decode actions, placeholder function for more complex action spaces
    def decode_actions(self, a, ts, bid_or_ask, do_print=False):

        actions_dict = {}
        a = np.unravel_index(int(a), self.shape_action_space)
        # print(price)
        actions_dict[bid_or_ask] = {str((ts+60, ts+2*60)):
                                        {'quantity': self.actions['quantity'][a[1]],
                                        'price': self.actions['price'][a[0]],
                                        'source': 'solar',
                                        'participant_id': self.learner
                                        }
                                    }
        return actions_dict

    # figure out the reward/weight of one transition
    def evaluate_transition(self, s_now, a):
        # for now the state tuple is: (time)
        timestamp, _ = self.decode_states(s_now) # _ being a placeholder for now

        #find the appropriate row in the dataframee
        row = self.participants_dict[self.learner]['metrics'].index[self.participants_dict[self.learner]['metrics']['timestamp'] == timestamp]
        row = row[0]
        #ToDO: change this to sth more elegant, otherwise we won't have full flexibility here in terms of what actions the agent can do
        bid_or_ask = [action for action in self.participants_dict[self.learner]['metrics'].at[row, 'actions_dict']]
        actions = self.decode_actions(a, timestamp, bid_or_ask[0])
        self.participants_dict[self.learner]['metrics'].at[row, 'actions_dict'] =  actions  # update the actions dictionary
        # print(self.participants_dict[self.learner]['metrics']['actions_dict'][row])
        r, _, __ = self._query_market_get_reward_for_one_tuple(row, self.learner, do_print=False)
        s_next = self.encode_states(timestamp+60)

        # print(r)
        return r, s_next

    # determine the next state
    # ToDO: this will have to change with batteries!
    def _next_states(self, s_now, a):
        t_now = s_now[0]
        t_next = t_now + 60
        s_next = self.encode_states(t_next)
        return s_next

    # a single step of MCTS, one node evaluation
    def _one_MCTS_step(self, game_tree, s_now, action_space):
        #see if wee are in a leaf node
        finished = False

        # if game_tree[s_now]['N'] == 0: #we have an unvisited leaf node:
        #     print('rollout')
        #     V_s_now = self.random_rollout(s_now, action_space)
        #     game_tree[s_now]['V'] = V_s_now
        #     game_tree[s_now]['N'] += 1
        #     return game_tree, None, True
        #
        # else: # we want to identify all followup states
        # ts, _ = self.decode_states(s_now)
        # if ts == self.time_end:
            # print('final time')
        if 'a' not in game_tree[s_now]:
            game_tree = self.__add_s_next(game_tree, s_now, action_space)

        a = self._ucb(game_tree, s_now)

        if game_tree[s_now]['a'][a]['n'] == 0:
            r, s_next = self.evaluate_transition(s_now, a)
            # print('rollout')
            game_tree[s_now]['a'][a]['r'] = r
            game_tree[s_now]['a'][a]['n'] += 1
            game_tree[s_now]['N'] += 1


            ts, _ = self.decode_states(s_next)
            if ts <= self.time_end:
                game_tree[s_next]['V'] = self.default_rollout(s_next, action_space)

            return game_tree, None, True
        else:

            r, s_next = self.evaluate_transition(s_now, a)

            game_tree[s_now]['a'][a]['r'] = r
                # ToDo: make this workk with a rolling average, calculation sth like (game_tree[s_now]['a'][a]['n']*game_tree[s_now]['a'][a]['r'] + r)/(game_tree[s_now]['a'][a]['n']+1)
            game_tree[s_now]['a'][a]['s_next'] = s_next
            game_tree[s_now]['a'][a]['n'] +=1
            game_tree[s_now]['N'] += 1
            ts_now, _ = self.decode_states(s_now)
            if ts_now >= self.time_end:
                finished = True

            return game_tree, s_next, finished

    # add to the game tree
    def __add_s_next(self, game_tree, s_now, action_space):
        a_next = {}
        for action in action_space:

            s_next = self._next_states(s_now, action)
            ts, _ = self.decode_states(s_next)
            if s_next not in game_tree and ts <= self.time_end:
                game_tree[s_next] = {'N': 0}

            a_next[str(action)] = {'r': None,
                              'n': 0,
                              's_next': s_next}


        game_tree[s_now]['a'] = a_next

        return game_tree

    # here's the two policies that we'll be using for now:
    # UCB for the tree traversal
    # random action selection for rollouts
    def one_default_step(self, s_now, action_space):
        finished = False
        a = np.random.choice(action_space)
        ts, _ = self.decode_states(s_now)
        r, s_next = self.evaluate_transition(s_now, a)
        if ts == self.time_end:
            finished = True

        return r, s_next, a, finished

    def default_rollout(self, s_now, action_space):
        finished = False
        V = 0
        while not finished:
            r, s_next, _, finished = self.one_default_step(s_now, action_space)
            s_now = s_next
            V += r

        return V

    def _ucb(self, game_tree, s_now, c=0.05):
        # UCB formula: V_ucb_next = V + c*sqrt(ln(N_s)/n_s_next)

        N_s = game_tree[s_now]['N']
        all_s_next = []

        actions = list(game_tree[s_now]['a'].keys())
        Q_ucb = [None]*len(actions) # determine the value of all followup transitions states
        for idx_a in range(len(actions)):
            a = actions[idx_a]
            s_next = game_tree[s_now]['a'][a]['s_next']


            n_next = game_tree[s_now]['a'][a]['n']
            if n_next != 0:
                r = game_tree[s_now]['a'][a]['r']
                ts, _ = self.decode_states(s_next)
                if ts >= self.time_end:
                    V_next = 0
                else:
                    V_next = game_tree[s_next]['V']
                Q = r + V_next
                Q_ucb[idx_a]= Q + self.c_ucb*np.sqrt(np.log(N_s)/n_next)
            else:
                Q_ucb[idx_a] = np.inf

        #making sure we pick the maximums at random
        a_ucb_index = np.random.choice(np.where(Q_ucb == np.max(Q_ucb))[0])
        a_ucb = actions[a_ucb_index]
        return a_ucb

    # plotter for the MCTS metrics, could easily be spun out of the solver class tbh
    def _plot_log(self, log):

        # count number of plots to make:
        #ToDO maybe less hacky than assuming all have the same metrics
        num_plots = len(log[self.learner].keys())
        fig, axs = plt.subplots(num_plots, sharex=True)
        plt.title('Variable profiles')

        plot_nbr = 0
        for metric in log[self.learner]:
            axs[plot_nbr].set_ylabel(metric)
            plot_nbr += 1
        for participant in log:
            plot_nbr = 0
            for metric in log[participant]:
                print(type(log[participant][metric]), metric)
                if not type(log[participant][metric]) is dict:

                    data = log[participant][metric]
                    label = participant

                    axs[plot_nbr].plot(data, label= label)
                else:
                    for submetric in log[participant][metric]:
                        data = log[participant][metric][submetric]
                        label = participant + submetric
                        if not np.isnan(data).all():
                            axs[plot_nbr].plot(data, label=label)

                print(data, metric, label)

                plot_nbr += 1
        # for ax in axs:
        #     ax.legend()
        # plt.legend()
        plt.show()

