# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _extractor.extractor import Extractor
import numpy as np
import copy
import itertools
from _utils.market_simulation import sim_market, _test_settlement_process, _map_market_to_ledger
from _utils.rewards_proxy import NetProfit_Reward as Reward
from _utils.metrics import Metrics_manager
from _utils.Bess import Storage
# ----------------------------------------------------------------------------------------------------------------------
def _add_metrics_to_participants(participants_dict, extractor, start_gen, sim_type):
    for participant in participants_dict.keys():
        participant_dict = participants_dict[participant]

        if 'track_metrics' not in participant_dict['trader']:
            print('no <track_metrics> in participant dict!')

        if participant_dict['trader']['track_metrics'] == True:
            participants_dict[participant]['metrics'] = extractor.from_metrics(start_gen, sim_type, participant)

    return participants_dict
# ----------------------------------------------------------------------------------------------------------------------
class Solver(object):
    def __init__(self,
                 sim_db_path='postgresql://postgres:postgres@stargate/remote_agent_test_np',
                 agent_id='egauge19821',
                 sim_type='training',
                 start_gen=0,
                 fake_data_scenario='test_buy_fixed', #None, 'test_buy_fixed', 'test_buy_variable1', 'test_buy_variable1'
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
        # _test_settlement_process(self.participants_dict, agent_id, market_df)

        self.reward_fun = Reward()
        min_price = exp_config['data']['market']['grid']['price']
        max_price = min_price * (1 + exp_config['data']['market']['grid']['fee_ratio'])
        self.prices_max_min = (max_price, min_price)
        self.grid_transaction_dummy = (None, max_price, None, min_price)
        # self._test_G()

        if fake_data_scenario is not None:
            self._create_test_participants_dict(24, test_scenario=fake_data_scenario) #teest scenarios go over 24 steps at the moment

        self.timesteps = list(range(min(market_df['time_creation']), max(market_df['time_creation']) + 60, 60))

        self.EES = Storage()

    def return_participant_dict(self):
        return self.participants_dict

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
                        print(participant,  'FAILED Return check, please into cause of offset. return: ', metered_G)

    def _calculate_G(self):
        G = {}
        for participant in self.participants_dict:
            if 'learning' in self.participants_dict[participant]['trader']:
                if self.participants_dict[participant]['trader']['learning'] == True:
                        rewards = []
                        for row in range(self.participants_dict[participant]['metrics']['reward'].shape[0]):
                            r, _ = self._query_market_get_reward_for_one_tuple(row=row, learning_participant=participant)
                            rewards.append(r)
                        G[participant] = sum(rewards)

        return G

    def _query_market_get_reward_for_one_tuple(self, row, learning_participant, do_print=False):
        market_df = sim_market(participants=self.participants_dict, learning_agent_id=learning_participant, row=row)
        market_ledger = []
        quant = 0
        for index in range(market_df.shape[0]):
            settlement = market_df.iloc[index]
            quant = settlement['quantity']
            market_ledger.append(_map_market_to_ledger(settlement, learning_participant))
        if do_print:
            print(quant)

        # key = list(participants_dict[learning_participant]['metrics']['actions_dict'][ts]['bids'].keys())[0]
        # bat_action = participants_dict[learning_participant]['metrics']['actions_dict'][ts]['battery']
        if 'battery' in self.participants_dict[learning_participant]['metrics']['actions_dict'][row]:
            if row == 0:
                battery_soc_previous = 0
            else:
                bat_key = \
                list(self.participants_dict[learning_participant]['metrics']['actions_dict'][row - 1]['battery'].keys())[
                    0]
                battery_soc_previous = self.participants_dict[learning_participant]['metrics']['actions_dict'][row - 1]['battery'][bat_key]['SoC']
            bat_key = \
            list(self.participants_dict[learning_participant]['metrics']['actions_dict'][row]['battery'].keys())[0]
            battery_amt = self.participants_dict[learning_participant]['metrics']['actions_dict'][row]['battery'][bat_key]['action']

            battery_amt, battery_soc_previous = self.EES.simulate_activity(battery_soc_previous,
                                       energy_activity=battery_amt)

        else:
            battery_amt = 0

        grid_transactions = self._extract_grid_transactions(market_ledger=market_ledger,
                                                           learning_participant=learning_participant,
                                                           ts=row,
                                                           battery=battery_amt)

        rewards = self.reward_fun.calculate(market_transactions=market_ledger, grid_transactions=grid_transactions)
        # print('market', market_ledger)
        # print('grid', grid_transactions)
        # print('r', rewards)
        # print('metered_r', participants_dict[learning_participant]['metrics']['reward'][ts])
        return rewards, quant

    def _extract_grid_transactions(self, market_ledger, learning_participant, ts, battery=0.0):
        sucessfull_bids = sum([sett[1] for sett in market_ledger if sett[0] == 'bid'])
        sucessfull_asks = sum([sett[1] for sett in market_ledger if sett[0] == 'ask'])

        net_influx = self.participants_dict[learning_participant]['metrics']['next_settle_generation'][ts] + sucessfull_bids
        net_outflux = self.participants_dict[learning_participant]['metrics']['next_settle_load'][ts] + sucessfull_asks

        grid_load = net_outflux - net_influx + battery

        return (max(0, grid_load), self.prices_max_min[0], max(0, -grid_load), self.prices_max_min[1])

    def _create_test_participants_dict(self, test_length, test_scenario):

        # for our tests we only want 2 participants, one learning one not learning
        if len(self.participants_dict) > 2:
            for unnecessary_participant in list(self.participants_dict.keys)[2:]:
                del self.participants_dict[unnecessary_participant]
            remaining_participants = list(self.participants_dict.keys())
            self.participants_dict[remaining_participants[0]]['trader']['learning'] = True
            print(remaining_participants[0], self.participants_dict[remaining_participants[0]]['trader']['learning'])
            self.participants_dict[remaining_participants[1]]['trader']['learning'] = False
            print(remaining_participants[1], self.participants_dict[remaining_participants[1]]['trader']['learning'])
        else:
            remaining_participants = list(self.participants_dict.keys())

        # we then want to make sure that we have only thee desired amount of steps
        for participant in self.participants_dict:
            self.participants_dict[participant]['metrics'] = self.participants_dict[participant]['metrics'][:test_length]

        if test_scenario == 'test_buy_fixed':
            print('building test_buy_fixed test scenario fake data')
            # generate arbitrary timestamps for settlements / actions dict:
            # this is necessary FOR NOW to use the same market settlement process that the large T-REX simulation usees ... sigh
            ts_actions_list = []
            for action in self.participants_dict[remaining_participants[0]]['metrics']['actions_dict']:
                ts_actions_list.append(list(action[list(action.keys())[0]].keys())[0])

            # set buyer /learner load to always exced generatiion by 20
            self.learner = remaining_participants[0]
            self.participants_dict[self.learner]['metrics']['next_settle_load'] = [40]*test_length
            self.participants_dict[self.learner]['metrics']['next_settle_generation'] = [20]*test_length
            # create the learner's optimal actions dict to get the optimal reward
            for idx in range(len(self.participants_dict[participant]['metrics'])):
                actions_dict = {'bids': {}}
                actions_dict['bids'] = {ts_actions_list[idx]:
                                            {'quantity': 20,
                                             'source': 'solar',
                                             'price': 0.10,
                                             'participant_id': self.learner},
                                        }

                self.participants_dict[self.learner]['metrics'].at[idx, 'actions_dict'] = actions_dict

            # set seller generation to always exceed load by 20
            # set seller to always sell full residual load for 11cts/Wh
            non_learner =  remaining_participants[1]
            self.participants_dict[non_learner]['metrics']['next_settle_load'] = [20]*test_length
            self.participants_dict[non_learner]['metrics']['next_settle_generation'] = [40]*test_length
            self.participants_dict[non_learner]['metrics']['ask_price'] = [0.11]*test_length
            self.participants_dict[non_learner]['metrics']['ask_quantity'] = [20]*test_length
            # create the seller's actions_dict
            for idx in range(len(self.participants_dict[participant]['metrics'])):
                actions_dict = {'asks': {}}
                actions_dict['asks'] = {ts_actions_list[idx]: {'quantity': 20,
                                             'source': 'solar',
                                             'price': 0.10,
                                             'participant_id': non_learner},
                                        }

                self.participants_dict[non_learner]['metrics'].at[idx, 'actions_dict']  = actions_dict

            print(self.participants_dict[participant]['metrics'].columns)
            self.test_current_policy(self.learner)


    def test_current_policy(self, learner):
        G = 0
        q_max = 0
        #
        for row in range(len(self.participants_dict[learner]['metrics'])):
            r, q = self._query_market_get_reward_for_one_tuple(row, learner)
            G += r
            q_max += q

        print('Policy achieves thee following return: ', G)
        print('settled quantity is: ', q_max)

    def MCTS(self, max_it=500, action_space=[2]):
    # here's where the MCTS comes in
    # We need to define an action space for every time step

        # ToDo; Extend to N dimensions
        self.actions = {
            # np.linspace(self.prices_max_min[1], self.prices_max_min[0], action_space[0])
                        'bid_price': [0.1, 0.2],
            # np.linspace(self.prices_max_min[1], self.prices_max_min[0], action_space[0])
                        'bid_quantity': [0, 10, 20, 30]
                   }
        self.shape_action_space = []
        for action_dimension in self.actions:
            self.shape_action_space.append(len(self.actions[action_dimension]))
        num_individual_entries = 1

        for dimension in self.shape_action_space:
            num_individual_entries = num_individual_entries*dimension
        action_space =  np.arange(num_individual_entries).tolist()

        start = self.participants_dict[self.learner]['metrics']['timestamp'][0] #first state of the cropped data piece
        self.time_end = self.participants_dict[self.learner]['metrics']['timestamp'][len(self.participants_dict[self.learner]['metrics'])-1]
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

        it = 0
        while it < max_it:
            game_tree = self._one_MCT_rollout_and_backup(game_tree, s_0, action_space)
            it += 1
            # print('ieration: ', it)

        # establish the best policy and test
        self._update_policy_from_tree(self.learner, game_tree)
        self.test_current_policy(learner=self.learner)

        return game_tree

    def _update_policy_from_tree(self, participants, game_tree):
        for state in game_tree:
            # find the best action by determining Q = r + V_next
            actions = list(game_tree[state]['a'].keys())
            Q = []
            for a in actions:
                r = game_tree[state]['a'][a]['r']
                s_next = game_tree[state]['a'][a]['s_next']
                ts, _ = self.decode_states(s_next)
                if ts >= self.time_end:
                    V = 0
                else:
                    V = game_tree[s_next]['V']
                Q.append(V + r)

            a_greedy = actions[np.argmax(Q)]

            ts, _ = self.decode_states(state)

            row = self.participants_dict[self.learner]['metrics'].index[
                self.participants_dict[self.learner]['metrics']['timestamp'] == ts]
            row = row[0]
            self.participants_dict[self.learner]['metrics'].at[row, 'actions_dict'] = self.decode_actions(a_greedy, ts)




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

    def decode_states(self, s):
        timestamp = s[0]
        return timestamp, None

    def decode_actions(self, a, ts):

        actions_dict = {}
        a = np.unravel_index(int(a), self.shape_action_space)
        price = self.actions['bid_price'][a[0]]
        quantity = self.actions['bid_quantity'][a[1]]
        ts = (ts+60, ts+2*60)
        # print(price)
        actions_dict['bids'] = {str(ts): {'quantity': quantity,
                                     'price': price,
                                     'source': 'solar',
                                     'participant_id': self.learner
                                     }}
        return actions_dict

    def encode_states(self, ts):
        # for now we only encode  time
        s = (ts, None)
        return s

    def evaluate_transition(self, s_now, a):
        # for now the state tuple is: (time)
        timestamp, _ = self.decode_states(s_now) # _ being a placeholder for now

        #find the appropriate row in the dataframee
        row = self.participants_dict[self.learner]['metrics'].index[self.participants_dict[self.learner]['metrics']['timestamp'] == timestamp]
        row = row[0]

        actions = self.decode_actions(a, timestamp)
        self.participants_dict[self.learner]['metrics'].at[row, 'actions_dict'] =  actions  # update the actions dictionary
        # print(self.participants_dict[self.learner]['metrics']['actions_dict'][row])
        r, _ = self._query_market_get_reward_for_one_tuple(row, self.learner, do_print=False)
        s_next = self.encode_states(timestamp+60)

        # print(r)
        return r, s_next

    def _next_states(self, s_now, a):
        t_now = s_now[0]
        t_next = t_now + 60
        s_next = self.encode_states(t_next)
        return s_next

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
            if ts < self.time_end:
                game_tree[s_next]['V'] = self.random_rollout(s_next, action_space)

            return game_tree, None, True
        else:

            r, s_next = self.evaluate_transition(s_now, a)

            game_tree[s_now]['a'][a]['r'] = (game_tree[s_now]['a'][a]['n']*game_tree[s_now]['a'][a]['r'] + r)/(game_tree[s_now]['a'][a]['n']+1)
            game_tree[s_now]['a'][a]['s_next'] = s_next
            game_tree[s_now]['a'][a]['n'] +=1
            game_tree[s_now]['N'] += 1
            if s_next[0] >= self.time_end:
                finished = True

            return game_tree, s_next, finished

    def __add_s_next(self, game_tree, s_now, action_space):
        a_next = {}
        for action in action_space:

            s_next = self._next_states(s_now, action)
            ts, _ = self.decode_states(s_next)
            if s_next not in game_tree and ts < self.time_end:
                game_tree[s_next] = {'N': 0}

            a_next[str(action)] = {'r': None,
                              'n': 0,
                              's_next': s_next}

        game_tree[s_now]['a'] = a_next

        return game_tree


    # here's the two policies that we'll be using for now:
    # UCB for the tree traversal
    # random action selection for rollouts
    def random_rollout(self, s_now, action_space):
        finished = False
        V = 0
        while not finished:
            a = np.random.choice(action_space)
            ts, _ = self.decode_states(s_now)
            if ts == self.time_end:
                finished = True
            else:
                r, s_now = self.evaluate_transition(s_now, a)
                V += r

        return V

    def _ucb(self, game_tree, s_now, c=2):
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
                Q_ucb[idx_a]= Q + c*np.sqrt(np.log(N_s)/n_next)
            else:
                Q_ucb[idx_a] = np.inf

        #making sure we pick the maximums at random
        a_ucb_index = np.random.choice(np.where(Q_ucb == np.max(Q_ucb))[0])
        a_ucb = actions[a_ucb_index]
        return a_ucb

