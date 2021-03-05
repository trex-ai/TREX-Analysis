# contains the metrics for participants dict comparison during the BR jazz
import numpy as np

# merge metrics, because atm Q and G is separated
class Metrics_manager(object):
    def __init__(self, initial_Qs=None, initial_G=None, initial_pi=None,
                 calculate_Q_metrics=True, calculate_G_metrics=True, calculate_pi_metrics=False):

        if initial_Qs is None and calculate_Q_metrics == True:
            print('disabling calculation of Q metrics since no initial values were provided')
            calculate_Q_metrics = False
        self.old_Qs = initial_Qs
        self.calculate_Q_metrics = calculate_Q_metrics

        if initial_G is None and calculate_G_metrics == True:
            print('disabling calculation of G metrics since no initial values were provided')
            calculate_G_metrics = False
        self.old_G = initial_G
        self.calculate_G_metrics = calculate_G_metrics

        if initial_pi is None and calculate_pi_metrics == True:
            print('disabling calculation of pi metrics since no initial values were provided')
            calculate_pi_metrics = False
        self.old_pi = initial_pi
        self.calculate_pi_metrics = calculate_pi_metrics

        self.metrics_history = {}


    def calculate_metrics(self, new_Qs=None, new_G=None, new_pi=None, return_metrics_history=False):

        # id we can, calculate Q metrics
        if new_Qs is not None:

            Q_metrics = _calculate_Q_metrics(old_Q=self.old_Qs, new_Q=new_Qs, calculate_wasserstein=True)
            self.old_Qs = new_Qs

            for participant in Q_metrics:
                if participant not in self.metrics_history:
                    self.metrics_history[participant] = {}

                for action in Q_metrics[participant]:
                    if action not in self.metrics_history[participant]:
                        self.metrics_history[participant][action] = {}

                    for metric in Q_metrics[participant][action]:
                        if metric not in self.metrics_history[participant][action]:
                            self.metrics_history[participant][action][metric] = []

                        self.metrics_history[participant][action][metric].append(Q_metrics[participant][action][metric])
        else:
            print('did not supply both old_Q and new_Q')

        # if we can, calculate G metrics
        if new_G is not None:
            G_metrics = _calculate_G_metrics(old_G=self.old_G, new_G=new_G, calculate_delta_G=True, remember_G=True)
            self.old_G = new_G
            for participant in G_metrics:
                if participant not in self.metrics_history:
                    self.metrics_history[participant] = {}

                for metric in G_metrics[participant]:
                    if metric not in self.metrics_history[participant]:
                        self.metrics_history[participant][metric] = []

                    self.metrics_history[participant][metric].append(G_metrics[participant][metric])
        else:
            print('did not supply both old_G and new_G')

        if new_pi is not None:
            pi_metrics = _calculate_pi_metrics(old_G=self.old_pi, new_G=new_pi)

            for participant in pi_metrics:
                if participant not in self.metrics_history:
                    self.metrics_history[participant] = {}

                for action in pi_metrics[participant]:
                    if action not in self.metrics_history[participant]:
                        self.metrics_history[participant][action] = {}

                    for metric in pi_metrics[participant][action]:
                        if metric not in self.metrics_history[participant][action]:
                            self.metrics_history[participant][action][metric] = []

                        self.metrics_history[participant][action][metric].append(pi_metrics[participant][action][metric])

        if return_metrics_history:
            return self.metrics_history

def __Wasserstein(x:list, y:list):
    x_cumsum = np.cumsum(x)
    y_cumsum = np.cumsum(y)
    mass_delta = np.absolute(np.sum(x_cumsum - y_cumsum, axis=-1))
    return mass_delta

def _calculate_G_metrics(new_G, old_G,
                         calculate_delta_G=True,
                         remember_G=True):

    G_metrics = {}
    for participant in new_G:
        if participant in old_G:
            G_metrics[participant] = {}
            if calculate_delta_G:
                G_metrics[participant]['delta_G'] = new_G[participant] - old_G[participant]
            if remember_G:
                G_metrics[participant]['new_G'] = new_G[participant]
                G_metrics[participant]['old_G'] = old_G[participant]

    return G_metrics

# Q_metrics calculating, returns Q-metrics
def _calculate_Q_metrics(old_Q, new_Q,
                         calculate_wasserstein=True):
    metrics={}
    for participant in old_Q:
        if participant in new_Q:
            metrics[participant] = {}

            for action in old_Q[participant][0]:
                if action in new_Q[participant][0]:
                    metrics[participant][action] = {}

            if calculate_wasserstein:
                Wasserstein = {}
                for idx in range(len(old_Q[participant])):
                    x = old_Q[participant][idx]
                    y = new_Q[participant][idx]

                    for action in x:
                        if action not in Wasserstein:
                            Wasserstein[action] = []
                        if action in y:
                            w = abs(__Wasserstein(x[action], y[action]))
                            Wasserstein[action].append(w)
                        # if w != 0:
                        #     print('x', x)
                        #     print('y', y)
                        #     print('w', w)

                for action in metrics[participant]:
                    metrics[participant][action]['Wasserstein'] = sum(Wasserstein[action])

    return metrics

def _calculate_pi_metrics(old_pi, new_pi):
    pass
# def _calculate_market_metrics(participants_dict):
#     G_metrics = {}
#     for participant in new_G:
#         if participant in old_G:
#             G_metrics[participant] = {}