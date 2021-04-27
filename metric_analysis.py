from _utils import extract
# from analysis.simple_optimal_value import optimal_reward


def plot_metrics(agent_id = 'eGauge13830',
                db_path = 'postgresql://postgres:postgres@localhost/backwards_test',
                 gens=1,
                 simtype='csp',
                 metric='value' # value, network_loss, price_action ....
                 ):

    metric_ts = []
    for gen in range(gens):
        table_name = str(gen) + '_'+simtype+'_metrics'
        metrics = extract.from_database(db_path, table_name)
        metric_ts.extend(metrics[agent_id][metric])
    import plotly.graph_objects as go
    #
    # x = np.arange(10)
    # smooth_returns = savgol_filter(returns, 31, 5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=metric_ts, name=metric,
                             mode='markers',
                             marker=dict(
                            color='rgba(135, 206, 250, 0.5)',
                             opacity=0.5,
                            size=1,
                           line=dict(color='MediumPurple',
                               width=0.1)
                             )
                             )
                  )

    fig.update_layout(title=metric + 'for ' + str(gen) + ' gens',
                      xaxis_title='ts',
                      yaxis_title='$')

    fig.show()

def plot_prices(agent_id = 'eGauge13830',
                db_path = 'postgresql://postgres:postgres@localhost/Rewards_test',
                 gens=1,
                 simtype='csp',
                 metric='value' # value, network_loss, price_action ....
                 ):

    metric_ts = []
    for gen in range(gens):
        table_name = str(gen) + '_'+simtype+'_metrics'
        metrics = extract.from_database(db_path, table_name)
        metric_ts.extend(metrics[agent_id][metric])
    import plotly.graph_objects as go
    #
    # x = np.arange(10)
    # smooth_returns = savgol_filter(returns, 31, 5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=metric_ts, name=metric,
                             marker=dict(
                            color='rgba(135, 206, 250, 0.5)',
                             opacity=0.5,
                            size=1,
                            line=dict(color='MediumPurple',
                                width=1)
                             )
                             )
                  )

    fig.update_layout(title=metric + 'for ' + str(gen) + ' gens',
                      xaxis_title='ts',
                      yaxis_title='$')

    fig.show()

def plot_returns(agent_id = 'eGauge13830',
                db_path = 'postgresql://postgres:postgres@localhost/Rewards_test',
                 gens=1,
                 compare_with_optimal=False,
                 ):
    if compare_with_optimal:
        _, opt_reward = optimal_reward(36, 'eGauge13830')
        print('print optimal reward', opt_reward[:100])
        opt_returns = [sum(opt_reward)/1000]*gens
    returns = []
    returns_diff = []
    printed = False
    for gen in range(gens):
        table_name = str(gen) + '_vmarket_metrics'
        metrics = extract.from_database(db_path, table_name)
        if not printed:
            print('agent rewards, starting at 00:00', metrics[agent_id]['rewards'][:100])
        gen_return = sum(metrics[agent_id]['rewards'])/1000
        returns.append(gen_return)

        if compare_with_optimal:
            returns_diff.append(gen_return - opt_returns[0])

    import plotly.graph_objects as go
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=returns, name='achieved', mode='lines+markers'))
    if compare_with_optimal:
        fig.add_trace(go.Scatter(y=opt_returns, name='optimal', mode='lines+markers'))
        fig.add_trace(go.Scatter(y=returns_diff, name='difference', mode='lines+markers'))

    fig.update_layout(title='Returns',
                      xaxis_title='Generation',
                      yaxis_title='$')

    fig.show()



