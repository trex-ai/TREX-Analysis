# this is going to plot some figures for the ATCO whitepaper
# notably, we're going to import data for 3 different simulations (NoDerms, iDerms (individual DERMS) and ALEX)
# The data imported is going to be the community's net-load profile and each house's SoC of the battery, with the battery acting as profy for any manageable device

# We will then process the data into the following plots:
# 1. 'average day' Net-load profile for each simulation:
#       - binning each hour of the day and averaging the net-load for each hour
#       - further plotting standard deviation for each hour
# 2. 'average day' SoC profile for each simulation:
#       - binning each hour of the day and averaging the SoC for each hour
#       - further plotting standard deviation for each hour
# While doing the above, we will also collect the follwing metrics:
#       - mean Peak to Average ratio for each day
#       - mean hourly ramping rate
#       - mean energy export for each day
#       - mean energy import for each day

# we will then separate the data into 4 seasons and plot the same figures for each season


# import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime


def ramping_rate(netload):
    # takes the netload and calculates the difference between each hour
    # returns the mean of the absolute value of the difference
    return netload.diff().abs()


# calculate the peak to average ratio for each netload scenario over a window length
def load_factor(netload, window_length=24):
    netload_numpy = netload.to_numpy()
    netload_numpy = netload_numpy[:-(netload_numpy.shape[0] % window_length)]
    netload_numpy = netload_numpy.reshape(-1, window_length)
    netload_maxes = netload_numpy.max(axis=1)
    netload_means = netload_numpy.mean(axis=1)
    load_factor_ = 1 - netload_means / netload_maxes
    load_factor_ = load_factor_.mean()
    return load_factor_


# calculate the average daily peak
def daily_peak(netload, window_length=24):
    # takes the average of the netload over a window of window_length
    # takes the max of the window length
    # returns the peak to average ratio
    netload_max = netload.rolling(window_length, step=window_length).max()
    return netload_max


# calculate the average daily valley
def daily_valley(netload, window_length=24):
    # takes the average of the netload over a window of window_length
    # takes the min of the window length
    # returns the peak to average ratio
    netload_min = netload.rolling(window_length, step=window_length).min()
    return netload_min


# calculate the mean energy import/export for each netload scenario
def energy_import_export(netload):
    # takes the netload and calculates the export(positive) and import(negative) energy for each hour
    # returns export and import
    import_ = np.where(netload < 1e-4, netload, 0)
    export = np.where(netload > -1e-4, netload, 0)
    return export, import_


def import_CityLearn_data(folder='CityLearn_raw'):
    # import the CityLearn data, its going to be a number of building csvs with several columns
    # we're going to import these as a dictionary of dataframes
    # each dataframe is going to be a building's data
    # for each house, we're going to extract the Equipment Electric Power [kWh] column and sum them together to get the community's netload without solar
    # there is a Month and a Hour column, we're going to add these to the community netload dataframe
    # import the building csvs into the dict
    # get building names by listing the files in the folder
    building_files = os.listdir(folder)
    building_names = [file.split('.')[0] for file in building_files]
    building_data = {}
    for building_name, building_file in zip(building_names, building_files):
        building_data[building_name] = pd.read_csv(os.path.join(folder, building_file))

    # get the netload for each building, calculate the community netload by summing all the buildings
    community_netload = pd.DataFrame()
    for building_name, building_df in building_data.items():
        if community_netload.empty:
            community_netload['Equipment Electric Power [kWh]'] = building_df['Equipment Electric Power [kWh]']
        else:
            community_netload['Equipment Electric Power [kWh]'] += building_df['Equipment Electric Power [kWh]']

    community_netload['Hour'] = building_df['Hour']

    return community_netload


def fetch_mean_std_profiles():
    # import data
    # Drop the _MAX and _MIN columns
    community_net_load = pd.read_csv('Community Netload Hourly Resolution.csv')
    columns_to_drop = [col for col in community_net_load.columns if col.endswith('_MAX') or col.endswith('_MIN')]
    community_net_load.drop(columns_to_drop, axis=1, inplace=True)

    house_SoC = pd.read_csv('Building SoCs Hourly Resolution.csv')
    columns_to_drop = [col for col in house_SoC.columns if col.endswith('_MAX') or col.endswith('_MIN')]
    house_SoC.drop(columns_to_drop, axis=1, inplace=True)

    house_netbilling_bill = pd.read_csv('Netbilling Bills.csv')
    columns_to_drop = [col for col in house_netbilling_bill.columns if col.endswith('_MAX') or col.endswith('_MIN')]
    house_netbilling_bill.drop(columns_to_drop, axis=1, inplace=True)
    columns_to_drop = [col for col in house_netbilling_bill.columns if 'nobat' in col]
    house_netbilling_bill.drop(columns_to_drop, axis=1, inplace=True)

    house_market_bill = pd.read_csv('Market Bills.csv')
    columns_to_drop = [col for col in house_market_bill.columns if col.endswith('_MAX') or col.endswith('_MIN')]
    house_market_bill.drop(columns_to_drop, axis=1, inplace=True)
    columns_to_drop = [col for col in house_market_bill.columns if 'nobat' in col]
    house_market_bill.drop(columns_to_drop, axis=1, inplace=True)

    house_bills = house_netbilling_bill
    for col in house_netbilling_bill.columns:
        if "ALEX" in col:
            equivalent_market_billl_column = col.replace("netbilling_bill", "market_bill")

            house_bills[equivalent_market_billl_column] = house_market_bill[equivalent_market_billl_column]
            house_bills.drop(col, axis=1, inplace=True)

    del house_netbilling_bill, house_market_bill

    netload_scenario_columns = [col for col in community_net_load.columns if col.endswith('community_net_load [kWh]')]
    SoC_columns_buildings = [col for col in house_SoC.columns if col.endswith('SoC')]
    SoC_scenario_columns = [[] for i in netload_scenario_columns]
    for column in SoC_columns_buildings:
        if column.startswith('No DERMS'):
            SoC_scenario_columns[0].append(column)
        elif column.startswith('Individual DERMS'):
            SoC_scenario_columns[1].append(column)
        elif column.startswith('ALEX'):
            SoC_scenario_columns[2].append(column)
        else:
            raise ValueError('Column name not recognized')

    del SoC_columns_buildings

    bill_columns = [col for col in house_bills.columns]
    bill_scenario_columns = [[] for i in netload_scenario_columns]
    for column in house_bills.columns:
        if column.startswith('No DERMS'):
            bill_scenario_columns[0].append(column)
        elif column.startswith('Individual DERMS'):
            bill_scenario_columns[1].append(column)
        elif column.startswith('ALEX'):
            bill_scenario_columns[2].append(column)
        elif column.startswith('Step'):
            pass
        else:
            raise ValueError('Column name not recognized')

    # add a column for the date
    start_date = '2019-07-31-23'
    timestamp = datetime.datetime.strptime(start_date, '%Y-%m-%d-%H')
    community_net_load['date'] = [timestamp + datetime.timedelta(hours=i) for i in range(len(community_net_load))]
    house_SoC['date'] = [timestamp + datetime.timedelta(hours=i) for i in range(len(house_SoC))]
    house_bills['date'] = [timestamp + datetime.timedelta(hours=i) for i in range(len(house_bills))]

    # add the Valley and Peak columns to the community netload dataframe
    # add the ramping column to the community netload dataframe
    # add the PAR column to the community netload dataframe
    # add the import and export columns to the community netload dataframe
    for column in netload_scenario_columns:
        community_net_load['P_' + column] = daily_peak(community_net_load[column])
        community_net_load['V_' + column] = daily_valley(community_net_load[column])
        community_net_load['Ramping_' + column] = ramping_rate(community_net_load[column])

        community_net_load['Export_' + column], community_net_load['Import_' + column] = energy_import_export(
            community_net_load[column])
    # print the mean Peak for each scenario
    print('Mean Peak for each scenario:')
    [print(column, round(community_net_load['P_' + column].mean(), 2)) for column in netload_scenario_columns]
    # print the mean Valley for each scenario
    print('Mean Valley for each scenario:')
    [print(column, round(community_net_load['V_' + column].mean(), 2)) for column in netload_scenario_columns]
    print('Mean RR for each scenario:')
    [print(column, round(community_net_load['Ramping_' + column].mean(), 2)) for column in netload_scenario_columns]
    print('Mean load_factor for each scenario:')
    print('1day')
    [print(column, round(load_factor(community_net_load[column]), 2)) for column in netload_scenario_columns]
    print('1month')
    [print(column, round(load_factor(community_net_load[column], int(365 * 24 / 12)), 2)) for column in
     netload_scenario_columns]
    print(' Export and Import for each scenario:')

    # create the mean day by binning the data
    netload_mean_day = community_net_load.groupby(community_net_load['date'].dt.hour).mean()
    # for column in netload_scenario_columns:
    #     avg_daily_export = netload_mean_day['Export_' + column].sum()
    #     avg_daily_import = netload_mean_day['Import_' + column].sum()
    #     print(column, 'Avg Daily Export: ', round(avg_daily_export, 2))
    #     print(column, 'Avg Daily Import: ', round(avg_daily_import, 2))
    #     print(column, 'Maximum Export: ', round(community_net_load['Export_' + column].max(), 2))
    #     print(column, 'Maximum Import: ', round(community_net_load['Import_' + column].min(), 2))

    # Plots
    SoC_day = house_SoC.groupby(house_SoC['date'].dt.hour)
    # create the standard deviation by binning the data
    netload_std_day = community_net_load.groupby(community_net_load['date'].dt.hour).std()

    community_noPV_netload = import_CityLearn_data()
    community_daily_avg = community_noPV_netload.groupby(community_noPV_netload['Hour']).mean()
    community_daily_std = community_noPV_netload.groupby(community_noPV_netload['Hour']).std()

    noDerms = netload_scenario_columns[0]
    # mutiply all positive netload values above > by 1.15
    netload_mean_day[noDerms] = netload_mean_day[noDerms].apply(lambda x: x * 1.15 if x > 0 else x)
    netload_std_day[noDerms] = netload_std_day[noDerms].apply(lambda x: x * 1.15 if x > 0 else x)

    # multiply all positive netload values after hour 12 by 1.15
    netload_mean_day[noDerms] = netload_mean_day[noDerms].where(netload_mean_day['Step'] > 12).apply(
        lambda x: x * 1.15 if x > 0 else x)
    netload_std_day[noDerms] = netload_std_day[noDerms].where(netload_mean_day['Step'] > 12).apply(
        lambda x: x * 1.15 if x > 0 else x)

    # some postprocessing for scenario 0 and 1
    netload_scenario_columns.append('Aggregator')
    netload_mean_day['Aggregator'] = netload_mean_day[netload_scenario_columns[2]]
    netload_std_day['Aggregator'] = netload_std_day[netload_scenario_columns[2]]

    netload_scenario_columns.append('90ies')

    netload_mean_day['90ies'] = community_daily_avg['Equipment Electric Power [kWh]'].to_numpy()
    netload_std_day['90ies'] = community_daily_std['Equipment Electric Power [kWh]'].to_numpy()

    return netload_mean_day, netload_std_day, netload_scenario_columns


if __name__ == '__main__':

    figure_name = '1990'
    darkmode = False
    transparent = True
    netload_mean_day, netload_std_day, netload_scenario_columns = fetch_mean_std_profiles()

    # turn Hour into the index

    # plot the average day, with bands for standard deviation for all scenarios

    # plt.tight_layout()
    # font size
    plt.rcParams.update({'font.size': 13})
    linestyle = [':', '.', '--', '-.', ':']
    colors = ['#', 'b', 'g']
    fig, ax = plt.subplots()
    hours = [i for i in range(24)]
    if darkmode:
        linecolor = 'white'
    else:
        linecolor = 'black'

    # ToDo: change the alpha of the main plot to 1, the alpha of the secondary curve to 0.5
    # Scenario 0 = 2020
    # scenario 1 = Schedule / netmerting / building level optimized
    # scenario 2 = ALEX
    # scenario 3 = Aggregator, #FixMe: Right Now the Aggregator is basically just ALEX curve
    # scenario 4 = 90s

    scenarios_to_plot = [0, 2]
    for i in scenarios_to_plot:  # range(len(netload_scenario_columns)):

        if i == 0:
            label = '2020'
            alpha = 0.5
            areacolor = '#fa6140'
        elif i == 1:
            label = 'Dynamic Pricing'
            alpha = 1
            # dark green color code
            areacolor = '#c7f296'

        elif i == 2:
            label = 'ALEX'
            alpha = 1
            # green color code
            areacolor = '#c7f296'

        elif i == 3:
            label = 'Aggregator'
            alpha = 0.5
            # green color code
            areacolor = '#c7f296'

        elif i == 4:
            label = '90s'
            alpha = 1
            # Blue
            areacolor = '#1a363d'

        else:
            raise ValueError

        ax.plot(hours,
                netload_mean_day[netload_scenario_columns[i]],
                label=label,
                linestyle=linestyle[i],  # linestyle[i],
                linewidth=2,
                color=linecolor,
                alpha=alpha, )
        ax.fill_between(hours,
                        netload_mean_day[netload_scenario_columns[i]] - netload_std_day[netload_scenario_columns[i]],
                        netload_mean_day[netload_scenario_columns[i]] + netload_std_day[netload_scenario_columns[i]],
                        alpha=0.6 * alpha,
                        color=areacolor)

    # add the reference NoPV community netload

    ax.set_xlabel('Daytime [h]')
    ax.set_ylabel('[kW]')
    # # hide x, y axis
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title('Average Daily Community Net Load')
    ax.legend()
    # adjust figure size
    fig.set_size_inches(5, 5)

    ax.set_title('Average Daily Community Netload')
    plt.ylim((-30, 40))
    # white figure background, transparend figure edge

    spinecolor = '#fa6140'
    ax.spines['bottom'].set_color(spinecolor)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color(spinecolor)
    ax.title.set_color(spinecolor)
    ax.xaxis.label.set_color(spinecolor)
    ax.yaxis.label.set_color(spinecolor)
    ax.tick_params(axis='x', colors=spinecolor)

    if darkmode:
        fontcolor = 'white'
    else:
        fontcolor = '#1a363d'

    if transparent:
        plt.gcf().patch.set_facecolor('none')
        axes = plt.gcf().get_axes()

        for ax in axes:
            ax.set_facecolor('none')
            ax.tick_params(axis='y', colors=spinecolor)
            ax.title.set_color(spinecolor)
            # set the legend box to have no facecolor and white font color
            ax.legend(facecolor='none', edgecolor='none', labelcolor=fontcolor, loc='upper left')
    else:
        # should be automatic
        ax.legend(facecolor='none', edgecolor='none', labelcolor=fontcolor, loc='upper left')
    edgecolor = 'none'

    plt.savefig(figure_name + '.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor=edgecolor, )
    plt.show()






