# this will be the skeleton for a set of plotter functions
# written in matplotlib
# the idea is to have a set of functions that can be called to perform basic plots, such as a multiline plot and a histogram
# with a unified style language
# the seciondary function of this is to serve as code repository for more elaborate plotter functions


import matplotlib.pyplot as plt
import numpy as np



def finish_plot(fig, ax, title, darkmode=True, transparent=True):
    if transparent:
        plt.gcf().patch.set_facecolor('none')
    elif not transparent and darkmode:
        plt.gcf().patch.set_facecolor('black')
    axes = plt.gcf().get_axes()
    fontcolor = 'white' if darkmode else 'black'

    fig.set_size_inches(5, 5)
    for ax in axes:
        if transparent:
            ax.set_facecolor('none')
        elif not transparent and darkmode:
            ax.set_facecolor('black')
        ax.spines['bottom'].set_color(fontcolor)
        ax.spines['top'].set_color(fontcolor)
        ax.spines['right'].set_color(fontcolor)
        ax.spines['left'].set_color(fontcolor)
        ax.title.set_color(fontcolor)
        ax.xaxis.label.set_color(fontcolor)
        ax.yaxis.label.set_color(fontcolor)
        ax.tick_params(axis='x', colors=fontcolor)
        ax.tick_params(axis='y', colors=fontcolor)
        ax.title.set_color(fontcolor)
        # set the legend box to have no facecolor and white font color
        ax.legend(facecolor='none', edgecolor='none', labelcolor=fontcolor, loc='best')



    plt.savefig(title + 'darkmode' if darkmode else title + '.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
def multiline_plot(title, data: dict(), darkmode=False, transparent=True):
    # this function will take a list of data and plot it as a multiline plot
    # title is the title of the plot
    # x_axis_title is the title of the x axis
    # y_axis_title is the title of the y axis

    # data is a dictionary, and expected to have the following format:
        # data = {'vector_name': vector_dict, 'vector_name2': vector_dict, ...}
        # where vector_dict is a dictionary with the following format:
            # vector_dict = {'x': x_data, 'y': y_data}
    # if x is not provided, we will assume that the x axis is the index of the vector

    plt.rcParams.update({'font.size': 13})

    fig, ax = plt.subplots()
    for vector_name, vector_dict in data.items():
        # check if x is provided
        if 'x' not in vector_dict:
            vector_dict['x'] = list(range(len(vector_dict['y'])))

        ax.plot(vector_dict['x'], vector_dict['y'], label=vector_name)

    finish_plot(fig, ax, title, darkmode=darkmode, transparent=transparent)

def histogram(title, data, bins=None, darkmode=False, transparent=True):
    # this function will take a list of data and plot it as a histogram
    # title is the title of the plot
    # data is a dictionary, and expected to have the following format:
        # data = {'vector_name': vector_dict, 'vector_name2': vector_dict, ...}
        # where vector_dict is a dictionary with the following format:
        # vector_dict = {'y': y_data}
        # due to histogram format x data is not needed - it can be provided but will not be used
    # bins is the number of bins to use in the histogram, if not provided, it will be calculated automatically by combining all vectors to ensure even distribution

    plt.rcParams.update({'font.size': 13})

    if bins is None:
        # first combine all the data into a vector and make a histogram to autodetermine the number of bins
        all_data = []
        for vector_name, vector_dict in data.items():
            all_data.extend(vector_dict['y'])

        fig, ax = plt.subplots()
        (n2, bins, patches) = ax.hist(all_data, bins='auto')

        plt.close()


    fig, ax = plt.subplots()
    # if more than 1 key in the dictionary, we will plot all the histograms in the same plot and we'll adjust the opacity

    alpha = 0.5 if len(data.keys()) > 1 else 1

    for vector_name, vector_dict in data.items():
        ax.hist(vector_dict['y'], label=vector_name, bins=bins, alpha=alpha)

    finish_plot(fig, ax, title, darkmode=darkmode, transparent=transparent)
if __name__ == '__main__':
    # make a list of random numbers so we can test the histogram
    histogram_data1 = np.random.uniform(-3, 3, 1000).tolist()
    histogram_data2 = np.random.normal(0, 1, 1000).tolist()
    histogram_data_dict = {'uniform': {'y': histogram_data1}, 'normal': {'y': histogram_data2}}
    # make a list of a linear and a quadratic function, so we can test the multiline plot
    multiplot_data1 = np.linspace(0, 10, 100)
    multiplot_data2 = multiplot_data1 ** 2
    muliplot_data_dict = {'linear': {'y': multiplot_data1}, 'quadratic': {'y': multiplot_data2, 'x': multiplot_data1}}

    print('Testing multiline plot whitemode')
    multiline_plot('Test multiline plot', muliplot_data_dict)
    print('Testing multiline plot darkmode')
    multiline_plot('Test multiline plot', muliplot_data_dict, darkmode=True)
    print('finished testing multiline plot')

    print('Testing histogram whitemode')
    histogram('Test histogram', histogram_data_dict)
    print('Testing histogram darkmode')
    histogram('Test histogram', histogram_data_dict, darkmode=True)
    print('finished testing histogram')