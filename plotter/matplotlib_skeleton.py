# this will be the skeleton for a set of plotter functions
# written in matplotlib
# the idea is to have a set of functions that can be called to perform basic plots, such as a multiline plot and a histogram
# with a unified style language
# the seciondary function of this is to serve as code repository for more elaborate plotter functions


import matplotlib.pyplot as plt
import numpy as np

# use company colors as default
rex_orange = '#fa6140'
rex_offwhite = '#fffdef'
rex_darkblue = '#1a363d'
rex_green = '#c7f296'

linecolors = [rex_orange, rex_green]
spinecolor = rex_orange #change to darkblue if too aggressive

default_font_size = 13
# ToDo: implement T.rex AI fonts
def get_company_linecolors(darkmode=False):
    if darkmode:
        return [rex_orange, rex_green, rex_darkblue]
    else:
        return [rex_orange, rex_green, rex_offwhite]
def finish_plot(fig, title, darkmode=True, transparent=True, disable_axes_ticks=False):
    if transparent:
        plt.gcf().patch.set_facecolor('none')
    elif not transparent and darkmode:
        plt.gcf().patch.set_facecolor('black')
    axes = plt.gcf().get_axes()
    fontcolor = rex_offwhite if darkmode else rex_darkblue

    fig.set_size_inches(5, 5)

    # set the title, if we have multiple axes then we will have to set the title for the center plot in the top column
    fig.suptitle(title, fontsize=default_font_size, color=fontcolor)

    for ax in axes:

        if disable_axes_ticks:
            # # hide x, y axis
            ax.set_xticks([])
            ax.set_yticks([])

        ax.spines['bottom'].set_color(rex_orange)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color(rex_orange)
        ax.title.set_color(fontcolor)
        ax.xaxis.label.set_color(fontcolor)
        ax.yaxis.label.set_color(fontcolor)
        ax.tick_params(axis='x', colors=rex_orange)



        if transparent:
            plt.gcf().patch.set_facecolor('none')
            axes = plt.gcf().get_axes()

            for ax in axes:
                ax.set_facecolor('none')
                ax.tick_params(axis='y', colors=rex_orange)
                ax.title.set_color(rex_orange)
                # set the legend box to have no facecolor and white font color
                ax.legend(facecolor='none', edgecolor='none', labelcolor=fontcolor, loc='upper left')
        else:
            # should be automatic
            ax.legend(facecolor='white', edgecolor='none', labelcolor=fontcolor, loc='upper left')



    plt.savefig(title + 'darkmode' if darkmode else title + '.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
def multiline_plot(title, data: dict(), darkmode=False, transparent=True, disable_axes_ticks=False):
    # this function will take a list of data and plot it as a multiline plot
    # title is the title of the plot
    # x_axis_title is the title of the x axis
    # y_axis_title is the title of the y axis

    # data is a dictionary, and expected to have the following format:
        # data = {'vector_name': vector_dict, 'vector_name2': vector_dict, ...}
        # where vector_dict is a dictionary with the following format:
            # vector_dict = {'x': x_data, 'y': y_data, 'y_label': y_label, 'x_label': x_label}
    # if x is not provided, we will assume that the x axis is the index of the vector

    plt.rcParams.update({'font.size': default_font_size})

    for vector_name, vector_dict in data.items():
        if 'y_label' not in vector_dict:
            vector_dict['y_label'] = vector_name
        if 'x_label' not in vector_dict:
            vector_dict['x_label'] = 'Steps'

    #figure out if our labels are the same:
    y_labels = [vector_dict['y_label'] for vector_dict in data.values()]
    if len(set(y_labels)) > 1:
        print('conflicting y labels, defaulting to: ', y_labels[0])
    y_label = y_labels[0]

    x_labels = [vector_dict['x_label'] for vector_dict in data.values()]
    if len(set(x_labels)) > 1:
        print('conflicting x labels, defaulting to: ', x_labels[0])
    x_label = x_labels[0]


    fig, ax = plt.subplots()
    # if the number of vectors is less or equal to 3, we will use the company colors
    if len(data) <= 3:
        linecolors = get_company_linecolors(darkmode)
    else:
        linecolors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for vector_name, vector_dict in data.items():
        # check if x is provided
        if 'x' not in vector_dict:
            vector_dict['x'] = list(range(len(vector_dict['y'])))


        ax.plot(vector_dict['x'], vector_dict['y'], label=vector_name, color=linecolors.pop(0))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    finish_plot(fig, title, darkmode=darkmode, transparent=transparent, disable_axes_ticks=disable_axes_ticks)

def multi_plot(title, data: dict(), num_rows='auto', num_columns='auto', darkmode=False, transparent=True, disable_axes_ticks=False):
    # this function will take a dict of data and plot it as a multiplot
    # title is the title of the plot
    # x_axis_title is the title of the x axis
    # y_axis_title is the title of the y axis

    # data is a dictionary, and expected to have the following format:
        # data = {'vector_name': vector_dict, 'vector_name2': vector_dict, ...}
        # where vector_dict is a dictionary with the following format:
            # vector_dict = {'x': x_data, 'y': y_data, 'y_label': y_label, 'x_label': x_label}
    # if x is not provided, we will assume that the x axis is the index of the vector

    # num_rows and num_columns are the number of rows and columns to use in the plot
    # if not provided, it will be calculated automatically to ensure even distribution

    plt.rcParams.update({'font.size': default_font_size})

    if num_rows == 'auto' and num_columns == 'auto':
        # calculate the number of rows and columns to use
        num_plots = len(data)
        num_rows = int(np.sqrt(num_plots))
        num_columns = int(np.ceil(num_plots / num_rows))
    elif num_rows == 'auto':
        num_rows = int(np.ceil(len(data) / num_columns))
    elif num_columns == 'auto':
        num_columns = int(np.ceil(len(data) / num_rows))
    else: # both are provided, make sure they fit the data
        assert num_rows * num_columns >= len(data), 'num_rows * num_columns must be greater than or equal to the number of plots'

    fig, axs = plt.subplots(num_rows, num_columns)
    for vector_name, vector_dict in data.items():
        # check if x is provided
        if 'x' not in vector_dict:
            vector_dict['x'] = list(range(len(vector_dict['y'])))

        if 'y_label' not in vector_dict:
            vector_dict['y_label'] = vector_name
        if 'x_label' not in vector_dict:
            vector_dict['x_label'] = 'Steps'

        # get the current axis
        ax = axs.flatten()[list(data.keys()).index(vector_name)]
        ax.plot(vector_dict['x'], vector_dict['y'], label=vector_name, color=rex_orange)
        ax.set_ylabel(vector_dict['y_label'])
        ax.set_xlabel(vector_dict['x_label'])


    finish_plot(fig, title, darkmode=darkmode, transparent=transparent, disable_axes_ticks=disable_axes_ticks)

def histogram(title, data, bins=None, darkmode=False, transparent=True, disable_axes_ticks=False):
    # this function will take a list of data and plot it as a histogram
    # title is the title of the plot
    # data is a dictionary, and expected to have the following format:
        # data = {'vector_name': vector_dict, 'vector_name2': vector_dict, ...}
        # where vector_dict is a dictionary with the following format:
        # vector_dict = {'y': y_data}
        # due to histogram format x data is not needed - it can be provided but will not be used
    # bins is the number of bins to use in the histogram, if not provided, it will be calculated automatically by combining all vectors to ensure even distribution

    plt.rcParams.update({'font.size': default_font_size})



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
    if len(data) <= 3:
        color = get_company_linecolors(darkmode)
    else:
        color = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for vector_name, vector_dict in data.items():
        ax.hist(vector_dict['y'], label=vector_name, bins=bins, alpha=alpha, color=color.pop(0))

    finish_plot(fig, title, darkmode=darkmode, transparent=transparent, disable_axes_ticks=disable_axes_ticks)
if __name__ == '__main__':
    # make a list of random numbers so we can test the histogram
    histogram_data1 = np.random.uniform(-3, 3, 1000).tolist()
    histogram_data2 = np.random.normal(0, 1, 1000).tolist()
    histogram_data_dict = {'uniform': {'y': histogram_data1}, 'normal': {'y': histogram_data2}}
    # make a list of a linear and a quadratic function, so we can test the multiline plot
    multiplot_data1 = np.linspace(0, 10, 100)
    multiplot_data2 = multiplot_data1 ** 2
    muliplot_data_dict = {'linear': {'y': multiplot_data1, 'y_label': 'test1'}, 'quadratic': {'y': multiplot_data2, 'x': multiplot_data1, 'x_label': 'test2'}}

    print('Testing multiline plot whitemode')
    multiline_plot('Test multiline plot', muliplot_data_dict)
    print('Testing multiline plot darkmode')
    multiline_plot('Test multiline plot', muliplot_data_dict, darkmode=True)
    print('finished testing multiline plot')

    print('Testing multiplot whitemode')
    multi_plot('Test multiplot', muliplot_data_dict)
    print('Testing multiplot darkmode')
    multi_plot('Test multiplot', muliplot_data_dict, darkmode=True)

    print('Testing histogram whitemode')
    histogram('Test histogram', histogram_data_dict)
    print('Testing histogram darkmode')
    histogram('Test histogram', histogram_data_dict, darkmode=True)
    print('finished testing histogram')