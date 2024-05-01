# this will be the skeleton for a set of plotter functions
# written in plotly
# the idea is to have a set of functions that can be called to perform basic plots, such as a multiline plot and a histogram
# with a unified style language
# the seciondary function of this is to serve as code repository for more elaborate plotter functions

import kaleido
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import plotly.io as pio
pio.kaleido.scope.mathjax = None


# make a list of random numbers so we can test the histogram
histogram_data = np.random.randn(1000).tolist()

# make a list of a linear and a quadratic function, so we can test the multiline plot
multiplot_data1 = np.linspace(0, 10, 100)
multiplot_data2 = multiplot_data1 ** 2
muliplot_data_dict = {'linear': {'y': multiplot_data1}, 'quadratic': {'y': multiplot_data2, 'x': multiplot_data1}}


def multiline_plot(title, x_axis_title, y_axis_title, data: dict()):
    # this function will take a list of data and plot it as a multiline plot
    # title is the title of the plot
    # x_axis_title is the title of the x axis
    # y_axis_title is the title of the y axis

    # data is a dictionary, and expected to have the following format:
        # data = {'vector_name': vector_dict, 'vector_name2': vector_dict, ...}
        # where vector_dict is a dictionary with the following format:
            # vector_dict = {'x': x_data, 'y': y_data}
    # if x is not provided, we will assume that the x axis is the index of the vector

    # create the figure
    fig = go.Figure()
    for vector_name, vector_dict in data.items():
        # check if x is provided
        if 'x' not in vector_dict:
            vector_dict['x'] = list(range(len(vector_dict['y'])))

        fig.add_trace(go.Scatter(x=vector_dict['x'], y=vector_dict['y'], mode='lines', name=vector_name))

    # update the layout
    fig.update_layout(title=title, xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    # ensure consistent sizing
    fig.update_layout(width=800, height=600)
    # save plot as png

    # fig.show()


    img_bytes  = fig.to_image(format='png',  engine='kaleido')
    with open(title+'.png', 'wb') as f:
        f.write(img_bytes )
    # # save the plot as 'title' png


if __name__ == '__main__':
    print('Testing multiline plot')
    multiline_plot('Test multiline plot', 'x', 'y', muliplot_data_dict)
    print('finished testing multiline plot')