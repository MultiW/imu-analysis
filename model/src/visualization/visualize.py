import numpy as np
import matplotlib.pyplot as plt
import math

# import types
from pandas import DataFrame
from numpy import ndarray
from typing import List, Tuple, Optional, Callable

def multiplot(num_plots, plot_f: Callable[[int, any], None]):
    """
    Uses subplots. Useful even for a single chart, because the default plt plot is stateful, 
    so multiple calls to plt.plot() will be show up in the same figure, even when called in different 
    jupyter notebook cells.
    """
    if num_plots == 1:
        plots_layout = (1, 1)
    else:
        plots_layout = (math.ceil(num_plots/2), 2)
    _, axs = plt.subplots(*plots_layout)

    curr_plot_idx = 0
    for i in range(plots_layout[0]):
        for j in range(plots_layout[1]):
            curr_plot: any
            if num_plots == 1:
                curr_plot = axs
            elif plots_layout[0] == 1 or plots_layout[1] == 1:
                curr_plot = axs[curr_plot_idx]
            else:
                curr_plot = axs[i][j]

            plot_f(curr_plot_idx, curr_plot)

            curr_plot_idx += 1
            if curr_plot_idx == num_plots:
                break

    plt.show()