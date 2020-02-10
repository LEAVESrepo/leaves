import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heat_map(mat_for_heatmap, max_, save=False):
    df_for_heatmap = pn.DataFrame(data=mat_for_heatmap)

    ax = sns.heatmap(data=df_for_heatmap, cmap="Purples", cbar_kws={'alpha': 0.8}, vmin=0, vmax=max_)
    ax.invert_yaxis()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)

    plt.show()

    if save:
        ax.get_figure().savefig("/Users/marcoromanelli/Desktop/htmp.png", dpi=3000)
    return ax
