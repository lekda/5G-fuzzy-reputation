"""utils for plotting the fairness of the reputation system 
includes the distribution of interactions between participants, their number, ... 

"""

import copy
import json
from copy import deepcopy
from pathlib import Path
from turtle import color
from typing import Dict, Iterator, List, OrderedDict, Tuple, Optional, Any

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()


import numpy as np
from omegaconf import ListConfig
from plot_utils import (
    load_participants_interactions,
    load_participants_interactions_overtime,
    colors,
    markers,
    edge_colors,
    hatches,
)


def plot_per_peer_interactions(
    path: str,
    title: str = "Interactions per peer over the full simulation",
    outcome: str = "total",
    out_path: str = "",
    font_size: int = 12,
    fuzzy=False,
    multiseed=False,
):
    """Ordered stick graph of the peer total number of interactions.

    Args:
        path (str): Path to extract results from.
        title (str, optional): Name of the plot. Defaults to "".
        out_path (str, optional): file in which the plot should be saved. Defaults to "".
        fontsize (int, optional): Fontsize used in the graph. Defaults to 12.
        fuzzy (bool, optional): Plot separated participants category. Defaults to False.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
    """

    # Trier par catégorie et faire 3 ax.bar
    fuzzy_category = ["urllc", "embb", "mmtc"]
    l_colors = deepcopy(colors)

    res: Dict[str, List[int]] = {}
    if fuzzy:
        p_interactions: Dict[str, int] = load_participants_interactions(
            Path(path), multiseed=multiseed
        )
        for c in fuzzy_category:
            res[c] = [v for k, v in p_interactions.items() if c in k]
    else:
        res["Peer"] = list(
            load_participants_interactions(Path(path), multiseed=multiseed).values()
        )
    handles: List[Patch] = []
    fig, ax = plt.subplots()
    x_pos = np.arange(len(np.concatenate(list(res.values()))))
    i: int = 0

    for k, v in res.items():
        c = l_colors.pop(0)
        v.sort(reverse=True)
        ax.bar(x_pos[i : i + len(v)], v, color=c)  # ajouter couleur içi
        i += len(v)
        handles.append(
            Patch(linewidth=2, label=k, color=c),
        )
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    ax.legend(
        handles=handles,
        loc="upper right",
    )
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel="Peer",
        ylabel="Total number of interactions",
    )

    plt.show()


def plot_per_peer_interactions_overtime(
    path: str,
    title: str = "",
    out_path: str = "",
    font_size: int = 12,
    zoomed_in: bool = False,
    multiseed=False,
):
    """For every peer plot the number of interactions at each timeframe.

    Args:
        path (str): Path to extract results from.
        title (str, optional): Name of the plot. Defaults to "".
        out_path (str, optional): file in which the plot should be saved. Defaults to "".
        fontsize (int, optional): Fontsize used in the graph. Defaults to 12.
        zoomed_in (bool, optional): Remove peers whose max value is >= 3 * mean . Defaults to False.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
    """
    interactions = load_participants_interactions_overtime(Path(path), multiseed=False)

    fig, ax = plt.subplots()
    for l in interactions.values():
        ax.plot(
            list(range(1, 11)),
            l,
            color=edge_colors[0],
        )
    legend_elements: List[Any] = []
    legend_elements.append(
        Line2D([0], [0], color=edge_colors[0], lw=2, label="a single peer"),
    )

    # Adaptive y-axis
    if zoomed_in:
        m = np.mean(np.array(list(interactions.values())))
        removed: List[str] = []
        for p in interactions:
            if max(interactions[p]) > 3 * m:
                removed.append(p)
        it = {k: v for k, v in interactions.items() if k not in removed}
        plt.ylim(0, np.max(np.array(list(it.values()))))

    # Legend
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set(
        # axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel="Time periods",
        ylabel="Peers interctions",
    )
    ax.set_xticks(list(range(1, 11)))

    rcParams.update({"font.size": font_size})
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_per_peer_interactions(Path("./results/run/2024-01-26/11-55-04"))
