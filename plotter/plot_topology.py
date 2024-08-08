"""Contains the functions used to plot the topology metrics
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
import string

import numpy as np


from plot_utils import load_topology_stats, colors


# Matplotlib / Seaborn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

sns.set_theme()


def plot_max_capacity(
    path: Path,
    title: str = "Cumulated capacity on every access network zone in the network",
    multiseed: bool = False,
    outpath: Path = "",
):
    ts: Any = load_topology_stats(path, multiseed=multiseed)
    ans_capacity: Dict[str, float] = dict(
        sorted(ts["an_max_capacity"].items(), key=lambda item: item[1])
    )

    fig, ax = plt.subplots()

    # Create bars
    ax.bar(ans_capacity.keys(), ans_capacity.values())
    # plt.tick_params(
    #     axis="x",  # changes apply to the x-axis
    #     which="both",  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False,
    # )  # labels along the bottom edge are off
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel="Access networks",
        ylabel="Total capacity",
    )
    # plt.xticks(rotation=45)
    plt.xticks(rotation="vertical")
    if outpath:
        fig.savefig(outpath)

    plt.show()


def plot_type_capacity(
    path: Path,
    title: str = "Cumulated capacity on every access network zone in the network",
    multiseed: bool = False,
    outpath: Path = "",
):
    """Bar plot of access network capacity per type of possible transactions."""

    ts: Any = load_topology_stats(path, multiseed)
    l_colors = deepcopy(colors)

    l_type_capacity: Dict[str, List[float]] = {"urllc": [], "embb": [], "mmtc": []}

    ans_type_capacity: List[Dict[str, float]] = list(
        ts["an_max_capacity_per_type"].values()
    )

    for an in ans_type_capacity:
        for k, v in an.items():
            l_type_capacity[k].append(v)

    # Une couleur + légende par catégorie.
    handles: List[Patch] = []
    fig, ax = plt.subplots()
    fig.set_size_inches((5, 3))

    x_pos = np.arange(len(np.concatenate(list(l_type_capacity.values()))))
    i: int = 0

    for k, v in l_type_capacity.items():
        c = l_colors.pop(0)
        v.sort(reverse=True)
        ax.bar(x_pos[i : i + len(v)], v, color=c)
        i += len(v)
        handles.append(
            Patch(linewidth=2, label=k, color=c),
        )
    ax.legend(
        handles=handles,
        loc="upper right",
    )
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel="Access network",
        ylabel="Zone capacity",
    )
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    plt.xticks(rotation="vertical")
    if outpath:
        fig.savefig(outpath)

    plt.show()


def plot_an_capacity_per_type(
    path: Path,
    title: str = "Cumulated capacity on every access network zone in the network",
    multiseed: bool = False,
    outpath: Path = "",
):
    """Bar plot of access network capacity per type of possible transactions"""
    # Data init
    ans: Dict[str, Dict[str, float]] = load_topology_stats(path, multiseed)[
        "an_max_capacity_per_type"
    ]

    # Plotter init
    fig, ax = plt.subplots(layout="constrained")
    fig.set_size_inches((5, 3))
    labels_colors: Dict[str:str] = {
        label: colors[i] for i, label in enumerate(list(list(ans.values())[0].keys()))
    }
    handles: List[Patch] = [
        Patch(linewidth=2, label=l, color=c) for l, c in labels_colors.items()
    ]

    nb_x_values: int = len(ans) * (len(next(iter(ans.values()))) + 1)
    x_pos = np.arange(len(ans))
    width = 1 / (len(next(iter(ans.values()))) + 1)

    i: int = 0
    max_y = 0
    for an, capacities in ans.items():
        for label, capacity in capacities.items():
            offset = i * width
            ax.bar(offset, int(capacity), width, color=labels_colors[label])
            max_y = max(max_y, int(capacity))
            i += 1
        i += 1
    ax.legend(
        handles=handles,
        ncol=3,
        loc="upper right",
    )
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel="Geographical zones",
        ylabel="Max simultaneous services",
    )

    ax.set_xticks(
        x_pos,
        ["Zone " + l for l in list(string.ascii_uppercase)[: len(ans)]],
    )
    # plt.tick_params(
    #     axis="x",  # changes apply to the x-axis
    #     which="both",  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False,
    # )  # labels along the bottom edge are off
    plt.ylim(top=max_y + 3)
    plt.xticks(rotation=45)
    if outpath:
        fig.savefig(outpath)
    plt.show()


if __name__ == "__main__":
    plot_max_capacity(Path("./results/run/2024-03-13/14-53-40"))
