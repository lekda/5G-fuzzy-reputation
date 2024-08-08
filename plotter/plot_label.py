"""utils for reput plotting
"""

import copy
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, OrderedDict, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# just imported for those fancy graph looks
import seaborn as sns

sns.set_theme()

import numpy as np
from omegaconf import ListConfig
from plot_utils import *


def plot_participant_interactions_overtime(
    path: Path,
    labels_filter: List[str] = [""],
    outage: List[Tuple[float, float]] = [],
    multiseed=False,
):
    """

    Args:
        path (str): Path to the run. If multiseed is True, then the path should be a multirun.
        labels_filter List(str): labels of participants that should be selected. By default all participants are selected
        outage (List[Tuple[float,float]]) : Put an emphasis on the forced outage if there are some.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
    """
    _colors = copy.deepcopy(edge_colors)
    legend_elements = []
    fig, ax = plt.subplots()

    p_interactions_overtime: Dict[str, List[int]] = (
        load_participants_interactions_overtime(path, multiseed=multiseed)
    )
    labels = get_labels(path, multiseed=multiseed)

    if labels_filter:
        labels = [l for l in labels if any(f in l for f in labels_filter)]
        p_interactions_overtime = {
            l: v
            for l, v in p_interactions_overtime.items()
            if any(f in l for f in labels_filter)
        }

    for l in labels:
        c = _colors.pop()
        legend_elements.append(
            Line2D([0], [0], color=c, lw=2, label=l),
        )
        p_keys = [l_p for l_p in p_interactions_overtime if l in l_p]
        for p in p_keys:
            p_interactions_overtime[p].insert(0, 0)
            plt.plot(
                range(0, simulation_lenght(path, multiseed) + 1),
                p_interactions_overtime[p],
                color=c,
            )

    # ax.set_ylim(0, max(n_i_overtime.values()))

    if outage:
        add_outage_on_plot(
            plt,
            ax,
            outage,
            y=max(
                max(interactions) for interactions in p_interactions_overtime.values()
            )
            + 2,
            simulation_end=simulation_lenght(path, multiseed),
            bbox_to_anchor=(0.3, 0.80),
        )

    ax.set_xticks([i for i in range(0, simulation_lenght(path, multiseed) + 1)])

    plt.tick_params(labelbottom=False)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 0.95))

    plt.title("Nombre total d'interactions par intervalle de temps")
    plt.xlabel("Intervalle de temps")
    plt.ylabel("Nb interactions")
    plt.show()


def plot_labels_interactions_overtime(
    path: Path,
    labels_filter: List[str] = [""],
    outage: List[Tuple[float, float]] = [],
    title: str = "Mean number of interactions per participant type",
    outpath: str = "",
    small: bool = False,
    print_legend: bool = True,
    multiseed=False,
):
    """Mean number of interactions per participant of each labels.

    Args:
        path (str): Path to the run. If multiseed is True, then the path should be a multirun.
        labels_filter List(str): labels of participants that should be selected. By default all participants are selected
        outage (List[Tuple[float,float]]) : Put an emphasis on the forced outage if there are some.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
    """
    _p_colors = copy.deepcopy(edge_colors)
    _p_markers = copy.deepcopy(primary_markers)
    _s_colors = copy.deepcopy(colors)
    _s_markers = copy.deepcopy(secondary_markers)

    legend_elements = []
    fig, ax = plt.subplots()
    if small:
        fig.set_size_inches((5, 3))

    l_interactions_overtime: Dict[str, Dict[str, float]] = (
        load_mean_labels_interactions_overtime(path, multiseed=multiseed)
    )
    l_interactions_overtime: Dict[str, List[float]] = overtime_dict_to_list(
        l_interactions_overtime
    )
    labels = get_labels(path, multiseed=multiseed)

    if labels_filter:
        labels = [l for l in labels if any(f in l for f in labels_filter)]
        l_interactions_overtime = {
            l: v
            for l, v in l_interactions_overtime.items()
            if any(f in l for f in labels_filter)
        }
    p_labels = get_primary_labels(path, multiseed=multiseed)

    for p_l in p_labels:
        ll = [l for l in labels if p_l in l]
        # Primary color
        label1 = ll[0]
        c = _p_colors.pop()
        m = _p_markers.pop()
        legend_elements.append(
            Line2D([0], [0], color=c, marker=m, lw=2, label=label1),
        )
        l_interactions_overtime[label1].insert(0, 0)
        plt.plot(
            range(0, simulation_lenght(path, multiseed) + 1),
            l_interactions_overtime[label1],
            color=c,
            marker=m,
        )
        # Secondary color if there are several secondary labels
        # for the same primary label.
        if len(ll) > 1:
            label2 = ll[1]
            c = _s_colors.pop()
            m = _s_markers.pop()
            legend_elements.append(
                Line2D([0], [0], color=c, linestyle="--", marker=m, lw=2, label=label2),
            )
            l_interactions_overtime[label2].insert(0, 0)
            plt.plot(
                range(0, simulation_lenght(path, multiseed) + 1),
                l_interactions_overtime[label2],
                linestyle="--",
                color=c,
                marker=m,
            )
    # ax.set_ylim(0, max(n_i_overtime.values()))
    # TODO, shift l'outage de 1
    if outage:
        add_outage_on_plot(
            plt,
            ax,
            outage,
            y=max(
                max(interactions) for interactions in l_interactions_overtime.values()
            )
            + 2,
            simulation_end=simulation_lenght(path, multiseed),
            bbox_to_anchor=(1.0, 0.30) if small else (1.35, 0.50),
            print_legend=print_legend,
        )

    ax.set_xticks([i for i in range(0, simulation_lenght(path, multiseed) + 1)])

    plt.tick_params(labelbottom=False)
    if print_legend:
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 0.95))
    plt.title(title)
    plt.xlabel("Time intervals")
    plt.ylabel("Mean total interactions")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def plot_interactions_overtime(path: str, multiseed=False):
    """

    Args:
        path (str): Path to the run. If multiseed is True, then the path should be a multirun.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
    """
    fig, ax = plt.subplots()
    interactions = load_nb_interactions_overtime(Path(path), multiseed)

    ax.bar(list(interactions.keys()), list(interactions.values()))

    ax.set_ylabel("Number of interactions")
    plt.show()


def plot_negative_interactions_overtime(
    paths: List[Tuple[str, str]],
    multiseed=False,
    fuzzy: bool = False,
    outage: List[Tuple[float, float]] = [],
    title: str = "Total negative interactions per time interval",
    outpath: Path = "",
):
    """

    Args:
        paths (List[Tuple[str,str]]): List of label + Path to the run. If multiseed is True, then the path should be a multirun.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
        outage (List[Tuple[float,float]]) : Put an emphasis on the forced outage if there are some.

    """
    fig, ax = plt.subplots()
    marks = deepcopy(markers)
    i: int = 0
    fig.set_size_inches((5, 3))
    max_y = 0
    for label, path in paths:
        if fuzzy:
            n_i_overtime = fuzzy_load_negative_overtime(Path(path), multiseed)
        else:
            n_i_overtime = load_negative_overtime(Path(path), multiseed)
        # ax.set_ylim(0, max(n_i_overtime.values()))
        max_y = max(max_y, max(n_i_overtime.values()))
        plt.plot(
            n_i_overtime.keys(),
            n_i_overtime.values(),
            marker=marks.pop(0),
            markevery=(i, len(paths)),
            markersize=8,
            linewidth=2,
            label=label,
        )
        i += 1

    if outage:
        add_outage_on_plot(
            plt,
            ax,
            outage,
            y=max_y,
            simulation_end=simulation_lenght(path, multiseed),
            bbox_to_anchor=(1.0, 0.95),
        )

    plt.tick_params(labelbottom=False)
    plt.title(title)
    plt.xlabel("Time intervals")
    plt.ylabel("Total negative interactions")
    plt.legend(loc="center right")

    if outpath:
        fig.savefig(outpath)
    plt.show()


def plot_negative_interactions_overtime_no_reput(
    paths: List[Tuple[str, List[Tuple[str, str]]]],
    multiseed=False,
    fuzzy: bool = False,
    outage: List[Tuple[float, float]] = [],
    title: str = "Total negative interactions per time interval",
    outpath: Path = "",
):
    """plot_negative_interactions_overtime but with some legend tuning for readability when adding a no_reput run.

    Args:
        paths (List[Tuple[str,List[Tuple[str, str]]]]): label of big categories (eg no_reput; reput)+ same as plot_negative_interactions_overtime. If multiseed is True, then the path should be a multirun.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
        outage (List[Tuple[float,float]]) : Put an emphasis on the forced outage if there are some.

    """
    fig, ax = plt.subplots()
    marks = deepcopy(markers)
    i: int = 0
    fig.set_size_inches((5, 3))
    max_y = 0
    locations: List[Tuple[float, float]] = [(1.0, 0.6), (1.0, 0.85)]
    for category, tup in paths:
        lines = []
        labels = []
        for label, path in tup:
            if fuzzy:
                n_i_overtime = fuzzy_load_negative_overtime(Path(path), multiseed)
            else:
                n_i_overtime = load_negative_overtime(Path(path), multiseed)
            # ax.set_ylim(0, max(n_i_overtime.values()))
            max_y = max(max_y, max(n_i_overtime.values()))
            (l,) = plt.plot(
                n_i_overtime.keys(),
                n_i_overtime.values(),
                marker=marks.pop(0),
                markevery=(i, len(paths)),
                markersize=8,
                linewidth=2,
                label=label,
            )
            i += 1
            labels.append(label)
            lines.append(l)
        l1 = ax.legend(lines, labels, title=category, bbox_to_anchor=locations.pop(0))
        ax.add_artist(l1)

    if outage:
        add_outage_on_plot(
            plt,
            ax,
            outage,
            y=max_y,
            simulation_end=simulation_lenght(path, multiseed),
            bbox_to_anchor=(1.0, 0.95),
        )

    plt.tick_params(labelbottom=False)
    plt.title(title)
    plt.xlabel("Time intervals")
    plt.ylabel("Total negative interactions")

    if outpath:
        fig.savefig(outpath)
    plt.show()


def plot_cumulative_negative_interactions(
    paths: List[Tuple[str, str]],
    multiseed=False,
    fuzzy: bool = False,
    outage: List[Tuple[float, float]] = [],
    title: str = "Total negative interactions per time interval",
    outpath: Path = "",
):
    """

    Args:
        paths (List[Tuple[str,str]]): List of label + Path to the run. If multiseed is True, then the path should be a multirun.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
        outage (List[Tuple[float,float]]) : Put an emphasis on the forced outage if there are some.

    """
    fig, ax = plt.subplots()
    marks = deepcopy(markers)
    for label, path in paths:
        if fuzzy:
            n_i_overtime = fuzzy_load_negative_overtime(Path(path), multiseed)
        else:
            n_i_overtime = load_negative_overtime(Path(path), multiseed)
        n_i_overtime = cumulate_dict_key(n_i_overtime)
        ax.set_ylim(0, max(n_i_overtime.values()))
        plt.plot(
            n_i_overtime.keys(),
            n_i_overtime.values(),
            marker=marks.pop(0),
            markersize=8,
            linewidth=2,
            label=label,
        )
    if outage:
        add_outage_on_plot(
            plt, ax, outage, simulation_end=simulation_lenght(path, multiseed)
        )

    plt.tick_params(labelbottom=False)
    plt.title(title)
    plt.xlabel("Intervalle de temps")
    plt.ylabel("Nb interactions négatives ")
    plt.legend()
    if outpath:
        fig.savefig(outpath)
    plt.show()


def mean_interactions_label_overtime():
    pass


def plot_reput_overtime(
    path: str,
    title: str = "",
    out_path: str = "",
    labels_filter: List[str] = [],
    font_size: int = 12,
    outage: List[Tuple[float, float]] = [],
    multiseed=False,
):
    """Plot participants reputation overtime.

    Args:
        path (str): Path to extract results from.
        title (str, optional): Name of the plot. Defaults to "".
        out_path (str, optional): file in which the plot should be saved. Defaults to "".
        labels_filter (List[str], optional): If specified only show participants with one of the specified string in their label.
        fontsize (int, optional): Fontsize used in the graph. Defaults to 12.
        outage (List[Tuple[float,float]]) : Put an emphasis on the forced outage if there are some.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
    """
    # Je veux un trait part participant.
    # Trait de couleur différentes en fonction du label des participants.
    # oscillatory_outage = []
    path = Path(path)
    _colors = copy.deepcopy(edge_colors)
    fig, ax = plt.subplots()
    fig.set_size_inches((5, 3))

    labels = get_labels(path, multiseed=multiseed)
    if labels_filter:
        labels = [l for l in labels if any(f in l for f in labels_filter)]
    legend_elements = []
    reput: Dict[str, List[float]] = {}
    x_len: int
    for l in labels:
        c = _colors.pop()
        legend_elements.append(
            Line2D([0], [0], color=c, lw=2, label=l),
        )
        # Rajouter içi une première étape ou la réput est à 1.
        reput[l] = load_mean_reput_on_label_overtime(path, label=l, multiseed=multiseed)
        for r in reput[l].values():
            r.insert(0, 1.0)
            x_len = len(r) + 1
            ax.plot(
                list(range(0, x_len - 1)),
                r,
                color=c,  # change color and marker per label
            )
    ax.set(
        # axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel="Time periods",
        ylabel="Peers reputation",
    )
    if outage:
        add_outage_on_plot(
            plt, ax, outage, simulation_end=simulation_lenght(path, multiseed)
        )

    ax.set_xticks([i for i in range(0, simulation_lenght(path, multiseed) + 1)])

    plt.tick_params(labelbottom=False)
    bot, top = plt.ylim()
    plt.ylim((bot, 1.0))
    # Add a legend

    ax.legend(handles=legend_elements, loc="lower left")
    rcParams.update({"font.size": font_size})
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def plot_mean_reput_overtime(
    path: str,
    title: str = "",
    out_path: str = "",
    labels_filter: List[str] = [],
    font_size: int = 12,
    outage: List[Tuple[float, float]] = [],
    multiseed=False,
):
    """Plot participants reputation overtime.

    Args:
        path (str): Path to extract results from.
        title (str, optional): Name of the plot. Defaults to "".
        out_path (str, optional): file in which the plot should be saved. Defaults to "".
        labels_filter (List[str], optional): If specified only show participants with one of the specified string in their label.
        fontsize (int, optional): Fontsize used in the graph. Defaults to 12.
        outage (List[Tuple[float,float]]) : Put an emphasis on the forced outage if there are some.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
    """
    # Je veux un trait part participant.
    # Trait de couleur différentes en fonction du label des participants.
    # oscillatory_outage = []
    path = Path(path)
    _colors = copy.deepcopy(edge_colors)
    _s_colors = copy.deepcopy(colors)
    _p_markers = copy.deepcopy(primary_markers)
    _s_markers = copy.deepcopy(secondary_markers)

    fig, ax = plt.subplots()
    fig.set_size_inches((5, 3))

    labels = get_labels(path, multiseed=multiseed)
    if labels_filter:
        labels = [l for l in labels if any(f in l for f in labels_filter)]
    legend_elements = []
    reput: Dict[str, List[float]] = {}
    x_len: int
    for l in labels:
        abnormal: bool = ("outage" in l) or ("oscillatory" in l)
        if abnormal:
            c = _s_colors.pop()
            m = _s_markers.pop()
            ls = "--"
        else:
            c = _colors.pop()
            m = _p_markers.pop()
            ls = "-"
        legend_elements.append(
            Line2D([0], [0], color=c, linestyle=ls, marker=m, lw=2, label=l),
        )
        # Rajouter içi une première étape ou la réput est à 1.
        reput_values = load_mean_reput_on_label_overtime(
            path, label=l, mean_labels=True, multiseed=multiseed
        )[l]
        reput_values.insert(0, 1.0)
        x_len = len(reput_values) + 1
        ax.plot(
            list(range(0, x_len - 1)),
            reput_values,
            linestyle=ls,
            marker=m,
            color=c,
        )
    ax.set(
        # axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel="Time periods",
        ylabel="Peers reputation",
    )
    if outage:
        add_outage_on_plot(
            plt, ax, outage, simulation_end=simulation_lenght(path, multiseed)
        )

    ax.set_xticks([i for i in range(0, simulation_lenght(path, multiseed) + 1)])

    plt.tick_params(labelbottom=False)
    bot, top = plt.ylim()
    plt.ylim((bot, 1.0))
    # Add a legend

    ax.legend(handles=legend_elements, loc="lower left")
    rcParams.update({"font.size": font_size})
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def boxplot_label_overtime(
    path: str,
    title: str = "",
    out_path: str = "",
    font_size: int = 12,
    labels: List[str] = [],
    multiseed=False,
):
    """Boxplot of a metric for participants from different baselines.

    Args:
        path (str): Path to extract results from.
        title (str, optional): Name of the plot. Defaults to "".
        out_path (str, optional): file in which the plot should be saved. Defaults to "".
        fontsize (int, optional): Fontsize used in the graph. Defaults to 12.
        labels (List[str],optional) : labels to add on each box.
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.
    """
    a: Dict[str, Dict[str, List[int]]] = load_label_interactions_number_overtime(
        Path(path), multiseed
    )
    nb_labels: int = len(list(a.values())[0].keys())
    labels = [
        item for sublist in zip(a.keys(), [""] * len(a.keys())) for item in sublist
    ]
    mustaches: List[List[int]] = [v for q in a.values() for v in list(q.values())]
    fig, ax = plt.subplots()

    ax.set(
        # axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel="Time periods",
        ylabel="Participant number of interactions",
    )

    plt.plot([7.0, 7.0], [7.0, 350.0], "r-", lw=2)  # Red straight line
    plt.plot([15.0, 15.0], [15.0, 350.0], "r-", lw=2)  # Red straight line

    ax.set_xticklabels(labels)
    rcParams.update({"font.size": font_size})

    boxes = ax.boxplot(mustaches, patch_artist=True)
    cols = [c for _ in range(len(a.keys())) for c in colors[0:2]]
    edge_cols = [c for _ in range(len(a.keys())) for c in edge_colors[0:2]]
    hatchs = [c for _ in range(len(a.keys())) for c in hatches[0:2]]
    for box, w1, w2, c1, c2, med, f, color, edge_color, h in zip(
        boxes["boxes"],
        boxes["whiskers"][::2],
        boxes["whiskers"][1::2],
        boxes["caps"][::2],
        boxes["caps"][1::2],
        boxes["medians"],
        boxes["fliers"],
        cols,
        edge_cols,
        hatchs,
    ):
        for el in [box, w1, w2, c1, c2]:
            el.set(color=edge_color, linewidth=2)
        med.set(color="deeppink", linewidth=2)
        f.set(marker="o", color="maroon", alpha=0.5)

        box.set(facecolor=color)
        box.set(hatch="/")

    for flier in boxes["fliers"]:
        flier.set(marker="o", color="#e7298a", alpha=0.5)

    # Duplicate the number of interactions for each labels.
    # control for multiseed
    mean_interactions_per_participant = [
        i
        for i in load_mean_interactions_overtime(Path(path), multiseed)
        for _ in range(nb_labels)
    ]

    ax.plot(
        list(range(1, 21)),
        mean_interactions_per_participant,
        color="tab:purple",
        marker="d",
        markevery=2,
    )
    plt.plot([], c="purple", marker="d", label="Mean number of interactions")

    # Custom legend to better include mustache plot.
    legend_elements = [
        Patch(
            facecolor=colors[0], edgecolor=edge_colors[0], linewidth=2, label="Benign"
        ),
        Patch(
            facecolor=colors[1], edgecolor=edge_colors[1], linewidth=2, label="Outage"
        ),
        Line2D([0], [0], color="tab:purple", lw=2, label="Mean number of interactions"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()


# def transaction_success_rate(in_dir: List[str], labels : List[str]):


def plot_stats_labels(in_dirs: List[str], names: List[str], multiseed=False):
    """Plot the mean number of interactions per label on multiple run

    Args:
        in_dirs (List[str]): Two run / multirun for comparison.
        names (List[str]): Names associated with the run
        multiseed (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
    """
    if len(in_dirs) != len(names):
        raise ValueError("There should be one name per experiment")

    # move to plot utils ?
    results: Dict[str, List[float]] = {}
    for dir, name in zip(in_dirs, names):
        l_stats: Dict[str, Dict[str, int]] = load_labels_stats(Path(dir), multiseed)
        l_nb_interactions: Dict[str, int] = {
            l: stats["total"] for l, stats in l_stats.items()
        }
        l_nb = load_nb_participants_labels(Path(dir), multiseed)
        l_mean_interactions: Dict[str, float] = {}
        for l in l_nb_interactions:
            if not l in results:
                results[l] = []
            results[l].append(l_nb_interactions[l] / l_nb[l])

    # TODO make it works for arbitrary number of labels
    position = np.arange(len(in_dirs))
    width = 0.35  # 1.0/(nb_labels+1) ?
    fig, ax = plt.subplots()

    ax.bar(position - width / 2, results["good"], width)
    ax.bar(position + width / 2, results["bad"], width)

    ax.set_xticks(position)
    ax.set_xticklabels(names)

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c="#1f77b4", label="good")
    plt.plot([], c="#ff7f0e", label="bad")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.show()


if __name__ == "__main__":

    pass
