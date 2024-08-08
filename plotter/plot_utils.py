""" utils for plotter"""

from ast import Raise
import json
from math import floor
import statistics
from pathlib import Path
from typing import Callable, Dict, Iterable, List, OrderedDict, Tuple, Any
from omegaconf import OmegaConf
from unittest import result
import numpy as np

from matplotlib import path
from matplotlib.lines import Line2D


#########################
# plotter cosmetic utils
#########################

markers = ["o", "D", "v", "*", "+", "^", "p", ".", "P", "<", ">", "X"]

primary_markers = ["o", "X", "D", "P"]
secondary_markers = [".", "x", "d", "+"]
colors = ["skyblue", "lightcoral", "moccasin", "darkseagreen", "palevioletred", "peru"]
edge_colors = [
    "steelblue",
    "brown",
    "goldenrod",
    "seagreen",
    "mediumvioletred",
    "sienna",
]
hatches = ["/", "\\", "x", ".", "|", "-", "+", "o", "O", "*"]


#########################
# Matplotlib manipulation
#########################
def add_outage_on_plot(
    plt,
    ax,
    outage,
    y=1.0,
    simulation_end: float = 10,
    bbox_to_anchor=(1.0, 0.98),
    print_legend: bool = True,
):
    online = find_gaps(outage, 0, simulation_end)
    for t_period in outage:
        start, end = t_period
        plt.hlines(
            linewidth=10,
            y=y,
            xmin=start,
            xmax=end,
            color="#d62728",
        )
    for t_period in online:
        start, end = t_period
        plt.hlines(
            linewidth=10,
            y=y,
            xmin=start,
            xmax=end,
            color="#2ca02c",
        )
    ax.spines["top"].set_visible(False)
    if print_legend:
        legend = ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#d62728",
                    lw=5,
                    label="period",
                ),
                Line2D([0], [0], color="#2ca02c", lw=5, label="online period"),
            ],
            bbox_to_anchor=bbox_to_anchor,
        )
        ax.add_artist(legend)


#########################
# hydra conf manipulation
#########################
def is_fuzzy_run(path: Path) -> bool:
    if (path / ".hydra/config.yaml").exists():
        conf = OmegaConf.load(path / ".hydra/config.yaml")
    else:
        conf = OmegaConf.load(path / "0/.hydra/config.yaml")
    return conf.simulation.fuzzy


def simulation_lenght(path: Path, multiseed: bool = False) -> bool:
    if multiseed:
        path = get_dirs_from_multirun(path)[0]

    conf = OmegaConf.load(path / ".hydra/config.yaml")
    return conf.simulation.sim_length


#########################
# path/dirs utils
#########################


def dir_from_multirun(*in_dirs: str) -> List[Path]:
    """Extracts path from a multi-run

    Args:
        in_dir (str):multi-run path

    Returns:
        List[str]: list of hydra subfolder in the specified folder
    """
    dirs: List[Path] = []
    for multipath in in_dirs:
        dirs += [
            p
            for p in Path(multipath).iterdir()
            if p.is_dir() and Path(p / ".hydra").exists()
        ]
    return dirs


def get_dirs_from_multirun(*in_dirs: str) -> List[Path]:
    """Extracts path from a multi-run

    Args:
        in_dir (str):multi-run path

    Returns:
        List[str]: list of hydra subfolder in the specified folder
    """
    dirs: List[Path] = []
    for multipath in in_dirs:
        dirs += [
            p
            for p in Path(multipath).iterdir()
            if p.is_dir() and Path(p / ".hydra").exists()
        ]
    return dirs


def check_paths(*spaths: Path) -> None:
    """Raise exception is the submitted path doesn't exist"""
    for p in spaths:
        if not p.absolute().exists():
            raise ValueError(f"Path {p} does not exist")


#########################
# Dict utils
#########################


def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two nested dict that share the same keys.

    Args:
        d1 (Dict[str, Any]): _description_
        d2 (Dict[str, Any]): _description_

    Returns:
        Dict[str, Any]: Merged dict, for similar keys, values are added.
    """
    result_dict = d1.copy()
    for key, value in d2.items():
        if (
            key in result_dict
            and isinstance(result_dict[key], dict)
            and isinstance(value, dict)
        ):
            result_dict[key] = merge_dicts(d1[key], value)
        else:
            result_dict[key] = d1[key] + d2[key]
    return result_dict


def divide_nested_dict(d: Dict[str, Any], divisor: int) -> Dict[str, Any]:
    """Divide all leaf values of nested a dict by a specify value.

    Args:
        d (Dict[str,Any]): _description_
        divisor (int): _description_

    Returns:
        Dict[str,Any]: _description_
    """
    result_dict = d.copy()
    for key, value in result_dict.items():
        if isinstance(value, dict):
            result_dict[key] = divide_nested_dict(value, divisor)
        else:
            # integer = isinstance(value, int)
            # result_dict[key] = int(value / divisor) if integer else value / divisor
            result_dict[key] = value / divisor
    return result_dict


def overtime_dict_to_list(d: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
    """ """
    new_d = {}
    for label in list(d.values())[0].keys():
        new_d[label] = [v[label] for v in d.values()]
    return new_d


def cumulate_dict_key(d):
    cumulative_sum = 0
    cumulative_dict = {}
    for key, value in d.items():
        cumulative_sum += value
        cumulative_dict[key] = cumulative_sum
    return cumulative_dict


#########################
# Multiseed
#########################
def get_mean_stats_multiseed(path: Path, load_stat: Callable[[Path], Any]) -> Any:
    """Return the mean stat object from a multiseed run using the furnished load_stat function,
    Args:
        path (Path): path of a multirun over different seeds.
        load_stat (Callable[[Path],Any]): Function used to load the stats in each run from the multirun

    Returns:
        Any: mean of the stats that are returned from the load_stat function.
    """
    dirs = get_dirs_from_multirun(path)
    nb_seeds = len(dirs)
    mean_stats = load_stat(dirs.pop(0))
    for d in dirs:
        g_s = load_stat(d)
        mean_stats = merge_dicts(mean_stats, g_s)
    return divide_nested_dict(mean_stats, nb_seeds)


def get_mean_fuzzy_stats_multiseed(path: Path, load_stat: Callable[[Path], Any]) -> Any:
    """Return the mean stat object from a multiseed run using the furnished load_stat function,
    Args:
        path (Path): path of a multirun over different seeds.
        load_stat (Callable[[Path],Any]): Function used to load the stats in each run from the multirun

    Returns:
        Any: mean of the stats that are returned from the load_stat function.
    """
    usecases: List[str] = ["embb", "urllc", "mmtc"]
    dirs = get_dirs_from_multirun(path)
    nb_seeds = len(dirs)
    l1_keys: List[str] = list(load_stat(dirs[0]).keys())
    # Averaging a list of labels make no sense.
    l1_keys.remove("an_peer")
    mean_stats = {}
    for stat_category in l1_keys:
        mean_stats[stat_category] = {}

        if stat_category == "an_max_capacity":
            mean_stats[stat_category] = {
                an: 0 for an in list(load_stat(dirs[0])[stat_category].keys())
            }
            for dir in dirs:
                capacities: List[int] = sorted(
                    list(load_stat(dir)[stat_category].values()), reverse=True
                )
                for an in mean_stats[stat_category]:
                    mean_stats[stat_category][an] += capacities.pop(0)
            mean_stats[stat_category] = divide_nested_dict(
                mean_stats[stat_category], nb_seeds
            )

        elif stat_category == "an_max_capacity_per_type":
            mean_stats[stat_category] = {
                an: {k: 0 for k in usecases}
                for an in list(load_stat(dirs[0])[stat_category].keys())
            }
            for uc in usecases:
                for dir in dirs:
                    capacities: List[int] = sorted(
                        [
                            stat[uc]
                            for stat in load_stat(dir)[
                                "an_max_capacity_per_type"
                            ].values()
                        ],
                        reverse=True,
                    )
                    for an in mean_stats[stat_category]:
                        mean_stats[stat_category][an][uc] += capacities.pop(0)
                for an in mean_stats[stat_category]:
                    mean_stats[stat_category][an][uc] = (
                        mean_stats[stat_category][an][uc] / nb_seeds
                    )
        else:
            mean_stats[stat_category] = {
                an: 0 for an in list(load_stat(dirs[0])[stat_category].keys())
            }
            for d in dirs:
                capacities = sorted(
                    list(load_stat(d)[stat_category].values()), reverse=True
                )
                for an in mean_stats[stat_category]:
                    mean_stats[stat_category][an] += capacities.pop(0)
            mean_stats[stat_category] = divide_nested_dict(
                mean_stats[stat_category], nb_seeds
            )

    return mean_stats


############################
# Misc
############################


def find_gaps(
    intervals: List[Tuple[float, float]], min: float, max: float
) -> Tuple[float, float]:
    """Find gaps in an ordered list of Tuples that contains two strictly ordered values e.g. [(0.2,0.4),(0.6,0.8)]

    Args:
        intervals (List[Tuple[float,float]]): e.g. [(0.2,0.4),(0.6,0.8)]
        min (float): min possible value (e.g. 0.0)
        max (float): max possible value (e.g. 1.0)

    Returns:
        Tuple[float,float]: e.g. [(0.0,0.2),(0.4,0.6),(0.8,1.0)]
    """
    gaps = []
    previous_end = min

    for start, end in intervals:
        if start > previous_end:
            gaps.append((previous_end, start))
        previous_end = end

    if previous_end < max:
        gaps.append((previous_end, max))

    return gaps


############################
# Specific value requirement
############################


def _load_global_stats(path: Path) -> Any:
    check_paths(path)
    with open(path / "global_stats.json") as f:
        return json.load(f)


def load_global_stats(path: Path, multiseed: bool) -> Any:
    if not multiseed:
        return _load_global_stats(path)
    else:
        return get_mean_stats_multiseed(path, _load_global_stats)


def load_total_interactions(path: Path, multiseed: bool = False) -> float:
    """Return the total number of interactions in a run or multirun

    Args:
        path (Path): run directory
        multiseed (bool, optional): Wether its a single run or multiseed multiun. Defaults to False.

    Returns:
        float: interactions
    """
    return load_global_stats(path, multiseed)["nb_interactions"]


# Rajouter une possibilité de sélectionner si c'est la transaction
# réellement effectuée ou bien le résultat attendu;
def load_total_positive(
    path: Path, multiseed: bool = False, expected: bool = False
) -> float:
    """Return the total number of positive interactions in a run or multirun

    Args:
        path (Path): run directory
        multiseed (bool, optional): Wether its a single run or multiseed multiun. Defaults to False.
        expected (bool, optional): Considering the actual or expected results in the stats. Defaults to False.
    Returns:
        float: _description_
    """
    entry = "labels_control_stats" if expected else "labels_stats"
    gs: Dict[str, Dict[str, float]] = load_global_stats(path, multiseed)[entry]
    s = 0
    positive: List[str]
    if is_fuzzy_run(path):
        positive = ["satisfied", "very satisfied"]
    else:
        positive = ["positive"]
    for d in gs.values():
        s += sum([v for k, v in d.items() if k in positive])
    return s


def load_total_negative(
    path: Path, multiseed: bool = False, expected: bool = False
) -> float:
    """Return the total number of negative interactions in a run or multirun

    Args:
        path (Path): run directory
        multiseed (bool, optional): Wether its a single run or multiseed multiun. Defaults to False.
        expected (bool, optional): Considering the actual or expected results in the stats. Defaults to False.

    Returns:
        float: _description_
    """
    entry = "labels_control_stats" if expected else "labels_stats"
    gs: Dict[str, Dict[str, float]] = load_global_stats(path, multiseed)[entry]
    s = 0
    negative: List[str]
    if is_fuzzy_run(path):
        negative = ["neutral", "unsatisfied", "very unsatisfied"]
    else:
        negative = ["negative"]
    for d in gs.values():
        s += sum([v for k, v in d.items() if k in negative])
    return s


def load_negative_overtime(path: Path, multiseed: bool = False) -> Dict[str, int]:
    """Return the number of negative interactions overtime in a run or multirun

    Args:
        path (Path): run directory
        multiseed (bool, optional): Wether its a single run or multiseed multiun. Defaults to False.

    Returns:
        Dict[str, int]: number of negative interactive for each time interval
    """
    ls_overtime: Dict[str, Dict[str, float]] = load_stats_overtime(path, multiseed)[
        "labels_stats"
    ]
    negative_overtime: Dict[str, int] = {}  # {q1:nb_interactions, ...}
    s = 0
    for k, v in ls_overtime.items():
        negative_overtime[k] = sum([i["negative"] for i in v.values()])
    return negative_overtime


def fuzzy_load_negative_overtime(path: Path, multiseed: bool = False) -> Dict[str, int]:
    """Return the number of negative interactions overtime in a run or multirun

    Args:
        path (Path): run directory
        multiseed (bool, optional): Wether its a single run or multiseed multiun. Defaults to False.

    Returns:
        Dict[str, int]: number of negative interactive for each time interval
    """
    ls_overtime: Dict[str, Dict[str, float]] = load_stats_overtime(path, multiseed)[
        "labels_stats"
    ]
    negative_overtime: Dict[str, int] = {}  # {q1:nb_interactions, ...}
    s = 0
    for k, v in ls_overtime.items():
        negative_overtime[k] = sum(
            [
                i["very unsatisfied"] + i["unsatisfied"] + i["neutral"]
                for i in v.values()
            ]
        )
    return negative_overtime


def _load_stats_overtime(path: Path) -> Any:
    check_paths(path)
    with open(path / "stats_overtime.json") as f:
        return json.load(f)


def load_stats_overtime(path: Path, multiseed: bool) -> Any:
    """Return the contents of stats_overtime.json from a single run or the mean values from a multirun with multiple seeds.

    Args:
        path (Path): path of the run/multirun.
        multiseed (bool):  True if path is a multirun used with multiple seed.

    Returns:
        Any:
    """
    if not multiseed:
        return _load_stats_overtime(path)
    else:
        return get_mean_stats_multiseed(path, _load_stats_overtime)


def get_labels(path: Path, multiseed: bool) -> List[str]:
    """Return labels that were used in a run

    Args:
        path (Path): Run path
        multiseed (bool): True if path is a multirun with multiple seed.


    Returns:
        List[str]: label used in the run
    """
    return list(load_labels_stats(path, multiseed).keys())


def get_primary_labels(path: Path, multiseed: bool) -> List[str]:
    """Return primary labels used in a run, e.g. urllc for urllc_good or urllc_outage

    Args:
        path (Path): Run path
        multiseed (bool): True if path is a multirun with multiple seed.


    Returns:
        List[str]: label used in the run
    """
    p_labels = set([])
    for label in list(load_labels_stats(path, multiseed).keys()):
        p_labels.add(label.split("_")[0])

    return list(p_labels)


def load_labels_stats(path: Path, multiseed: bool) -> Dict[str, Dict[str, int]]:
    """return stats per labels.
        multiseed (bool): True if path is a multirun with multiple seed.



    Structure is :
            "label_1": {
                "positive": 2994,
                "negative": 752,
                "total": 3746
            },
            "label_2": {
                "positive": 105,
                "negative": 149,
                "total": 254
            }
    """
    global_stats = load_global_stats(path, multiseed)
    return global_stats["labels_stats"]


def load_labels_stats_overtime(
    path: Path, multiseed: bool
) -> Dict[str, Dict[str, int]]:
    """return stats per labels overtime.
        multiseed (bool): True if path is a multirun with multiple seed.



    Structure is :
            "label_1": {
                "positive": 2994,
                "negative": 752,
                "total": 3746
            },
            "label_2": {
                "positive": 105,
                "negative": 149,
                "total": 254
            }
    """
    stats_overtime = load_stats_overtime(path, multiseed)
    return stats_overtime["labels_stats"]


def load_nb_participants_labels(path: Path, multiseed: bool) -> Dict[str, int]:
    """return number of participants per label
    structure is :
    {
        "label_1": 15,
        "label_2": 5
    }
    """
    global_stats = load_global_stats(path, multiseed)
    return global_stats["nb_participants_labels"]


############################
# Interactions loaders
############################


def load_nb_interactions_per_peer(path: Path) -> List[int]:
    """Load the total number of interactions from each peer on a certain label

    Args:
        path (Path): path of the run containing stats_overtime.json.
        label (str): label to compute interactions number over.
        multiseed (bool, optional): True if there are multiple seed.

    Returns:
        Dict[str, List[int]]: List containing each participants number of interactions on each time period.
    """
    stats = load_global_stats(path, multiseed=False)
    labels = get_labels(path, multiseed=False)
    ret: Dict[str, List[int]] = {}
    for label in labels:
        ret[label] = [v["total"] for k, v in stats["peer_stats"].items() if label in k]
    return ret


def load_label_interactions_number_overtime(
    path: Path, multiseed: bool
) -> Dict[str, Dict[str, List[int]]]:
    """For every label on each time period return each participants number of interactions

    Args:
        path (Path): path of the run containing stats_overtime.json
        multiseed (bool, optional): True if there are multiple seed.

    Returns:
        Dict[str, Dict[str, List[int]]]: List containing each participants number of interactions on each time period.
    """

    stats = load_stats_overtime(path, multiseed)
    labels = get_labels(path, multiseed)
    res = {}
    for q, participants in stats["peer_stats"].items():
        res[q] = {l: [] for l in labels}
        for p in participants:
            for l in labels:
                if l in p:
                    res[q][l].append(participants[p]["total"])
    return res


def _load_nb_interactions_overtime(path: path) -> Dict[str, List[int]]:
    """For each time period return the number of interactions

    Args:
        path (Path): path of the run containing stats_overtime.json


    Returns:
        Dict[str, List[int]]: Number of interactions on each time period.
    """
    with open(path / "stats_overtime.json") as f:
        s = json.load(f)
        res = {}
        for q, labels in s["labels_stats"].items():
            res[q] = sum([s["labels_stats"][q][l]["total"] for l in labels.keys()])
        return res


def load_nb_interactions_overtime(path: Path, multiseed: bool) -> Dict[str, List[int]]:
    """For each time period return the number of interactions

    Args:
        path (Path): path of the run containing stats_overtime.json
        multiseed (bool): True if path is a multirun with multiple seed.


    Returns:
        Dict[str, List[int]]: Number of interactions on each time period.
    """
    if multiseed:
        dirs: List[Path] = get_dirs_from_multirun(path)
        nb_seed: int = len(dirs)
        total_interactions: Dict[str:int] = {}
        for d in dirs:
            nb_interaction = _load_nb_interactions_overtime(d)
            for k in nb_interaction.keys():
                if k in total_interactions:
                    total_interactions[k] += nb_interaction[k]
                else:
                    total_interactions[k] = nb_interaction[k]
        return {k: int(v / nb_seed) for k, v in total_interactions.items()}
    else:
        return _load_nb_interactions_overtime(path)


def load_participants_interactions_overtime(
    path: Path, multiseed: bool, outcome: str = "total", fuzzy=False
) -> Dict[str, List[int]]:
    """For each particicpants return a list containing the number of interactions at each period.
    Args:
        path (Path): path of the run containing stats_overtime.json
        multiseed (bool): True if path is a multirun with multiple seed.
        fuzzy (bool): True if this is a fuzzy run. Default to False.
        outcome (str): ["positive", "negative", "total"] What transactions should be returned. Default to "total".

    Returns:
        Dict[str, List[int]]: Number of interactions per participant.
    """
    # if fuzzy :
    #     negative_outcomes = ["very unsatisfied", "unsatisfied", "neutral"]
    #     positive_outcomes = ["satisfied", "very satisfied"]

    if outcome not in ["positive", "negative", "total"]:
        Raise(TypeError("Outcome must be :\n positive\n negative\n total"))
    interactions: Dict[str, List[int]] = {}

    p_stats = load_stats_overtime(path, multiseed)["peer_stats"]
    for p in next(iter(p_stats.values())):
        interactions[p] = []
    for q in p_stats.values():
        for p in q:
            interactions[p].append(q[p][outcome])
    return interactions


def load_participants_interactions(
    path: Path, multiseed: bool, outcome: str = "total"
) -> Dict[str, List[int]]:
    """Return the glboal number of interactions per participant
    Args:
        path (Path): path of the run containing stats_overtime.json
        multiseed (bool): True if path is a multirun with multiple seed.
        outcome (str): ["positive", "negative", "total"] What transactions should be returned. Default to "total".

    Returns:
        Dict[str, int]: Number of interactions per participant.
    """
    if outcome not in ["positive", "negative", "total"]:
        Raise(TypeError("Outcome must be :\n positive\n negative\n total"))
    p_stats = load_global_stats(path, multiseed)["peer_stats"]
    return {k: v[outcome] for k, v in p_stats.items()}


def load_mean_interactions_overtime(path: Path, multiseed: bool) -> List[int]:
    """For each time period return the mean number of interactions per participants.

    Args:
        path (Path): path of the run containing stats_overtime.json
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.

    Returns:
        List[int]: Mean number of interactions per participants on each time period.
    """
    nb_participants = sum(load_nb_participants_labels(path, multiseed).values())
    interactions_overtime: List[int] = list(
        load_nb_interactions_overtime(path, multiseed).values()
    )
    return [
        int(floor(interactions / nb_participants))
        for interactions in interactions_overtime
    ]


def load_mean_labels_interactions_overtime(path: Path, multiseed: bool) -> List[int]:
    """For each label and time period return the mean number of interactions per participants.

    Args:
        path (Path): path of the run containing stats_overtime.json
        multiseed (bool, optional): True if there are multiple seed. Defaults to False.

    Returns:
        List[int]: Mean number of interactions per participants on each time period.
    """
    l_stats_overtime = load_labels_stats_overtime(path, multiseed=multiseed)
    l_nb_participants = load_nb_participants_labels(path, multiseed)
    l_s_overtime = {}
    for q, lstats in l_stats_overtime.items():
        # l_s_overtime[q] = {
        #     label: divide_nested_dict(stat, l_nb_participants[label])
        #     for label, stat in lstats.items()
        # }
        l_s_overtime[q] = {
            label: sum(list(stat.values())) / l_nb_participants[label]
            for label, stat in lstats.items()
        }
    return l_s_overtime


def load_mean_number_of_transaction_category(path: Path):
    """Return mean the mean number of transaction per time unit on each category of participants.

    Args:
        path (Path): path of the run from where results should be extracted.
    Return:
           "label_1": {
                "1": {
                    "positive": 2994,
                    "negative": 752,
                    "total": 3746
                },
                "2": {
                    "positive": 105,
                    "negative": 149,
                    "total": 254
                },
                ...
            },
            "label_2": {
                "1": {
                        "positive": 2994,
                        "negative": 752,
                        "total": 3746
                    }, ...
            }
    """
    pass


def load_reput_overtime(
    path: Path, emited: bool = False, multiseed: bool = False
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return the reputation of participants over different time step.

    Args:
        path (Path): run path
        emited (bool, optional): Wether the loaded reput should be the received or emited reput. Defaults to False.
        multiseed (bool, optional): Wether the provided path correspond to a multiseed run or not. Defaults to False.

    Returns:
         Dict[str, Dict[str, Dict[str, float]]]:
            {
                "q1" : {
                        "good_001" : {"good_002":0.2, "good_003":0.8, ...},
                        "good_002" : {"good_001":0.1, "good_003":0.5, ...}
                    },
                "q2" : ...
            }

    """
    overtime = load_stats_overtime(path, multiseed)
    reput_overtime = overtime["peer_reputation"]
    if emited:
        return reput_overtime
    else:
        received_reput: Dict[str, Dict[str, Dict[str, float]]] = {}
        for q in reput_overtime:
            received_reput[q] = {}
            for receiver in reput_overtime[q]:
                received_reput[q][receiver] = {
                    emiter: reput_overtime[q][emiter][receiver]
                    for emiter in reput_overtime[q]
                    if receiver in reput_overtime[q][emiter]
                }
        return received_reput


def load_reput_on_label_overtime(
    path: Path, label: str, emited: bool = False, multiseed: bool = False
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return the reputation of participants over different time step.

    Args:
        path (Path): run path.
        label (str): Label that should be in the peer name.
        emited (bool, optional): Wether the loaded reput should be the received or emited reput. Defaults to False.
        multiseed (bool, optional): Wether the provided path correspond to a multiseed run or not. Defaults to False.

    Returns:
         Dict[str, Dict[str, Dict[str, float]]]:
            {
                "q1" : {
                        "good_001" : {"good_002":0.2, "good_003":0.8, ...},
                        "good_002" : {"good_001":0.1, "good_003":0.5, ...}
                    },
                "q2" : ...
            }

    """
    reput = load_reput_overtime(path, emited, multiseed)
    reput_label: Dict[str, Dict[str, Dict[str, float]]] = {}
    for q in reput:
        reput_label[q] = {k: v for k, v in reput[q].items() if label in k}
    return reput_label


def load_mean_reput_on_label_overtime(
    path: Path,
    label: str,
    emited: bool = False,
    mean_labels: bool = False,
    multiseed: bool = False,
) -> Dict[str, List[float]]:
    """Return the reputation of participants over different time step.

    Args:
        path (Path): run path.
        label (str): Label that should be in the peer name.
        emited (bool, optional): Wether the loaded reput should be the received or emited reput. Defaults to False.
        mean_labels (bool, optional): When true participants sharing the same label are averaged together. Defaults to False.
        multiseed (bool, optional): Wether the provided path correspond to a multiseed run or not. Defaults to False.

    Returns:
         Dict[str, List[float]]:
            {
                "good_001" : [0.5, 0.6, .., 0.4]
                "good_002" : [0.7, 0.6., ..,0.8]
            }

    """
    fuzzy = is_fuzzy_run(path)
    reput = load_reput_on_label_overtime(path, label, emited, multiseed)
    mean_reput: Dict[str, List[float]] = {}
    for p in reput[next(iter(reput.keys()))]:
        mean_reput[p] = []
    for q in reput:
        for k, v in reput[q].items():
            if fuzzy:
                # When fuzzy reput is used, reputation is subjectvie so it make more sense to
                # judge reput based on participants with the same requirements.
                mean_reput[k].append(
                    np.mean([vv for kk, vv in v.items() if k.split("_")[0] in kk])
                )
            else:
                mean_reput[k].append(np.mean(list(v.values())))

    if mean_labels:
        return {
            label: [
                statistics.mean([p[i] for p in mean_reput.values()])
                for i in range(len(next(iter(mean_reput.values()))))
            ]
        }
    else:
        return mean_reput


def load_interaction_variance_over_label(
    path: Path, label: str, multiseed: bool
) -> float:
    """Return the variance of the number of interactions for a certain abel.

    Args:
        path (Path): run path
        label (str): label to compute the variance over.
        multiseed (bool): True if the Path is a multiseed path

    Returns:
        float: float that represent the variance for a specific value. If multiseed is true the mean variance of the run is used.
    """
    if multiseed:
        vars: List[float] = []
        for dir in get_dirs_from_multirun(path):
            vars.append(np.var(load_nb_interactions_per_peer(dir)[label]))
        return np.mean(vars)
    else:
        a = load_nb_interactions_per_peer(path)[label]
        return np.var(load_nb_interactions_per_peer(path)[label])


def get_dirs_from_multirun(dir: Path) -> List[Path]:
    """Extracts path from a multi-run

    Args:
        in_dir (Path):multi-run path

    Returns:
        List[str]: list of hydra subfolder in the specified folder
    """
    dirs: List[Path] = [
        p for p in dir.iterdir() if p.is_dir() and Path(p / ".hydra").exists()
    ]
    return dirs


###########################
# Topology utils
###########################


def _load_topology_stats(path: Path) -> Any:
    check_paths(path)
    with open(path / "topology_stats.json") as f:
        return json.load(f)


def load_topology_stats(path: Path, multiseed: bool = False) -> Any:
    if not multiseed:
        return _load_topology_stats(path)
    else:
        return get_mean_fuzzy_stats_multiseed(path, _load_topology_stats)
