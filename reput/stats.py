from typing import List, Dict, Tuple, Any
from pathlib import Path
from copy import deepcopy
import json

from .peer import Peer
from .environment import Environment
from .topology_stats import log_ans_dropped_transactions
from .fuzzy import satisfaction_terms
from .common_utils import sum_dicts


def get_global_stats(env: Environment) -> Dict[Peer, Dict[str, int]]:
    """Return the global stats of an experiment.

    Args:
        env (Environment): environment of an experiment that has ended.

    Returns:
        Dict[Peer, Dict[str, int]]:
    """

    global_stats: Dict[Peer, Any] = {}
    if not env.fuzzy:
        for p1 in env.peers:
            global_stats[p1] = {"positive": 0, "negative": 0, "total": 0}
            for p2 in p1.neighbors:
                positive, negative, total = p2.neighbors[p1].count_transactions()
                global_stats[p1]["positive"] += positive
                global_stats[p1]["negative"] += negative
                global_stats[p1]["total"] += total
    else:
        for p1 in env.peers:
            global_stats[p1] = {term: 0 for term in satisfaction_terms}
            for p2 in p1.neighbors:
                global_stats[p1] = sum_dicts(
                    [global_stats[p1], p2.neighbors[p1].count_transactions()]
                )
    return global_stats


def get_global_control_stats(env: Environment) -> Dict[Peer, Dict[str, float]]:
    """Return the control stats of an experiment including the reput at transaction initiation and failure occurences

    Args:
        env (Environment): environment of an experiment that has ended.

    Returns:
        Dict[Peer, Dict[str, int]]:
    """

    control_stats: Dict[Peer, Any] = {}
    if not env.fuzzy:
        return control_stats
    else:
        for p1 in env.peers:
            control_stats[p1] = {}
            control_stats[p1] = {term: 0 for term in satisfaction_terms}
            for p2 in p1.neighbors:
                control_stats[p1] = sum_dicts(
                    [
                        control_stats[p1],
                        p2.neighbors[p1].count_transactions(expected=True),
                    ]
                )
    return control_stats


def get_global_stats_overtime(
    env: Environment,
    nb_steps: float = 10,
    step_length: float = 0.0,
) -> Dict[str, Dict[Peer, Dict[str, int]]]:
    """
    Produce execution statistics over time
    Args:
        env (Environment): env on which stats should be extracted.
        nb_steps (float, optional): nb of steps to produce. Defaults to 10.
        step_length (float, optional): Lentgth of a step. If specified take precedence over nb_steps. Defaults to 0.0.

    Return:
        Dict[str, Dict[Peer, Dict[str, int]]]
            {
                "q1" :
                        "peer1" :
                            {
                            "positive" : 115,
                            "negative" : 40,
                            "155" :
                            }
                        ...,
                "q2" :
            }
    """
    global_stats_over_time: Dict[str, Dict[Peer, Any]] = {}
    if step_length:
        nb_steps = env.sim_length / step_length
    else:
        step_length = env.sim_length / nb_steps
    if env.fuzzy:
        # TODO_clarity : further merge fuzzy and not fuzzy
        for i in range(nb_steps):
            global_stats_over_time[f"q{i+1}"] = {}
            for p1 in env.peers:
                # TODO_clarity until here.
                global_stats_over_time[f"q{i+1}"][p1] = {
                    term: 0 for term in satisfaction_terms
                }
                for p2 in p1.neighbors:
                    global_stats_over_time[f"q{i+1}"][p1] = sum_dicts(
                        [
                            global_stats_over_time[f"q{i+1}"][p1],
                            p2.neighbors[p1].count_transactions(
                                start=i * step_length, end=(i + 1) * step_length
                            ),
                        ]
                    )

    else:
        for i in range(nb_steps):
            global_stats_over_time[f"q{i+1}"] = {}
            for p1 in env.peers:
                global_stats_over_time[f"q{i+1}"][p1] = {
                    "positive": 0,
                    "negative": 0,
                    "total": 0,
                }
                for p2 in p1.neighbors:
                    positive, negative, total = p2.neighbors[p1].count_transactions(
                        start=i * step_length, end=(i + 1) * step_length
                    )
                    global_stats_over_time[f"q{i+1}"][p1]["positive"] += positive
                    global_stats_over_time[f"q{i+1}"][p1]["negative"] += negative
                    global_stats_over_time[f"q{i+1}"][p1]["total"] += total

    return global_stats_over_time


def get_labels_stats(
    global_stats: Dict[Peer, Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    """_summary_

    Args:
        global_stats (Dict[Peer, Dict[str, int]]): _description_

    Returns:
        Dict[str, Dict[str, int]]:
            "labels_stats": {
                "good": {
                    "positive": 3100,
                    "negative": 662,
                    "total": 3762
                },
                ...
            },
    """
    labels_stats: Dict[str, Dict[str, int]] = {}
    for p in global_stats:
        if p.label not in labels_stats:
            labels_stats[p.label] = {}
        labels_stats[p.label] = sum_dicts([labels_stats[p.label], global_stats[p]])
    return labels_stats


def count_interactions(global_stats: Dict[Peer, Dict[str, int]]):
    return sum([v["total"] for v in global_stats.values()])


def count_interactions_per_label(global_stats: Dict[Peer, Dict[str, int]]):
    labels_interactions_nb: Dict[str, int] = {}
    for p in global_stats:
        labels_interactions_nb[p.label] = (
            labels_interactions_nb.get(p.label, 0) + global_stats[p]["total"]
        )
    return labels_interactions_nb


def count_peer_per_label(global_stats: Dict[Peer, Dict[str, int]]) -> Dict[str, int]:
    labels_nb: Dict[str, int] = {}
    for p in global_stats:
        if p.label not in labels_nb:
            labels_nb[p.label] = 0
        labels_nb[p.label] += 1
    return labels_nb


def change_peer_to_id(stats: Dict[Peer, Any]):
    """Take a a dict that have peers as keys and replace the peer object with it's id. If the value is also a dict with peer as key do the same.

    Args:
        stats (Dict[Peer, Any]): Dict of stats with key as a Peer

    Returns:
        Dict[str, Any]: where str is the id of the peer.
    """
    return {q.id: v for q, v in stats.items()}


def get_labels_stats_overtime(
    global_stats_overtime: Dict[str, Dict[Peer, Dict[str, int]]]
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """stats divided by label over different time period.
    Args:
        global_stats_overtime (Dict[str, Dict[Peer, Dict[str, int]]]):

    Returns:
        Dict[str, Dict[str, Dict[str, int]]]:
            {   "q1":
                    {
                        "good": {
                            "positive": 2994,
                            "negative": 752,
                            "total": 3746
                        },
                        "bad": {
                            "positive": 2994,
                            "negative": 752,
                            "total": 3746
                        }
                    },
                "q2": ...
            }

    """
    return {k: get_labels_stats(v) for k, v in global_stats_overtime.items()}


def get_reputation_overtime(env: Environment):
    reputation_over_time: Dict[str, Dict[Peer, Dict[Peer, float]]] = {}
    for k1, v1 in env.reput_snapshots.items():
        reputation_over_time[k1] = {}
        for k2, v2 in v1.items():
            reputation_over_time[k1][k2.id] = {k3.id: v3 for k3, v3 in v2.items()}
    return reputation_over_time


def print_stats(
    global_stats: Dict[Peer, Dict[str, int]],
    control_stats: Dict[Peer, Dict[str, float]] = {},
    out_path: Path = "",
):
    """Format and print global stats to a file.
    Args:
        global_stats (Dict[Peer, Dict[str, int]]): stats that should be printed. .
        out_path (Path, optional): Path to print results, if left unspecified results are printed in the terminal. Defaults to "".
    """
    params: Dict[str, Any] = {
        "nb_participants": len(global_stats),
        "nb_participants_labels": count_peer_per_label(global_stats),
        "nb_interactions": count_interactions(global_stats),
        "nb_interactions_labels": count_interactions_per_label(global_stats),
        "labels_stats": get_labels_stats(global_stats),
        "peer_stats": change_peer_to_id(global_stats),
        "peer_control_stats": change_peer_to_id(control_stats),
        "labels_control_stats": get_labels_stats(control_stats),
    }
    json.dump(params, open("global_stats.json", "w"), indent=4)


def print_stats_overtime(env: Environment):
    stats_overtime = get_global_stats_overtime(env,nb_steps=env.sim_length)
    params: Dict[str, Any] = {
        "labels_stats": get_labels_stats_overtime(stats_overtime),
        "peer_stats": {k: change_peer_to_id(v) for k, v in stats_overtime.items()},
        "peer_reputation": get_reputation_overtime(env),
    }

    json.dump(
        params,
        open("stats_overtime.json", "w"),
        indent=5,
    )


def print_all_stats(env: Environment, out_path: str = ""):
    print_stats(get_global_stats(env), get_global_control_stats(env))
    print_stats_overtime(env)
    log_ans_dropped_transactions(env)
