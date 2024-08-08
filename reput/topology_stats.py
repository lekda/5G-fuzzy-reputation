from typing import List, Dict, Tuple, Any
from pathlib import Path
from copy import deepcopy
import json

from .topology import AccessNetwork
from .environment import Environment
from .peer import Peer


def get_access_network_max_capacity(env: Environment) -> Dict[str, float]:
    """Return the max network capacity of each AcccesNetwork

    Args:
        env (Environment): Simulation environment containing pointer to the different AccessNetwork

    Returns:
        Dict[str,float]: {
            "an1" : 6.66,
            "an2" : 9.99,
            ...
        }
    """
    capacity: Dict[str, float] = {}
    for an in env.access_networks:
        peers: List[Peer] = an.peers
        capacity[an.id] = sum([p.capacity[an] for p in peers])
    return capacity


def get_access_network_max_capacity_per_type(
    env: Environment,
) -> Dict[str, Dict[str, float]]:
    """Return the per type max network capacity of each AcccesNetwork

    Args:
        env (Environment): Simulation environment containing pointer to the different AccessNetwork

    Returns:
        Dict[str,float]: {
            "an1" : 6.66,
            "an2" : 9.99,
            ...
        }
    """
    types = ["embb", "urllc", "mmtc"]
    capacity: Dict[str, float] = {}
    for an in env.access_networks:
        capacity[an.id] = {}
        peers: List[Peer] = an.peers
        for t in types:
            capacity[an.id][t] = sum([p.capacity[an] for p in peers if t in p.id])
    return capacity


def get_an_peers(env: Environment) -> Dict[str, List[str]]:
    """Return the peer id for each an.

    Args:
        env (Environment): Simulation environment containing pointer to the different AccessNetwork

    Returns:
        Dict[str,float]: {
            "an1" : {embb_good1, urrlc_good2,...}
            "an2" : {embb_good2, urrlc_good2,...}
            ...
        }
    """
    an_peers: Dict[str, List[str]] = {}
    for an in env.access_networks:
        peers: List[Peer] = an.peers
        an_peers[an.id] = [p.id for p in peers]
    return an_peers


def get_capacity_dropped_transactions(env: Environment) -> Dict[str, int]:
    """Return the number of requested interactions dropped because there were not
    enough capacity for each AcccesNetwork

    Args:
        env (Environment): simulaiton environment containing pointer to the different AccessNetwork

    Returns:
        Dict[str,float]: {
            "an1" : 25,
            "an2" : 0,
            ...
        }
    """
    return {an.id: an.no_peer_for_transaction for an in env.access_networks}


def log_ans_capacity(env: Environment):
    #
    json.dump(
        {
            "an_max_capacity": get_access_network_max_capacity(env),
            "an_max_capacity_per_type": get_access_network_max_capacity_per_type(env),
            "an_peer": get_an_peers(env),
        },
        open("topology_stats.json", "w"),
        indent=3,
    )


def log_ans_dropped_transactions(env: Environment):

    ts = json.load(open("topology_stats.json"))
    ts["transactions_dropped_no_peer_in_an"] = {
        an.id: an.no_peer_for_transaction for an in env.access_networks
    }
    ts["transactions_dropped_no_peer_capacity_in_an"] = {
        an.id: an.no_capacity_for_transaction for an in env.access_networks
    }
    json.dump(
        ts,
        open("topology_stats.json", "w"),
        indent=3,
    )
