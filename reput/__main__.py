from os import environ
from pathlib import Path

import hydra
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from typing import Callable, List
import numpy as np
from numpy.random import default_rng

from reput.topology import AccessNetwork, CoreNetwork


from .environment import Environment
from .stats import print_all_stats
from .topology_stats import log_ans_capacity
from .peer import Peer
from .transactions import SimpleTransaction, FuzzyTransaction


# Custom resolver for transactions
def transaction_selector(fuzzy, *, _parent_):
    if fuzzy:
        return FuzzyTransaction
    else:
        return SimpleTransaction


OmegaConf.register_new_resolver("transaction_selector", transaction_selector)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    np.random.seed = cfg.simulation.seed
    rng = default_rng(cfg.simulation.seed)
    decay_fn = hydra.utils.call(cfg.reputation.decay)
    result_path = Path("./results/")

    # Plug hydra instantiated topologies here.
    i: int = 1
    access_networks: List[AccessNetwork] = []
    for access in cfg.topology.access_topo:
        for _ in range(access.nb_zones):
            access_networks.append(
                hydra.utils.instantiate(access.build_func, id=f"an_{i:02}")
            )
            i += 1

    i: int = 1
    core_networks: List[CoreNetwork] = []
    for core in cfg.topology.core_topo:
        for _ in range(core.nb_zones):
            core_networks.append(
                hydra.utils.instantiate(core.build_func, id=f"cn_{i:02}")
            )
            i += 1

    environment: Environment = instantiate(
        cfg.simulation.environment,
        core_networks=core_networks,
        access_networks=access_networks,
    )

    # Replace this with a custom resolver
    # https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html
    peers: List[Peer] = []
    for peer_type in cfg.simulation.distribution:
        i: int = 0
        label: str = peer_type.build_func.label
        # /!\ Partial not callable
        peer_fn: Callable = peer_type.build_func
        nb_peers: int = peer_type.nb_peers
        for _ in range(nb_peers):
            peers.append(
                hydra.utils.instantiate(
                    peer_fn, id=f"{label}_{i+1:03}", decay=decay_fn, rng=rng
                )
            )
            i += 1
    environment.add_peer_list(peers)

    # TODO_topology Update neighborhood depending on the core network topology
    environment.add_all_peers_as_neighbors()

    # Log topology full capacity.
    log_ans_capacity(environment)

    environment.run_simulation()
    print_all_stats(environment, result_path)


if __name__ == "__main__":
    run()
