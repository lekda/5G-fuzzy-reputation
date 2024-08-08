from os import environ
from typing import List, Dict, Tuple, Any
from collections import OrderedDict
import copy
import bisect

import numpy as np

from .topology import AccessNetwork, CoreNetwork
from .peer import Peer


class Environment:
    def __init__(
        self,
        sim_length: float,
        newcommer_random_rate: OrderedDict[float, float],
        random_rate: OrderedDict[float, float],
        r_sample_nb: float,
        reput: bool,
        fuzzy: bool,
        core_networks: List[CoreNetwork],
        access_networks: List[AccessNetwork],
        seed: float,
    ):
        """_summary_

        Args:
            sim_length (float): _description_
            newcommer_random_rate (OrderedDict[float, float]): _description_
            random_rate (OrderedDict[float, float]): _description_
            r_sample_nb (float): Number of reputation samples done in the simulation.
            reput (bool): Wether a reputation system should be used.
            fuzzy (bool): Use fuzzy logic for trust ? Reput must be True.
            core_networks (List[CoreNetwork]) : Core networks used by the environement
            access_networks (List[CoreNetwork]) : Access networks used by the environement
        """
        # Randomness handle
        self.rng = np.random.default_rng(seed)

        self.newcommer_random_rate: OrderedDict[float, float] = newcommer_random_rate
        self.random_rate: OrderedDict[float, float] = random_rate
        self.peers: List[Peer] = []
        self.events: List[Tuple[float, Peer, AccessNetwork]] = []
        self.sim_length = sim_length
        self.r_sample_nb = r_sample_nb
        self.reput: bool = reput
        self.next_reput_log: float = sim_length / 10
        self.fuzzy = fuzzy

        # Topology handles
        self.core_networks: List[CoreNetwork] = core_networks
        self.access_networks: List[AccessNetwork] = access_networks

        self.cn_available_slots: int = sum([cn.peer_nb for cn in self.core_networks])
        self.an_available_slots: int = sum([an.peer_nb for an in self.access_networks])

        # reputation snapshot at each time interval.
        # format
        # {
        #     "q1" : {
        #             "Peer_1" : {"Peer_2":0.2, "Peer_3":0.8, ...},
        #             "Peer_2" : {"Peer_1":0.1, "Peer_3":0.5, ...}
        #         },
        #     "q2" : ...
        # }
        self.reput_snapshots: Dict[str, Dict[Peer, Dict[Peer, float]]] = {}
        self.q: int = 1

    def assert_random_rate(self, t: float, n_random_rate: bool = False) -> float:
        """Return random rate of the Peer at time t.

        Args:
            t (float): time.
            n_random_rate (bool): Assert random rate or newcommer random rate. Default to False.

        Returns:
            float: random rate at time t.
        """
        if n_random_rate:
            rate: OrderedDict[float, float] = self.newcommer_random_rate
        else:
            rate: OrderedDict[float, float] = self.random_rate
        if t < 0:
            raise ValueError("Time t must be >=0")
        buffer: float = 0.0
        for t_p in rate:
            if t < t_p:
                break
            else:
                buffer = t_p
        return rate[buffer]

    def end_ongoing_transactions(self):
        """Called at the end of the simulation to end all ongoing transactions"""
        for p in self.peers:
            for an, transactions in p.ongoing_transactions.items():
                for t in transactions:
                    p.end_transaction(t)
                p.ongoing_transactions[an] = []

    def select_neighbors(self, neighbor_nb):
        """
        Randomly select new neighbors.

        #Parameters:
        #    neighbor_nb (int): number of neighbors to select.
        #Returns:
        #    sample(list[Peer]): list of peers.
        """
        return self.rng.choice(self.peers, size=neighbor_nb, replace=False)
    def count_peer_labels(self)->int:
        return len({p.label for p in self.peers})
        
    def add_peer(self, p: Peer,nb_peers:int, nb_labels:int):

        if self.cn_available_slots > 0:
            # Each peer is part of a single core network .
            while 1:
                cn = self.rng.choice(self.core_networks)
                if cn.capacity != 0:
                    cn.peers.append(p)
                    p.cn = cn
                    break
            self.cn_available_slots -= 1
        else:
            raise (IndexError("Not enough space in Core Networks for this Peer"))

        #  Each peer is placed in 5 differents access networks.
        if self.an_available_slots > 5: 
            self.an_available_slots -= 5
            # Seeded placement, make sure that each label is represented in all access networks.
            i = int(nb_labels*len(self.access_networks)/nb_peers)  
            # Leftover unseeded placement
            j = 5-i 
            assert j >= 0 
            
            # First batch of seeded placement 
            r_access_networks:List[AccessNetwork] = self.rng.choice(self.access_networks, replace=False,size=len(self.access_networks))
            for an in r_access_networks:
                if i <= 0:
                    break
                if an.capacity != 0 and not an.check_labels(p.label):
                    an.peers.append(p)
                    p.add_an(an)
                    i -= 1    
                 
            # Second batch of unseed placement
            r_access_networks:List[AccessNetwork] = self.rng.choice(self.access_networks, replace=False,size=len(self.access_networks))
            for an in r_access_networks:
                if j <= 0:
                    break
                if an.capacity != 0 and p not in an.peers:
                    an.peers.append(p)
                    p.add_an(an)
                    j -= 1
        else:
            raise (IndexError("Not enough space in Access Networks for this Peer"))

        self.peers.append(p)
        interactions: List[float] = [
            self.rng.uniform(0, self.sim_length) for _ in range(p.nb_interactions)
        ]

        for i in interactions:
            # Interaction parmi l'ensemble des access network.
            bisect.insort(self.events, (i, p, self.rng.choice(self.access_networks)))

        # Make sure that all zone have atleast one peer.

    def add_peer_list(self, peers: List[Peer]):
        p_len = len(peers)
        for p in peers:
            self.add_peer(p,p_len,self.count_peer_labels())

    def log_reput(self):
        self.reput_snapshots["q" + str(self.q)] = {}
        for p in self.peers:
            self.reput_snapshots["q" + str(self.q)][p] = p.neighbors_reputation()
        self.q += 1

    def next_event(self) -> bool:
        if self.events:
            timestamp, peer, an = self.events.pop(0)
            if timestamp > self.next_reput_log:
                self.next_reput_log += self.sim_length / self.r_sample_nb
                self.log_reput()
        else:
            # End of the simulation
            self.end_ongoing_transactions()
            self.log_reput()
            return False
        p: Peer = None
        r: float = -1.0
        if self.reput:
            p, r = peer.choose_peer_for_transaction(
                random_factor=self.assert_random_rate(timestamp),
                newcommer_random_factor=self.assert_random_rate(
                    timestamp, n_random_rate=True
                ),
                timestamp=timestamp,
                an=an,
            )
        else:
            an_candidates:List[Peer] = self.rng.choice(an.peers,replace=False,size=len(an.peers)) 
            for p_candidate  in an_candidates :
                if p_candidate.availability_check(timestamp, an):
                    p = p_candidate 
                    break
            if not p and len(an.peers) == 0:
                an.no_peer_for_transaction += 1
            elif not p:
                an.no_capacity_for_transaction += 1
        if p:
            p.initiate_transaction(peer, timestamp, an, r)
        return True

    def add_all_peers_as_neighbors(self):
        """Add all peers as neighbors."""
        for p1 in self.peers:
            for p2 in self.peers:
                # if p1 != p2:
                p1.add_neighbors([p2])

    def run_simulation(self):
        while 1:
            if not self.next_event():
                break
