""" Different peers that can be on the system. 
"""

from multiprocessing.managers import ValueProxy

from typing import Dict, List, Type, Callable, Any, Tuple
from collections import OrderedDict
from matplotlib.streamplot import OutOfBounds

from skfuzzy import control as ctrl

from .decay import exponential_decay, double_window, simple_window, adaptive_window
from .transactions import (
    DecayedTransaction,
    FuzzyTransaction,
    SimpleTransaction,
    Transaction,
    OngoingTransaction,
)
from .common_utils import sum_dicts, diff_dicts
from .record import Records, FuzzyRecords
from .topology import AccessNetwork, CoreNetwork
from .fuzzy import (
    transaction_satisfaction,
    fuzzy_diff_normalization,
    worst_fuzzy_outcome,
    initial_crisp_values,
    rulesets,
    required_inputs,
    dispersion_map,
)
from numpy.random import Generator
from numpy import sqrt
from scipy.stats import gmean


class Peer:
    def __init__(
        self,
        behavior: OrderedDict[float, Any],
        total_capacity: float,
        id: str,
        breaking_rate: float,
        peer_interactions: int,
        transaction_type: Transaction,
        transaction_duration: float,
        initial_reput: float | Dict[str, float],
        forgiveness_delay: float,
        label: str,
        rng: Generator,
        decay: Callable = exponential_decay,
        dispersion: float = "",
        usecase: str = "",
    ) -> None:
        """init a peer
        Args:
            behavior (OrderedDict[float,Any]): define the success_rate of the peer over time must atleast contain 0.0 value.
            total_capacity (float) : Capacity of the peer in terms of simultaneously accepted transactions.
            id(str): Identifier of the node, should include the label and be unique.
            breaking_rate (float) : Rate at which a peer will fail.
            peer_interaction (int) : Define the number of interactions that the peer will initiate over the course of the simulation.
            transaction_type (Transaction) : Transaction that should be used (e.g. SimpleTransaction or FuzzyTransaction)
            transaction_duration (float) : duration of the transaction for the current peer.
            initial_reput (float|Dict[str,float]) : How much time before transactions that occured are forgiven?
            forgiveness_delay (float) : How much time before transactions that occured are forgiven?
            label (str): Label that define the category of the peer.
            rng (Generator): RNG generator used for the simulation
            decay (Callable): Partial function used for decay.
            dispersion (str) : Only used for fuzzy trust, "high"|"medium"|"low" determine dispersion of the crisp values for an outcome.
            usecase (str) : Only used for fuzzy trust, "eMBB"|"URLLC"|"mMTC" choose the fuzzy ruleset used for reputation.

        """
        # Random generator
        self.rng = rng

        # Handle Fuzzy logic implementation
        self.transaction_type: Transaction = transaction_type
        self.fuzzy = self.transaction_type == FuzzyTransaction
        if self.fuzzy:
            assert dispersion in list(dispersion_map.keys())
            assert usecase in list(rulesets.keys())
            self.fuzzy_cs: ctrl.ControlSystemSimulation = ctrl.ControlSystemSimulation(
                rulesets[usecase]
            )
            self.fuzzy_required_input: List[str] = required_inputs[usecase]

        # function used to compute the reputation or trust on other peers.
        self.behavior: (
            OrderedDict[float, float] | OrderedDict[float, Dict[str, float]]
        ) = OrderedDict(sorted(behavior.items()))

        self.indirect_opinion_function: Callable[
            [Type["Peer"], Type["Peer"]], float
        ] = self.indirect_opinion

        self.direct_opinion_function: Callable[
            [Type["Peer"], Type["Peer"], List[Records]], float
        ] = (self.fuzzy_local_opinion if self.fuzzy else self.local_opinion)

        # Decay
        self.decay: Callable = decay
        self.forgiveness_delay: float = forgiveness_delay
        self.initial_reput: float | Dict[str, float] = initial_reput

        # Handle failure simulation
        self.down_until: float = 0.0
        self.down_duration: float = 0.3
        self.breaking_rate: float = breaking_rate

        # Peer identification
        self.label: str = label
        self.id: str = id

        # Stats purposes
        self.nb_interactions = peer_interactions
        self.random_count: int = 0

        # Handle neighbor & cache
        self.neighbors: Dict["Peer", Records] = {}
        self.last_trust_update: Dict["Peer", float] = {}
        self.local_opinion_cache: Dict["Peer", float] = {}
        self.last_observation_update: Dict["Peer", float] = {}
        self.local_observations_cache: Dict["Peer", Dict[str, float]] = {}
        self.delta_local: float = 0.0
        self._similarity: Dict["Peer", float] = {}
        self.last_similarity = 0.0
        self.delta_similarity: float = 0.0
        self.timestamp: float = 0.0

        # Capacity of the peer
        self.ongoing_transactions: Dict[AccessNetwork, List[OngoingTransaction]] = {}
        self.total_capacity: float = total_capacity
        self.capacity: Dict[AccessNetwork, float] = {}
        self.transaction_duration: float = transaction_duration

        # Handle RAN network simulation.
        self.cn: CoreNetwork
        self.ans: List[AccessNetwork] = []

    def add_neighbors(self, peers: List["Peer"]):
        """"""
        for p in peers:
            if p in self.neighbors:
                pass
            # elif p is self:
            # raise ValueError("A peer can't be it's own neighbor")
            else:
                if self.fuzzy:
                    self.neighbors[p] = FuzzyRecords(self.decay)
                else:
                    self.neighbors[p] = Records(self.decay)
                # cache freshness negative so that delta won't trigger them.
                self.last_trust_update[p] = -10.0
                # TODO_naming replace opinion with trust
                self.local_opinion_cache[p] = self.initial_reput
                self.last_observation_update[p] = -10.0
                self.local_observations_cache[p] = initial_crisp_values

    def _update_ans_capacity(self):
        """Split the peer total capacity among all it's AccesNetworks."""
        an_nb = len(self.ans)
        for an in self.ans:
            self.capacity[an] = round(self.total_capacity / an_nb, 3)

    def add_an(self, an: AccessNetwork):
        """Add an acces network for the peer and manage it's capacity

        Args:
            an (AccessNetwork): access network that the peer operate on.
        """
        self.ans.append(an)
        self.capacity[an] = 0
        self._update_ans_capacity()
        self.ongoing_transactions[an] = []

    def select_random_neighbor(self) -> "Peer":
        """Randomly select and return one neighbor

        Returns:
            Peer: Peer that is a neighbor of self.
        """
        return self.rng.choice(list(self.neighbors))

    def is_down(self, timestamp: float) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return True if self.down_until >= timestamp else False

    def update_transactions(self, an: AccessNetwork) -> float:
        for t in self.ongoing_transactions[an][:]:
            if t.timestamp + t.duration <= self.timestamp:
                self.end_transaction(t)
                self.ongoing_transactions[an].remove(t)

    def current_capacity(self, an: AccessNetwork) -> float:
        self.update_transactions(an)
        return self.capacity[an] - sum(
            [t.weight for t in self.ongoing_transactions[an]]
        )

    def availability_check(self, timestamp: float, an: AccessNetwork) -> bool:
        """Check wether the probed peer should accept the transaction.

        Args:
            timestamp (float): time at the end of the transaction.

        Returns:
            bool: True if the request is accepted, False otherwise.
        """

        # TODO_evolution Refactor capacity handling & couple it to bandwith for multi-criterion cases.
        if an not in self.ans:
            # The current peer dont have any capacity on the requested AccessNetwork.
            return False
        if self.current_capacity(an) >= 1:
            return True
        else:
            return False

    def initiate_transaction(
        self, partner: "Peer", timestamp: float, an: AccessNetwork, reput_at_init: float
    ) -> bool:
        """Process a transaction request

        Args:
            partner (Peer): Peer that requested the transaction.
            timestamp (float): time at the end of the transaction.

        Returns:
            bool: True if the request is accepted, False otherwise.
        """
        # TODO_evolution handle neighbors adjacency using core networks
        if partner not in self.neighbors:
            raise ValueError("The transaction partner must be in the emiter peers")
        self.timestamp = timestamp
        self.ongoing_transactions[an].append(
            OngoingTransaction(
                partner, self.timestamp, partner.transaction_duration, reput_at_init
            )
        )
        return True

    def end_transaction(self, t: OngoingTransaction):
        """process a transaction that has ended and take its outcome into account.
        Args:
            t (OngoingTransaction): The transaction that just ended.
        """
        time = t.timestamp + t.duration

        if self.transaction_type == SimpleTransaction:
            t.requester_peer.neighbors[self].add_transaction(
                self.transaction_type(time, self.transaction_outcome(time))
            )

        elif self.transaction_type == FuzzyTransaction:
            outcome: Dict[str, float] = self.transaction_outcome(time)
            # count varitions due to a temporary failure
            failure: float = outcome == worst_fuzzy_outcome
            #
            t.requester_peer.neighbors[self].add_transaction(
                FuzzyTransaction(
                    time,
                    outcome,
                    transaction_satisfaction(
                        t.requester_peer.fuzzy_subjective_trust(outcome)
                    ),
                    transaction_satisfaction(t.reput_on_transaction_init),
                    failure,
                )
            )
        else:
            raise (ValueError(f"{self.transaction_type} is not implemented"))

    def transaction_outcome(self, t: float = 0.0) -> Any:
        """return the transaction outcome depending on the peer behavior.

        Args:
            t (float, optional): Timing of the transaction. Defaults to 0.

        Returns:
            (Any): transaction outcome, depend on the type of transaction.
        """
        # Return when broken is different when in normal or fuzzy case.
        if self.fuzzy:
            negative_outcome = worst_fuzzy_outcome
        else:
            negative_outcome = False

        # Handle the breaking_rate
        # if self.is_down(t):
        #     return negative_outcome

        if self.rng.random() > 1 - self.breaking_rate:
            # self.down_until = t + self.rng.normal(0.3, 0.5)
            return negative_outcome

        if self.transaction_type == SimpleTransaction:
            success_rate = self.assert_success_rate(t)
            return self.rng.choice([True, False], p=[success_rate, 1 - success_rate])
        elif self.transaction_type == FuzzyTransaction:
            return self.assert_fuzzy_outcome(t)
        else:
            raise (ValueError(f"{self.transaction_type} is not implemented"))

    def assert_fuzzy_outcome(self, t: float) -> Dict[str, float]:
        """Return the success rate of the Peer at time t.

        Args:
            t (float): time for the peer.

        Returns:
            Dict[str,float]: metrics outcome for the transaction on this peer.
        """
        if t < 0:
            raise ValueError("Time t must be >=0")
        # TODO_evolution : handle peer dispersion
        buffer: float = 0.0
        for t_p in self.behavior:
            if t < t_p:
                break
            else:
                buffer = t_p
        return self.behavior[buffer]

    def assert_success_rate(self, t: float) -> float:
        """Return the success rate of the Peer at time t.

        Args:
            t (float): time for the peer.

        Returns:
            float: success rate of the peer at time t.
        """
        if t < 0:
            raise ValueError("Time t must be >=0")
        buffer: float = 0.0
        for t_p in self.behavior:
            if t < t_p:
                break
            else:
                buffer = t_p
        return self.behavior[buffer]

    # Unfuzzy pendant of local_observation + fuzzy_trust
    def _local_opinion(self, peer: "Peer") -> float:
        """Trust that a current peer have on another peer based on it's own direct observation.
        Take decay into account.

        Args:
            peer (Peer): _description_

        Returns:
            float: _description_
        """
        self.neighbors[peer].forgive_transactions(
            self.timestamp, self.forgiveness_delay
        )
        time_window: List[SimpleTransaction] = self.decay(
            transactions=self.neighbors[peer].transactions,
            reputation=self.direct_opinion_function,
            reputation_function_parameter=[peer],
            timestamp=self.timestamp,
        )
        # TODO_coherence : either use direct_opinion_function for both fuzzy and unfuzzy
        # or direcly specify local_opinion for non fuzzy reput
        return self.direct_opinion_function(peer, time_window)

    # Fuzzy pendant of _local_opinion (once combined with fuzzy_trust)
    def local_observations(self, peer: "Peer") -> Dict[str, float]:
        """Return the decayed local observations from records (e.g. crisp values).
        Necessary to implement fuzzy trust where crisp values are shared instead of trust.

        Args:
            peer (Peer): Peer that is rated.

        Returns:
            Dict[str,float]: trust in rated peer.
        """
        # Cache handler
        if not (
            self.last_observation_update[peer] >= self.timestamp - self.delta_local
        ):

            # Forgiveness
            self.neighbors[peer].forgive_transactions(
                self.timestamp, self.forgiveness_delay
            )

            # Decay
            time_window: List[FuzzyTransaction] = self.decay(
                transactions=self.neighbors[peer].transactions,
                reputation=self.direct_opinion_function,
                reputation_function_parameter=[],
                timestamp=self.timestamp,
            )

            if not time_window:
                return initial_crisp_values

            # Crisp values combination
            weight_total: float = sum([t.weight for t in time_window])
            crisp_values: Dict[str, float] = sum_dicts(
                [t.weighted_outcome() for t in time_window]
            )
            self.local_observations_cache[peer] = {
                k: v / weight_total for k, v in crisp_values.items()
            }
            self.last_observation_update[peer] = self.timestamp

        return self.local_observations_cache[peer]

    def fuzzy_subjective_trust(self, crisp_values: Dict[str, float]) -> float:
        """Evaluate trust based on raw crisp values using the current peer subjective point of view

        Args:
            crisp_values (Dict[str,float]): Crisp values used for trust computation

        Returns:
            float: trust score defined on 0-1
        """
        # No initial transactions case.
        if not crisp_values:
            crisp_values = self.initial_reput
        self.fuzzy_cs.inputs(
            {k: v for k, v in crisp_values.items() if k in self.fuzzy_required_input}
        )
        self.fuzzy_cs.compute()
        return self.fuzzy_cs.output["trust"]

    # Fuzzy local opinion, take decay into account.
    def _fuzzy_local_opinion(self, peer: "Peer") -> float:
        """Fuzzy subjective trust that the current peer have on another peer based on it's own direct observation.

        Args:
            peer (Peer): Peer that is rated.
        Returns:
            float: trust in rated peer.
        """
        if not (self.last_trust_update[peer] >= self.timestamp - self.delta_local):

            crisp_values = self.local_observations(peer)
            if not crisp_values:
                return self.initial_reput

            self.local_opinion_cache[peer] = self.fuzzy_subjective_trust(crisp_values)
            self.last_trust_update[peer] = self.timestamp

        return self.local_opinion_cache[peer]

    # Fuzzy local opinion
    def fuzzy_local_opinion(self, time_window: List[Transaction]) -> float:
        """Fuzzy subjective trust that the current peer have on another peer based on it's own direct observation.

        Args:
            time_window(List[Transaction]) : Time windows that is already decayed.

        Returns:
            float: trust in rated peer.
        """
        if not time_window:
            return self.fuzzy_subjective_trust(self.initial_reput)
        # Crisp values combination
        # Mean of the opinion on each transaction instead of an aggregated SLA
        # Make more sense but currently inapplicable for indirect opinion
        weight_total = sum([t.weight for t in time_window])
        r_aggregated: float = 0.0
        for t in time_window:
            r_aggregated += self.fuzzy_subjective_trust(t.outcome) / t.weight
        return r_aggregated / weight_total
        
        weighted_transactions = [t.weighted_outcome() for t in time_window]
        crisp_values: Dict[str, float] = {}
        for metric in weighted_transactions[0].keys():
            crisp_values[metric] = gmean(
                [w[metric] for w in weighted_transactions],
                weights=[t.weight for t in time_window],
            )
        return self.fuzzy_subjective_trust(crisp_values)

        weight_total: float = sum([t.weight for t in time_window])
        return self.fuzzy_subjective_trust(
            {k: v / weight_total for k, v in crisp_values.items()}
        )
        
        
    # Non Fuzzy local opinion
    # TODO_bug : the cache in local opinion probably limit the interest of the double windows calculation
    # (i.e. small and big windows will have the same resutls), check this.
    def local_opinion(self, peer: "Peer", time_window: List[Transaction]) -> float:
        """Trust that a current peer have on another peer based on it's own direct observation.

        Args:
            peer (Peer): Peer that is rated.
            time_window(List[Transaction]) : Time windows that is already decayed.
        Returns:
            float: trust in rated peer.
        """
        if not time_window:
            return self.initial_reput

        if not (self.last_trust_update[peer] >= self.timestamp - self.delta_local):
            # Update peer local opinion if necessary.
            trust: float = 0.0
            max_trust: float = 0.0
            for transaction in time_window:
                max_trust += transaction.weight
                if transaction.outcome:
                    trust += transaction.weight
            self.last_trust_update[peer] = self.timestamp
            self.local_opinion_cache[peer] = trust / max_trust
        return self.local_opinion_cache[peer]

    def indirect_opinion(self, peer) -> float:
        """Opinion that neighbors + current peer have of another peer, ponderated using similarity.

        Args:
            peer (Peer): Peer that is rated.

        Returns:
            float: trust in rated peer.
        """
        neigh_sim: Dict[Peer, float] = self.neighbors_similarity()

        # we don't ask a peer for advice on himself.
        if peer in neigh_sim:
            neigh_sim.pop(peer)

        similarity_sum: float = sum([s for s in neigh_sim.values()])
        if self.fuzzy:
            trust: float = 0.0
            similarity_sum += 1  # Self have similarity of 1 with itself by definition.
            trust += self._fuzzy_local_opinion(peer) * (1 / similarity_sum)
            for p, sim in neigh_sim.items():
                trust += self.fuzzy_subjective_trust(p.local_observations(peer)) * (
                    sim / similarity_sum
                )
            return trust
        else:
            i_trust: float = 0.0
            similarity_sum += 1  # Self have similarity of 1 with itself by definition.
            i_trust += self._local_opinion(peer) * (1 / similarity_sum)
            for p, sim in neigh_sim.items():
                i_trust += p._local_opinion(peer) * (sim / similarity_sum)
            return i_trust

    def neighbors_reputation(self) -> Dict["Peer", float]:
        """Return neighbors reputation."""
        neighbors_reput: Dict["Peer", float] = {}
        for peer in self.neighbors:
            neighbors_reput[peer] = self.indirect_opinion_function(peer)
        return neighbors_reput

    def select_available_peer(
        self, candidates: List["Peer"], an: AccessNetwork, random: bool = False
    ) -> "Peer":
        """Return a peer from a list of candidates. Make sure that peer is available and in the specified an.
        Returns:
            Peer: peer from candidates that is available and in the specified an.
        """
        while candidates:
            i = self.rng.integers(0, len(candidates) - 1) if random else 0
            p = candidates.pop(i)
            if p.availability_check(self.timestamp, an) and p in an.peers:
                return p
        return None

    # Add the opinion function & the decay function directly in order to make double window work.
    def choose_peer_for_transaction(
        self,
        random_factor: float,
        newcommer_random_factor: float,
        timestamp: float,
        an: AccessNetwork,
    ) -> Tuple["Peer", float]:
        """Return the most appropriat neighbor for the transaction.

        Args:
            random_factor (float, optional): 0.0-1.0 Rate at which a random neighbor will be returned.
            newcommer_random_factor (float, optional): 0.0-1.0 Rate at which a random neighbor that this peer never interacted with will be returned.
            timestamp (float): Instant at which the choice of the transaction should be made (used for optimization purposes).
            an (float): AcessNetwork where the transaction should take place.
            opinion_function (Callable[Peer,Peer]->float):
        Raises:
            IndexError: Raised when there are no neighbors
        Returns:
            (Peer,float): peer deemed the most appropriate for the the next transaction and it's current reputation. If there is no reputation return 0.
        """
        self.timestamp = timestamp
        if not self.neighbors:
            raise IndexError(
                "Neighbors are empty, can't choose a nieghbor for transaction"
            )
        # Handle the two different random selection factors.
        if newcommer_random_factor:
            if self.rng.random() > (1 - newcommer_random_factor):
                nr: Dict[Peer, float] = self.neighbors_reputation()
                l = list(
                    [
                        n
                        for n, r in nr.items()
                        if self.initial_reput - 0.01 <= r <= self.initial_reput + 0.01
                    ]
                )
                if len(l) >= 1:
                    candidate = self.select_available_peer(l, an, random=True)
                    if candidate:
                        return candidate, 1.0
                        # TODO_fuzzy : create an agnostic handle to _local_opinion and _fuzzy_local_opinion
                        # Use it here instead of hadcoded 1.0

        if self.rng.random() > (1 - random_factor):
            self.random_count += 1
            candidate = self.select_available_peer(
                list(self.neighbors), an, random=True
            )
            if candidate:
                return candidate, self.indirect_opinion_function(candidate)

        n_reput: Dict[Peer, float] = dict(
            sorted(
                self.neighbors_reputation().items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        candidate = self.select_available_peer(list(n_reput.keys()), an)
        if candidate:
            return candidate, n_reput[candidate]

        # There are no available peers in the requested AccessNetwork
        if len(an.peers) == 0:
            an.no_peer_for_transaction += 1
        else:
            an.no_capacity_for_transaction += 1
        return None, -1

    def common_neighbors(self, peer: Type["Peer"]) -> List["Peer"]:
        """Find common neighbors between two peers

        Returns:
            List(Peer): Neighbors that the current peer and the provided peer have in common.
        """
        neigh_peers = peer.ask_neighbors()
        return list(set(neigh_peers) & set(self.neighbors.keys()))

    def similarity(self, peer: Type["Peer"]) -> float:
        """
        Return a similarity score as defined in eq. 5 from PeerTrust
        """
        # Control the relative value of the local peer compared to other peer given similarity over different metrics.

        cneigh: List[Peer] = self.common_neighbors(peer)
        diff = 0
        for p in cneigh:
            # Update timestamp of the observed peer to obtain comparable results
            p.timestamp = self.timestamp
            if self.fuzzy:
                d_dicts = diff_dicts(
                    self.local_observations(p), peer.local_observations(p)
                )
                d_dicts = fuzzy_diff_normalization(d_dicts)
                diff += sum([v**2 for v in d_dicts.values()]) / len(d_dicts)
            else:
                diff += (abs(self._local_opinion(p) - peer._local_opinion(p))) ** 2
        sim = 1 - sqrt((diff / len(cneigh)))
        try:
            assert 0.0 <= sim <= 1.0
        except AssertionError:
            print(
                "Out of bould similarity between the two peers check the similarity function"
            )
        return sim

    def neighbors_similarity(self) -> Dict[Type["Peer"], float]:
        """Similarity of all neighbors + cache handler

        Returns:
            Dict[Type["Peer"],float]: similarity with different neighbors
        """
        if (
            not (self.last_similarity >= self.timestamp - self.delta_similarity)
            or not self._similarity
        ):
            self._similarity = {p: self.similarity(p) for p in self.neighbors.keys()}
            self.last_similarity = self.timestamp
        return self._similarity

    def get_transactions(self, peer: Type["Peer"]) -> Records:
        """Return the transactions of a peer to another peer

        Args:
            peer (Peer): Peer on which information should be given.

        Returns:
            Records: Object that contain the transactions for the peer.
        Raise:
            KeyError : Requested peer is not in the neighbors
        """
        try:
            return self.neighbors[peer]
        except KeyError as e:
            print(f" Peer {peer} is not in this peer neighbors {e}")
            raise e

    def ask_neighbors(self) -> List["Peer"]:
        """Return the neighbors of a specific peer

        Returns:
            List[Peer]: Peer that are neighbors of the current peer.
        """
        return list(self.neighbors.keys())
