""" Definition of types for the reputation system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class OngoingTransaction:
    def __init__(
        self,
        requester_peer,
        timestamp: float,
        duration: float,
        reput_on_transaction_init: Any,
        weight: float = 1,
    ):
        self.requester_peer = requester_peer
        self.timestamp = timestamp
        self.duration = duration
        self.reput_on_transaction_init: Any = reput_on_transaction_init
        self.weight = weight


class Transaction(ABC):
    def __init__(self, timestamp: float, outcome: Any):
        self.timestamp: float = timestamp
        self.outcome: Any = outcome
        self.weight: int

    def weighted_outcome(self) -> Any:
        pass


class FuzzyTransaction(Transaction):
    def __init__(
        self,
        timestamp: float,
        outcome: Dict[str, float],
        evaluation: str,
        expected_satisfaction: str,
        failure: float,
    ):
        self.timestamp = timestamp
        self.outcome: Dict[str, float] = outcome
        self.satisfaction: str = evaluation
        # Reput when the transaction was chosen, used for stats purposes.
        self.expected_satisfaction: str = expected_satisfaction
        # Did a failure occur during this transaction ? used for stats purposes.
        self.failure: float = failure
        self.weight: float = 1  # Bit of a hack for adaptive windows TODO replace

    def weighted_outcome(self) -> Dict[str, float]:
        return {k: v * self.weight for k, v in self.outcome.items()}


class SimpleTransaction(Transaction):
    def __init__(self, timestamp: float, outcome: bool):
        self.timestamp = timestamp
        self.outcome = outcome
        self.weight = 1  # Bit of a hack for adaptive windows TODO replace

    def __add__(self, other):
        if isinstance(other, SimpleTransaction):
            return self.bool_to_rating(other.outcome) + self.bool_to_rating(
                self.outcome
            )

        elif isinstance(other, (int, float)):
            return self.bool_to_rating(self.outcome) + other
        else:
            raise TypeError(
                "SimpleTransaction support addition with int,float and other SimpleTransaction object."
            )

    def bool_to_rating(self, b: bool) -> int:
        """Tranform boolean to 1 or -1 value

        Args:
            b (bool): boolean to transform

        Returns:
            int: 1 if True -1 if False
        """
        return 1 if b else -1


# ---
# Legacy
class DecayedTransaction(ABC):
    def __init__(self, weight: float, outcome: Any):
        self.weight = weight
        self.outcome = outcome
