"""Records class
"""

from .transactions import FuzzyTransaction, SimpleTransaction, Transaction

from typing import List, Tuple, Callable, Dict


class Records:
    """Record the interaction from a Peer with another Peer"""

    def __init__(self, decay: callable) -> None:
        """_summary_

        Args:
            decay (callable): callable that handle decay
        """
        self._transactions: List[SimpleTransaction] = []
        self.forgiving_index: int = 0
        self.decay: Callable = decay
        self.last_update: float = 0.0

    @property
    def transactions(self) -> List[SimpleTransaction]:
        return self._transactions[self.forgiving_index :]

    def add_transaction(self, transaction: SimpleTransaction):
        """"""
        self._transactions.append(transaction)

    def forgive_transactions(self, t: float, max_transaction_delta: float):
        """
        Negate the effect of oldest transactions from the Records .
            t (float): Current timestamp.
            max_transaction_delta (float): Time after which transactions are deleted
        """
        # Several counting methods can be used for clearing the list :
        # - Number of transactions between both peers
        # - Total number of transactions from the peer
        # - Time elapsed since last transaction -> this last step is chosen.

        t_oldest_possible_transaction = max(t - max_transaction_delta, 0.0)

        if (not self._transactions) or (not t_oldest_possible_transaction):
            return

        oldest_transaction = self._transactions[0]
        if oldest_transaction.timestamp >= t_oldest_possible_transaction:
            return

        # Delete record(s) that are too old to be relevant.
        while 1:

            if self.forgiving_index == len(self._transactions) or (
                self._transactions[self.forgiving_index].timestamp
                >= t_oldest_possible_transaction
            ):
                break
            self.forgiving_index += 1

    def count_transactions(
        self, start: float = 0, end: float = 0
    ) -> Tuple[int, int, int]:
        """Count transactions that occured for stats purposes.

        Returns:
            Tuple[int, int, int]: first value is the sum of positive transactions, second is negative transactions and last is total transactions.
        """
        if not end:
            if self._transactions:
                end: float = max(
                    self._transactions, key=lambda t: t.timestamp
                ).timestamp
            else:  # transactions list is empty
                return (0, 0, 0)
        if not start:
            start: float = 0

        positive = 0
        negative = 0
        for t in self._transactions:
            if start <= t.timestamp <= end:
                if t.outcome:
                    positive += 1
                else:
                    negative += 1
        return (positive, negative, positive + negative)


class FuzzyRecords:
    """Record the interaction from a Peer with another Peer"""

    def __init__(self, decay: callable) -> None:
        """_summary_

        Args:
            decay (callable): callable that handle decay
        """
        self._transactions: List[FuzzyTransaction] = []
        self.forgiving_index: int = 0
        self.decay: Callable = decay
        self.last_update: float = 0.0

    @property
    def transactions(self) -> List[FuzzyTransaction]:
        return self._transactions[self.forgiving_index :]

    def add_transaction(self, transaction: FuzzyTransaction):
        """"""
        self._transactions.append(transaction)

    def forgive_transactions(self, t: float, max_transaction_delta: float):
        """
        Negate the effect of oldest transactions from the Records .
            t (float): Current timestamp.
            max_transaction_delta (float): Time after which transactions are deleted
        """
        # Several counting methods can be used for clearing the list :
        # - Number of transactions between both peers
        # - Total number of transactions from the peer
        # - Time elapsed since last transaction -> this last step is chosen.

        t_oldest_possible_transaction = max(t - max_transaction_delta, 0.0)

        if (not self._transactions) or (not t_oldest_possible_transaction):
            return

        oldest_transaction = self._transactions[0]
        if oldest_transaction.timestamp >= t_oldest_possible_transaction:
            return

        # Delete record(s) that are too old to be relevant.
        while 1:

            if self.forgiving_index == len(self._transactions) or (
                self._transactions[self.forgiving_index].timestamp
                >= t_oldest_possible_transaction
            ):
                break
            self.forgiving_index += 1

    def count_transactions(
        self, start: float = 0, end: float = 0, expected: bool = False
    ) -> Dict[str, int]:
        """Count transactions that occured for stats purposes.

        Returns:
            Dict [str,int]: statistics on the transactions outcome for the given time frame.
            {   "total" : 361,
                "very satisfied" : 230,
                "satisfied" : 50,
                "neutral" : 24,
                "unsatisfied" : 13,
                "very unsatisfied" : 44
            }
        """
        outcome: Dict[str, float] = {
            "total": 0,
            "very satisfied": 0,
            "satisfied": 0,
            "neutral": 0,
            "unsatisfied": 0,
            "very unsatisfied": 0,
        }

        if not end:
            if self._transactions:
                end: float = max(
                    self._transactions, key=lambda t: t.timestamp
                ).timestamp
            else:  # transactions list is empty
                return outcome
        if not start:
            start: float = 0

        for t in self._transactions:
            if start <= t.timestamp <= end:
                if expected:
                    outcome[t.expected_satisfaction] += 1
                else:
                    outcome[t.satisfaction] += 1
        outcome["total"] = sum([v for v in outcome.values()])
        return outcome

    def count_failure(self, start: float = 0, end: float = 0) -> int:
        """Count the number of failure in the

        Args:
            start (float, optional): _description_. Defaults to 0.
            end (float, optional): _description_. Defaults to 0.

        Returns:
            int: _description_
        """
        if not end:
            if self._transactions:
                end: float = max(
                    self._transactions, key=lambda t: t.timestamp
                ).timestamp
            else:  # transactions list is empty
                return 0
        if not start:
            start: float = 0
        count: int = 0
        for t in self._transactions:
            if start <= t.timestamp <= end:
                if t.failure:
                    count += 1
        return count
