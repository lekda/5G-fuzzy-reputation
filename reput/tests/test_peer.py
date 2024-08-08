"""unit test for Peer 
"""
from unittest import TestCase
from typing import List
from reput.environment import Environment

from reput.peer import Peer
from reput.record import Records


class TestPeer(TestCase):
    def setup_class(self):
        self.p1 = Peer(behavior={0.0: 0.8})
        self.p2 = Peer(behavior={0.0: 0.7})
        self.p3 = Peer(behavior={0.0: 0.2})

    def test_add_neighbor(self):
        self.p1.add_neighbors([self.p2, self.p3])
        assert self.p2 in self.p1.neighbors
        assert type(self.p1.neighbors[self.p2]) is Records

        with self.assertRaises(ValueError):
            self.p1.add_neighbors([self.p1])

        self.p1.add_neighbors([self.p2])

    def test_is_down(self):
        self.p1.down_until = 0.4
        assert self.p1.is_down(0.2) == True
        assert self.p1.is_down(0.0) == True
        assert self.p1.is_down(0.4) == True
        self.p1.down_until = 0.0

    def test_transaction_outcome(self):
        self.p1.down_until = 0.1
        assert self.p1.transaction_outcome((0.1)) == False

        self.p1.breaking_rate = 1.0
        assert self.p1.transaction_outcome((0.2)) == False

        self.p1.breaking_rate = 0.0
        self.p1.down_until = 0.1
        self.p1.behavior = {0.0: 1.0}

        assert self.p1.transaction_outcome((0.2)) == True
        self.p1.breaking_rate = 0.05
        self.p1.behavior = {0.0: 0.8}

        for i in range(10):
            res = self.p1.transaction_outcome((i * 0.1))
            assert (res == True) or (res == False)

    def test_make_transaction(self):
        timestamps: List[float] = [0.0, 0.1, 0.2, 1.1]
        self.p2.add_neighbors([self.p1, self.p3])
        for ts in timestamps:
            self.p2.initiate_transaction(self.p3, ts)
        p4 = Peer(behavior={0.0: 0.5})
        with self.assertRaises(ValueError):
            self.p2.initiate_transaction(p4, 1.2)

    def test_choose_peer_for_transaction(self):
        self.p1.add_neighbors([self.p2, self.p3])
        timestamps: List[float] = [0.0, 0.1, 0.2, 1.1]
        for t in timestamps:
            p = self.p1.choose_peer_for_transaction()
            self.p1.initiate_transaction(p, t)
        with self.assertRaises(IndexError):
            self.p3.choose_peer_for_transaction(0.9)

        self.p2.add_neighbors([self.p1, self.p3])
        with self.assertRaises(ValueError):
            self.p2.choose_peer_for_transaction(1.2)

        with self.assertRaises(ValueError):
            self.p2.choose_peer_for_transaction(-0.1)

    def test_get_transactions(self):
        self.p4 = Peer(behavior={0.0: 0.1})

        with self.assertRaises(KeyError):
            self.p4.get_transactions(self.p1)
        self.p4.add_neighbors([self.p1])
        p1p2trans = self.p4.get_transactions(self.p1)

        assert type(p1p2trans) is Records
        assert len(p1p2trans._transactions) == 0

    def test_ask_neighbors(self):
        self.p5 = Peer(behavior={0.0: 0.1})
        self.p5.add_neighbors([self.p1, self.p2])
        neigh = self.p5.ask_neighbors()
        assert set([self.p1, self.p2]) == set(neigh)

    def test_common_neighbors(self):
        self.p6 = Peer(behavior={0.0: 0.1})
        self.p7 = Peer(behavior={0.0: 0.1})

        self.p6.add_neighbors([self.p1, self.p2])
        assert self.p7.common_neighbors(self.p6) == []
        self.p7.add_neighbors([self.p3])
        assert self.p7.common_neighbors(self.p6) == []
        self.p7.add_neighbors([self.p2])
        assert self.p7.common_neighbors(self.p6) == [self.p2]

    def test_local_opinon(self):
        self.p1.neighbors = {}
        self.p1.add_neighbors([self.p2])

        assert self.p1.local_opinion(self.p2) == 0.5

        # arbitrarily set the transaction to a success for the test
        self.p1.initiate_transaction(self.p2, timestamp=0.2)
        r: Records = self.p1.neighbors[self.p2]
        r._transactions[0].outcome = True
        assert self.p1.local_opinion(self.p2) == 1

        # Same with a failure
        r._transactions[0].outcome = False
        assert self.p1.local_opinion(self.p2) == 0

        # Now a failure first and then a success
        self.p1.initiate_transaction(self.p2, timestamp=0.3)
        r._transactions[1].outcome = True
        assert self.p1.local_opinion(self.p2) >= 0.5

    def test_similarity(self):
        # Init test
        self.p1 = Peer(behavior={0.0: 0.8})
        self.p2 = Peer(behavior={0.0: 0.7})
        self.p3 = Peer(behavior={0.0: 0.2})
        self.p1.add_neighbors([self.p2, self.p3])
        self.p2.add_neighbors([self.p1, self.p3])
        self.p3.add_neighbors([self.p1, self.p2])

        assert self.p1.similarity(self.p2) == 1
        self.p2.initiate_transaction(self.p3, 0.2)
        self.p2.initiate_transaction(self.p3, 0.3)
        assert 0.0 <= self.p1.similarity(self.p2) <= 1
        [self.p2.initiate_transaction(self.p3, 0.3 + i * 0.1) for i in range(25)]
        assert 0.0 <= self.p1.similarity(self.p2) <= 1

    def test_indirect_opinion(self):
        # Init test
        self.p1 = Peer(behavior={0.0: 0.8})
        self.p2 = Peer(behavior={0.0: 0.7})
        self.p3 = Peer(behavior={0.0: 0.2})
        self.p1.add_neighbors([self.p2, self.p3])
        self.p2.add_neighbors([self.p1, self.p3])
        self.p3.add_neighbors([self.p1, self.p2])

        # test
        assert self.p1.indirect_opinion(self.p2) == 0.5
        self.p3.initiate_transaction(self.p2, 0.2)
        self.p3.initiate_transaction(self.p2, 0.3)
        a = self.p1.indirect_opinion(self.p2)
        assert self.p1.indirect_opinion(self.p2) != 0.5
        assert 0 <= self.p1.indirect_opinion(self.p2) <= 1


def test_assert_success_rate():
    p1 = Peer(behavior={0.0: 0.5, 2.0: 0.2, 4.0: 0.8})
    assert p1.assert_success_rate(0.0) == 0.5
    assert p1.assert_success_rate(0.1) == 0.5
    assert p1.assert_success_rate(2.0) == 0.2
    assert p1.assert_success_rate(2.3) == 0.2
    assert p1.assert_success_rate(4.1) == 0.8
    assert p1.assert_success_rate(55) == 0.8
