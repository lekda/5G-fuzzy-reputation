"""unit test for decay functions 
"""
from typing import List
from unittest import TestCase

from reput.decay import exponential_decay, simple_window, adaptive_windows
from reput.transactions import SimpleTransaction, Transaction, DecayedTransaction
from reput.record import Records


def test_exponential_decay():
    t1 = SimpleTransaction(0.01, True)
    t2 = SimpleTransaction(0.02, False)
    t3 = SimpleTransaction(0.8, True)

    #  Ajouter ces deux tabs
    assert exponential_decay(transactions=[]) == []
    dt: List[DecayedTransaction] = exponential_decay([t1, t2, t3])
    assert dt[0].weight <= dt[1].weight
    assert dt[2].weight == 1


def test_simple_window():
    t1 = SimpleTransaction(0.1, True)
    t2 = SimpleTransaction(0.2, True)
    t3 = SimpleTransaction(0.4, True)
    ts: List[Transaction] = [t1, t2, t3]
    assert len(simple_window(ts, 0.4)) == 3
    assert len(simple_window(ts, 0.3)) == 2
    assert len(simple_window(ts, 0.1)) == 1
    assert simple_window(ts, 0.1)[0].timestamp == 0.4
    assert len(simple_window(ts, 1.0)) == 3


def test_adaptive_window():
    # TODO test
    pass
