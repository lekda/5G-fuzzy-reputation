"""test for class Records
"""
from reput.record import Records
from reput.transactions import SimpleTransaction
from reput.decay import exponential_decay


class TestRecords:
    def setup_class(self):
        self.r = Records(exponential_decay)
        self.r.add_transaction(SimpleTransaction(0.01, True))
