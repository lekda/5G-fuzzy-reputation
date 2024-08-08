"""unit test for reputation 
"""

from reput.decay import exponential_decay
from reput.transactions import SimpleTransaction


class TestSimpleTransaction:
    def setup_class(self):
        self.transaction = SimpleTransaction(0.01, True)
