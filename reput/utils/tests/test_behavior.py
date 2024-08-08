"""tests for the behavior class
"""

from unittest import TestCase

from reput.utils.behavior import Behavior


class TestBehavior(TestCase):
    def setup_class(self):
        self.empty_behavior = ""
        self.undefined_0 = '{"0.2":1.0}'
        self.correct_behavior = '{"0.0":1.0,"0.5":0.0}'

    def test_init_behavior(self):
        with self.assertRaises(ValueError):
            Behavior(self.empty_behavior)
        self.b_undefined_0 = Behavior(self.undefined_0)
        self.b_correct_behavior = Behavior(self.correct_behavior)

    def test_get_behavior(self):
        self.b_undefined_0 = Behavior(self.undefined_0)
        self.b_correct_behavior = Behavior(self.correct_behavior)
        # value that doesn't exist
        with self.assertRaises(ValueError):
            self.b_undefined_0.get_behavior(0.0)

        ## nominal usecase
        assert self.b_correct_behavior.get_behavior(0.1) == 1.0

        ## exact match
        assert self.b_correct_behavior.get_behavior(0.0) == 1.0
        assert self.b_correct_behavior.get_behavior(0.5) == 0.0

        ## change of value
        assert self.b_correct_behavior.get_behavior(0.6) == 0.0

        ## big
        assert self.b_correct_behavior.get_behavior(30.5) == 0.0
