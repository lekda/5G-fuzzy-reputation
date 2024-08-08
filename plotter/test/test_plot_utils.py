"""test for plot utils
"""
from plotter.plot_utils import *


class TestRecords:
    def setup_class(self):
        self.stats_overtime_1 = load_stats_overtime(
            Path("./plotter/test/data_for_tests/simulation.seed=56"),
            multiseed=False,
        )
        self.stats_overtime_2 = load_stats_overtime(
            Path("./plotter/test/data_for_tests/simulation.seed=57"),
            multiseed=False,
        )
        self.path_stats_overtime_3 = Path("./plotter/test/data_for_tests/")

    def test_merge_dict(self):
        merged = merge_dicts(self.stats_overtime_1, self.stats_overtime_2)
        assert merged["labels_stats"]["q1"]["good"]["total"] == 422 + 387

    def test_divide_int_in_dict(self):
        divided = divide_nested_dict(
            merge_dicts(self.stats_overtime_1, self.stats_overtime_2), 2
        )
        assert divided["labels_stats"]["q1"]["good"]["total"] == int((422 + 387) / 2)

    def test_load_reput_per_round(self):
        r_emited_per_round = load_reput_overtime(
            self.path_stats_overtime_3, emited=True
        )
        r_received_per_round = load_reput_overtime(self.path_stats_overtime_3)
        assert (
            r_emited_per_round["q1"]["good_001"]["good_002"]
            == r_received_per_round["q1"]["good_002"]["good_001"]
        )

    def test_load_reput_per_round_on_label(self):
        r_emited_per_round = load_reput_on_label_overtime(
            self.path_stats_overtime_3, "good", emited=True
        )
        for v in r_emited_per_round.values():
            for k in v.keys():
                assert "outage" not in k
        r_emited_per_round = load_reput_on_label_overtime(
            self.path_stats_overtime_3, "outage", emited=True
        )
        for v in r_emited_per_round.values():
            for k in v.keys():
                assert "good" not in k
