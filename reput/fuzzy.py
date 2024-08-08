from typing import List, Dict

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


worst_latency: float = 350
best_latency: float = 1.0
best_bandwith: float = 1000
worst_bandwith: float = 1.0
best_loss: float = 100000
worst_loss: float = 1.0

worst_fuzzy_outcome = {
    "latency": worst_latency,
    "bandwith": worst_bandwith,
    "loss": worst_loss,
}
initial_crisp_values = {
    "latency": best_latency,
    "bandwith": best_bandwith,
    "loss": best_loss,
}
satisfaction_terms: List[str] = [
    "very satisfied",
    "satisfied",
    "neutral",
    "unsatisfied",
    "very unsatisfied",
]

# Dispersion of the crisp values outcome from a peer
# To normalize using best possible values for each fuzzy variable
dispersion_map: Dict[str, float] = {"low": 0.1, "medium": 0.2, "high": 0.4}

## Antecedents
u_latency = np.arange(0, 350, 0.1)
latency = ctrl.Antecedent(u_latency, "latency")

u_loss = np.arange(0, 100000, 1)
loss = ctrl.Antecedent(u_loss, "loss")

u_bandwith = np.arange(0, 1000, 1)
bandwith = ctrl.Antecedent(u_bandwith, "bandwith")

## Consequent
trust = ctrl.Consequent(np.arange(0.0, 1.0, 0.01), "trust")

## Antecedents membership functions
latency["excellent"] = fuzz.trapmf(u_latency, [0.0, 0, 1, 5.0])
latency["good"] = fuzz.trapmf(u_latency, [1.0, 5.0, 7.5, 20.0])
latency["neutral"] = fuzz.trapmf(u_latency, [7.5, 20, 40, 50])
latency["bad"] = fuzz.trapmf(u_latency, [40, 50, 75, 300])
latency["very bad"] = fuzz.trapmf(u_latency, [75, 300, 350, 350])

bandwith["very bad"] = fuzz.trapmf(u_bandwith, [1, 1, 1, 10])
bandwith["bad"] = fuzz.trapmf(u_bandwith, [1, 10, 50, 100])
bandwith["neutral"] = fuzz.trapmf(u_bandwith, [50, 100, 100, 300])
bandwith["good"] = fuzz.trapmf(u_bandwith, [100, 300, 500, 750])
bandwith["excellent"] = fuzz.trapmf(u_bandwith, [500, 750, 1000, 1000])

loss["excellent"] = fuzz.trapmf(u_loss, [10000, 50000, 100000, 100000])
loss["good"] = fuzz.trapmf(u_loss, [1000, 5000, 10000, 50000])
loss["neutral"] = fuzz.trapmf(u_loss, [500, 1000, 1000, 5000])
loss["bad"] = fuzz.trapmf(u_loss, [10, 100, 500, 1000])
loss["very bad"] = fuzz.trapmf(u_loss, [0, 0, 10, 100])

## Consequent membership functions
trust["excellent"] = fuzz.trimf(trust.universe, [0.75, 1.0, 1.0])
trust["good"] = fuzz.trimf(trust.universe, [0.5, 0.75, 1.0])
trust["neutral"] = fuzz.trimf(trust.universe, [0.25, 0.5, 0.75])
trust["bad"] = fuzz.trimf(trust.universe, [0.0, 0.25, 0.5])
trust["very bad"] = fuzz.trimf(trust.universe, [0.0, 0.0, 0.25])

# latence et bandwith excellent -> excellent
# latence good ou excellent et bandwith good -> good
# latence good et bandwith good ou excellent -> good
# latence neutre ou good ou excellent et bandwith neutre -> neutre
# latence neutre et bandwith neutre ou good ou excellent -> neutre
# latence bad et bandwith bad -> very bad
# latence bad ou bandwith bad -> bad
# latence ou bandwith very bad -> very bad

# eMBB ruleset
eMBB1 = ctrl.Rule(latency["excellent"] & bandwith["excellent"], trust["excellent"])
eMBB2 = ctrl.Rule(
    (latency["excellent"] | latency["good"]) & bandwith["good"], trust["good"]
)
eMBB3 = ctrl.Rule(
    (bandwith["excellent"] | bandwith["good"]) & latency["good"], trust["good"]
)
eMBB4 = ctrl.Rule(
    (bandwith["excellent"] | bandwith["good"] | bandwith["neutral"])
    & latency["neutral"],
    trust["neutral"],
)
eMBB5 = ctrl.Rule(
    (latency["excellent"] | latency["good"] | latency["neutral"]) & bandwith["neutral"],
    trust["neutral"],
)
eMBB6 = ctrl.Rule(latency["bad"] & bandwith["bad"], trust["very bad"])
eMBB7 = ctrl.Rule(latency["bad"] | bandwith["bad"], trust["bad"])
eMBB8 = ctrl.Rule(latency["very bad"] | bandwith["very bad"], trust["very bad"])

# eMBB_trust_ctrl
eMBB_trust_ctrl = ctrl.ControlSystem(
    [eMBB1, eMBB2, eMBB3, eMBB4, eMBB5, eMBB6, eMBB7, eMBB8]
)
eMBB_required_input: List[str] = ["bandwith", "latency"]

# URLLC rule set
uRLLC1 = ctrl.Rule(latency["excellent"] & loss["excellent"], trust["excellent"])
uRLLC2 = ctrl.Rule(
    (latency["excellent"] | latency["good"]) & loss["good"], trust["good"]
)
uRLLC3 = ctrl.Rule((loss["excellent"] | loss["good"]) & latency["good"], trust["good"])
uRLLC4 = ctrl.Rule(
    (loss["excellent"] | loss["good"] | loss["neutral"]) & latency["neutral"],
    trust["neutral"],
)
uRLLC5 = ctrl.Rule(
    (latency["excellent"] | latency["good"] | latency["neutral"]) & loss["neutral"],
    trust["neutral"],
)
uRLLC6 = ctrl.Rule(latency["bad"] & loss["bad"], trust["very bad"])
uRLLC7 = ctrl.Rule(latency["bad"] | loss["bad"], trust["bad"])
uRLLC8 = ctrl.Rule(latency["very bad"] | loss["very bad"], trust["very bad"])

# urLLC_trust_ctrl
uRLLC_trust_ctrl = ctrl.ControlSystem(
    [uRLLC1, uRLLC2, uRLLC3, uRLLC4, uRLLC5, uRLLC6, uRLLC7, uRLLC8]
)
uRLLC_required_input: List[str] = ["loss", "latency"]

# mMTC ruleset
mMTC1 = ctrl.Rule(loss["excellent"] & bandwith["excellent"], trust["excellent"])
mMTC2 = ctrl.Rule((loss["excellent"] | loss["good"]) & bandwith["good"], trust["good"])
mMTC3 = ctrl.Rule(
    (bandwith["excellent"] | bandwith["good"]) & loss["good"], trust["good"]
)
mMTC4 = ctrl.Rule(
    (bandwith["excellent"] | bandwith["good"] | bandwith["neutral"]) & loss["neutral"],
    trust["neutral"],
)
mMTC5 = ctrl.Rule(
    (loss["excellent"] | loss["good"] | loss["neutral"]) & bandwith["neutral"],
    trust["neutral"],
)
mMTC6 = ctrl.Rule(loss["bad"] & bandwith["bad"], trust["very bad"])
mMTC7 = ctrl.Rule(loss["bad"] | bandwith["bad"], trust["bad"])
mMTC8 = ctrl.Rule(loss["very bad"] | bandwith["very bad"], trust["very bad"])

# mMTC_trust_ctrl
mMTC_trust_ctrl = ctrl.ControlSystem(
    [mMTC1, mMTC2, mMTC3, mMTC4, mMTC5, mMTC6, mMTC7, mMTC8]
)
mMTC_required_input: List[str] = ["bandwith", "loss"]

rulesets: Dict[str, ctrl.ControlSystem] = {
    "eMBB": eMBB_trust_ctrl,
    "URLLC": uRLLC_trust_ctrl,
    "mMTC": mMTC_trust_ctrl,
}
required_inputs: Dict[str, List[str]] = {
    "eMBB": eMBB_required_input,
    "URLLC": uRLLC_required_input,
    "mMTC": mMTC_required_input,
}


# Handmade membership function from crisp value to terms
# TODO_evolution find the adequate skfuzzy replacement
def transaction_satisfaction(value):
    label_map = {
        (0.8, 1.0): "very satisfied",
        (0.6, 0.8): "satisfied",
        (0.4, 0.6): "neutral",
        (0.2, 0.4): "unsatisfied",
        (float("-inf"), 0.2): "very unsatisfied",
    }

    for (lower, upper), label in label_map.items():
        if lower < value <= upper:
            return label


def fuzzy_diff_normalization(d: Dict[str, float]) -> Dict[str, float]:
    d["latency"] = min(best_latency / d["latency"], 1.0) if d["latency"] else 1.0
    d["bandwith"] = d["bandwith"] / best_bandwith
    d["loss"] = d["loss"] / best_loss
    try:
        for v in d.values():
            assert 0.0 <= v <= 1.0
    except AssertionError:
        print(
            f"One of the maximum values used for fuzzy parameters normalization isn't adequate {d}"
        )
    return d
