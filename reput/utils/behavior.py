import json
from typing import Dict, List


class Behavior:
    def __init__(self, behavior: str):
        """_summary_

        Args:
            behavior (str): _description_
        """
        if not behavior:
            raise ValueError("behavior shouldn't be left empty")
        self.b: Dict[float:float] = {
            float(k): v for k, v in json.loads(behavior).items()
        }
        self.k: List[float] = list(self.b.keys())

    def get_behavior(self, t: float) -> float:
        """Return behavior at time t.

        Args:
            t (float): time at which the behavior is observed. >= 0

        Returns:
            float: behavior at time t
        """
        closest_inferior = max(x for x in self.k if x <= t)
        return self.b[closest_inferior]
