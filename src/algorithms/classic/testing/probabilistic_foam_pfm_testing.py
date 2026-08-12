"""
Testing class for the original Probabilistic Foam Method (PFM).

Extends BasicTesting with foam-specific metrics:
- total_bubbles: number of bubbles created during the search
- average_bubble_radius: mean bubble radius across all bubbles
"""

from typing import Dict, Any

from algorithms.basic_testing import BasicTesting
from simulator.services.services import Services


class ProbabilisticFoamPFMTesting(BasicTesting):

    def __init__(self, services: Services) -> None:
        super().__init__(services)

    def get_results(self) -> Dict[str, Any]:
        res: Dict[str, Any] = super().get_results()

        algo = self._services.algorithm.instance
        if hasattr(algo, '_bubbles'):
            bubbles = algo._bubbles
            res["total_bubbles"] = len(bubbles)
            if bubbles:
                res["average_bubble_radius"] = round(
                    sum(b["radius"] for b in bubbles) / len(bubbles), 2
                )
            else:
                res["average_bubble_radius"] = 0.0
        else:
            res["total_bubbles"] = 0
            res["average_bubble_radius"] = 0.0

        return res

    def print_results(self) -> None:
        super().print_results()
        results = self.get_results()
        from simulator.services.debug import DebugLevel
        self._services.debug.write(
            "Total bubbles: " + str(results["total_bubbles"]), DebugLevel.BASIC)
        self._services.debug.write(
            "Average bubble radius: {0:.2f}".format(results["average_bubble_radius"]),
            DebugLevel.BASIC)
