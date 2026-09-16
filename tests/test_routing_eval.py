import unittest

from evals.powerworld_routing_eval import QUERIES, golden_cases


class RoutingEvaluationDatasetTests(unittest.TestCase):
    def test_every_powerworld_tool_has_multiple_natural_language_requests(self) -> None:
        self.assertEqual(len(QUERIES), 46)
        self.assertTrue(all(len(queries) >= 2 for queries in QUERIES.values()))

    def test_dataset_includes_control_and_ambiguity_cases(self) -> None:
        categories = {case["category"] for case in golden_cases()}
        self.assertTrue({"tool", "ambiguous", "no_tool_needed", "missing_capability"} <= categories)


if __name__ == "__main__":
    unittest.main()
