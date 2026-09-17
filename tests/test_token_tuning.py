import unittest

from local_model_app.token_tuning import bracketed_budget_search


class TokenBudgetSearchTests(unittest.TestCase):
    def test_finds_and_confirms_smallest_passing_candidate(self) -> None:
        calls = []

        def evaluate(budget: int) -> bool:
            calls.append(budget)
            return budget >= 384

        result = bracketed_budget_search(
            [128, 192, 256, 384, 512, 768, 1024],
            evaluate,
            repeats=2,
        )

        self.assertEqual(result.selected_budget, 384)
        self.assertIn(256, [trial.budget for trial in result.trials])
        self.assertIn(512, [trial.budget for trial in result.trials])
        self.assertTrue(all(calls.count(budget) == 2 for budget in set(calls)))

    def test_returns_none_when_the_largest_candidate_fails(self) -> None:
        result = bracketed_budget_search([128, 256, 512], lambda budget: False)

        self.assertIsNone(result.selected_budget)
        self.assertEqual([trial.budget for trial in result.trials], [512])

    def test_required_pass_count_allows_a_controlled_failure_rate(self) -> None:
        attempts = {}

        def evaluate(budget: int) -> bool:
            attempts[budget] = attempts.get(budget, 0) + 1
            return budget >= 256 and attempts[budget] <= 2

        result = bracketed_budget_search(
            [128, 256, 512],
            evaluate,
            repeats=3,
            required_passes=2,
        )

        self.assertEqual(result.selected_budget, 256)
        selected = next(trial for trial in result.trials if trial.budget == 256)
        self.assertEqual(selected.passes, 2)
        self.assertTrue(selected.passed)

    def test_rejects_an_invalid_candidate_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "ascending order"):
            bracketed_budget_search([256, 128], lambda budget: True)


if __name__ == "__main__":
    unittest.main()
