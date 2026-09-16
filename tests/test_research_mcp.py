import unittest

from local_model_app.research_mcp import _require_public_url


class ResearchMcpTests(unittest.TestCase):
    def test_private_and_local_urls_are_rejected_before_fetch(self) -> None:
        for url in ("http://localhost/admin", "http://127.0.0.1/private", "file:///etc/passwd"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _require_public_url(url)


if __name__ == "__main__":
    unittest.main()
