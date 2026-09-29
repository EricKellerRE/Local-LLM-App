import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_model_app.research_mcp import (
    _require_public_url,
    _scholarly_metadata,
    _store_original_document,
)


class ResearchMcpTests(unittest.TestCase):
    def test_private_and_local_urls_are_rejected_before_fetch(self) -> None:
        for url in ("http://localhost/admin", "http://127.0.0.1/private", "file:///etc/passwd"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _require_public_url(url)

    def test_scholarly_metadata_distinguishes_abstract_from_generic_description(self) -> None:
        html = """
        <meta name="citation_title" content="Memory consolidation mechanisms">
        <meta name="citation_author" content="Ada Example">
        <meta name="citation_author" content="Ben Example">
        <meta name="citation_publication_date" content="2024/03/02">
        <meta name="citation_journal_title" content="Journal of Memory">
        <meta name="citation_abstract" content="A structured abstract about persistence.">
        """
        value = _scholarly_metadata(html, "https://example.org/paper")
        self.assertEqual(value["title"], "Memory consolidation mechanisms")
        self.assertEqual(value["authors"], ["Ada Example", "Ben Example"])
        self.assertEqual(value["year"], 2024)
        self.assertEqual(value["venue"], "Journal of Memory")
        self.assertEqual(value["abstract_source"], "citation_abstract")

        generic = _scholarly_metadata(
            '<meta property="og:description" content="A search-engine page description.">',
            "https://example.org/page",
        )
        self.assertIsNone(generic["abstract"])
        self.assertEqual(generic["description"], "A search-engine page description.")
        self.assertEqual(generic["description_source"], "og:description")

    def test_original_document_is_stored_outside_model_payload_with_integrity_metadata(self) -> None:
        raw = b"%PDF-1.7\noriginal paper bytes"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary, patch.dict(
            os.environ, {"LOCAL_APP_DATA_DIRECTORY": temporary}
        ):
            result = _store_original_document(
                raw,
                "https://papers.example/study.pdf",
                "application/pdf",
                "task-123",
                "source_456",
            )
            saved = Path(temporary) / result["original_file"]

            self.assertEqual(saved.read_bytes(), raw)
            self.assertEqual(saved.name, "source_456.pdf")
            self.assertEqual(result["original_bytes"], len(raw))
            self.assertEqual(result["original_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result["original_content_type"], "application/pdf")

    def test_original_document_rejects_unsafe_archive_identifiers(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary, patch.dict(
            os.environ, {"LOCAL_APP_DATA_DIRECTORY": temporary}
        ):
            with self.assertRaises(ValueError):
                _store_original_document(
                    b"paper",
                    "https://papers.example/study",
                    "text/html",
                    "../outside",
                    "source",
                )


if __name__ == "__main__":
    unittest.main()
