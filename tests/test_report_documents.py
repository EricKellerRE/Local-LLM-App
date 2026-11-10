import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from local_model_app.report_documents import (
    assemble_research_report,
    build_report_docx,
    canonical_report_url,
    format_source_reference,
    report_substantive_word_count,
    report_word_count,
    source_url_mentions,
    unique_urls,
)


class ReportDocumentTests(unittest.TestCase):
    def test_assembly_preserves_sections_and_deduplicates_source_index(self) -> None:
        report = assemble_research_report("Memory", [
            ("One", "Evidence one https://example.com/a"),
            ("Two", "Evidence two https://example.com/a and https://example.com/b"),
        ])

        self.assertIn("## One", report)
        self.assertIn("## Two", report)
        self.assertIn("## References", report)
        self.assertEqual(unique_urls(report), ["https://example.com/a", "https://example.com/b"])
        self.assertEqual(source_url_mentions(report), 5)
        self.assertGreaterEqual(report_word_count(report), 8)
        self.assertLess(report_substantive_word_count(report), report_word_count(report))

    def test_assembly_uses_scholarly_metadata_in_references(self) -> None:
        source = {
            "title": "Memory consolidation mechanisms",
            "url": "https://example.com/paper",
            "authors": ["Ada Example", "Ben Example"],
            "year": 2024,
            "venue": "Journal of Memory",
        }
        report = assemble_research_report(
            "Memory",
            [("Findings", "Supported claim https://example.com/paper")],
            sources=[source],
        )

        self.assertIn(
            "Ada Example, and Ben Example. (2024). "
            "[Memory consolidation mechanisms](https://example.com/paper). *Journal of Memory*.",
            report,
        )
        self.assertEqual(
            format_source_reference({"title": "Paper", "url": "https://example.com/paper"}),
            "[Paper](https://example.com/paper).",
        )

    def test_assembly_does_not_duplicate_an_existing_reference_section(self) -> None:
        report = assemble_research_report(
            "Memory",
            [("One", "Evidence https://example.com/a\n\n## References\n\n1. Existing")],
        )

        self.assertEqual(report.count("## References"), 1)

    def test_url_extraction_preserves_balanced_parentheses(self) -> None:
        text = "[Paper](https://example.com/article/A(14)B)"

        self.assertEqual(unique_urls(text), ["https://example.com/article/A(14)B"])

    def test_url_extraction_canonicalizes_cosmetic_variants(self) -> None:
        text = "https://EXAMPLE.com/paper/ and [Paper](https://example.com/paper#abstract)"

        self.assertEqual(unique_urls(text), ["https://example.com/paper"])
        self.assertEqual(canonical_report_url("https://example.com/"), "https://example.com")

    def test_docx_builder_creates_valid_word_package(self) -> None:
        markdown = """# Memory mechanisms

## Findings

Long-term memory depends on **synaptic plasticity** and supporting systems.

| Method | Strength |
| --- | --- |
| Optogenetics | Causal manipulation |

Source: [Example](https://example.com/paper)
"""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.docx"
            build_report_docx(markdown, output)

            self.assertTrue(output.is_file())
            with ZipFile(output) as package:
                self.assertIn("word/document.xml", package.namelist())
                document_xml = package.read("word/document.xml").decode("utf-8")
                footer_xml = package.read("word/footer1.xml").decode("utf-8")
                core_xml = package.read("docProps/core.xml").decode("utf-8")
                self.assertIn("Memory mechanisms", document_xml)
                self.assertIn("Optogenetics", document_xml)
                self.assertIn("w:tcMar", document_xml)
                self.assertIn(" PAGE ", footer_xml)
                self.assertIn("Memory mechanisms", core_xml)


if __name__ == "__main__":
    unittest.main()
