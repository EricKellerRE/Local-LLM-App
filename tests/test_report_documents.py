import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from local_model_app.report_documents import (
    assemble_research_report,
    build_report_docx,
    report_word_count,
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
        self.assertEqual(unique_urls(report), ["https://example.com/a", "https://example.com/b"])
        self.assertGreaterEqual(report_word_count(report), 8)

    def test_url_extraction_preserves_balanced_parentheses(self) -> None:
        text = "[Paper](https://example.com/article/A(14)B)"

        self.assertEqual(unique_urls(text), ["https://example.com/article/A(14)B"])

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
                self.assertIn("Memory mechanisms", document_xml)
                self.assertIn("Optogenetics", document_xml)


if __name__ == "__main__":
    unittest.main()
