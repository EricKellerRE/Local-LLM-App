import unittest

from local_model_app.token_benchmark import (
    LONG_URL,
    score_next_chunk,
    score_notes,
    score_post_tool_summary,
    score_relevance,
    score_section,
    score_section_chunk,
    score_tool_url,
)


def response(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


class TokenBenchmarkScoringTests(unittest.TestCase):
    def test_exact_long_url_tool_call(self):
        value = response(tool_calls=[{"function": {
            "name": "fetch_url", "arguments": {"url": LONG_URL},
        }}])
        self.assertTrue(score_tool_url(value)[0])
        self.assertFalse(score_tool_url(response(tool_calls=[]))[0])

    def test_next_chunk_requires_cursor_and_url(self):
        value = response(tool_calls=[{"function": {
            "name": "fetch_url", "arguments": {"url": LONG_URL, "start_index": 30000},
        }}])
        self.assertTrue(score_next_chunk(value)[0])

    def test_relevance_requires_exact_expected_ids(self):
        self.assertTrue(score_relevance(response(
            '{"keep_numbers":[1,3],"needs_abstract_numbers":[]}'
        ))[0])
        self.assertFalse(score_relevance(response(
            '{"keep_numbers":[2],"needs_abstract_numbers":[]}'
        ))[0])

    def test_notes_require_urls_method_mechanism_and_limitation(self):
        content = (
            "https://example.org/source-a method protein limitation "
            "https://example.org/source-b"
        )
        self.assertTrue(score_notes(response(content))[0])

    def test_post_tool_summary_requires_provenance_and_no_new_call(self):
        content = f"Protein finding and limitation from {LONG_URL}"
        self.assertTrue(score_post_tool_summary(response(content))[0])
        self.assertFalse(score_post_tool_summary(response(content, tool_calls=[{"function": {}}]))[0])

    def test_section_requires_length_sources_and_critical_terms(self):
        content = " ".join(["causal limitation conclusion"] * 200)
        content += " https://example.org/a https://example.org/b"
        self.assertTrue(score_section(response(content))[0])

    def test_section_chunk_uses_hardware_bounded_minimum(self):
        content = " ".join(["causal limitation conclusion"] * 100)
        content += " https://example.org/a https://example.org/b"
        self.assertTrue(score_section_chunk(response(content))[0])


if __name__ == "__main__":
    unittest.main()
