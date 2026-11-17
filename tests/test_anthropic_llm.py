"""AnthropicLLM tested against a fake `anthropic` module (no network, no API key)."""
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace as NS

from scilit.types import Chunk


class FakeBadRequest(Exception):
    pass


def install_fake(responses, reject_temperature=False):
    calls = []

    class Messages:
        def create(self, **kw):
            calls.append(kw)
            if reject_temperature and "temperature" in kw:
                raise FakeBadRequest("temperature not supported")
            return NS(content=[NS(type="text", text=responses.pop(0))], usage=NS(input_tokens=10, output_tokens=5))

    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda: NS(messages=Messages())
    mod.BadRequestError = FakeBadRequest
    sys.modules["anthropic"] = mod
    return calls


CHUNK = Chunk("P001:results", "P001", "T", "results", "GlacierNet achieves IoU of 0.81.")


class AnthropicLLMTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("anthropic", None)

    def make(self, responses, cache_dir=None, **kw):
        calls = install_fake(responses, **kw)
        from scilit.llm import AnthropicLLM

        return AnthropicLLM(cache=cache_dir is not None, cache_dir=cache_dir or ".cache/llm"), calls

    def test_parses_fenced_json_and_uses_temperature_zero(self):
        llm, calls = self.make(['```json\n{"abstain": false, "claims": [{"text": "IoU 0.81", "chunk_ids": ["P001:results"]}]}\n```'])
        out = llm.answer("q", [CHUNK])
        self.assertEqual(out["claims"][0]["chunk_ids"], ["P001:results"])
        self.assertEqual(calls[0]["temperature"], 0)
        self.assertIn("[P001:results]", calls[0]["messages"][0]["content"])
        self.assertEqual(llm.usage["input_tokens"], 10)

    def test_unparseable_output_becomes_abstention(self):
        llm, _ = self.make(["I think the answer is 0.81"])
        self.assertTrue(llm.answer("q", [CHUNK])["abstain"])

    def test_cache_prevents_second_call(self):
        with tempfile.TemporaryDirectory() as d:
            llm, calls = self.make(['{"entailed": true}'], cache_dir=d)
            self.assertTrue(llm.entails("claim", "evidence"))
            self.assertTrue(llm.entails("claim", "evidence"))
            self.assertEqual(len(calls), 1)

    def test_retries_without_temperature_when_rejected(self):
        llm, calls = self.make(['{"query": "glacier iou"}'], reject_temperature=True)
        self.assertEqual(llm.rewrite("What IoU?"), "glacier iou")
        self.assertNotIn("temperature", calls[-1])

    def test_no_chunks_means_no_api_call(self):
        llm, calls = self.make([])
        self.assertTrue(llm.answer("q", [])["abstain"])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
