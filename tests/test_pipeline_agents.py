import unittest

from scilit.agents import ExtractionAgent, VerificationAgent
from scilit.llm import OfflineLLM
from scilit.pipeline import Pipeline, PipelineConfig, build_index, make_system
from scilit.types import Claim
from tests.helpers import CORPUS


class Pipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = OfflineLLM()
        cls.index = build_index(CORPUS)
        cls.pipe = Pipeline(cls.index, cls.llm, PipelineConfig())

    def test_answers_with_citation(self):
        a = self.pipe.ask("What IoU does GlacierNet achieve on AlpineSat-12k?")
        self.assertFalse(a.abstained)
        self.assertIn("0.81", a.text)
        self.assertEqual(a.cited, {"P001:results"})

    def test_abstains_when_corpus_has_no_evidence(self):
        a = self.pipe.ask("What F1 score does OrbitalNet reach on the SkyMap dataset?")
        self.assertTrue(a.abstained)
        self.assertEqual(a.cited, set())

    def test_timings_recorded(self):
        a = self.pipe.ask("Which dataset was CardioTrace trained on?")
        for k in ("retrieve", "answer", "verify", "total"):
            self.assertIn(k, a.timings)

    def test_same_question_same_answer(self):
        q = "What limitation of ReefWatch do the authors report?"
        a, b = self.pipe.ask(q), self.pipe.ask(q)
        self.assertEqual((a.text, a.retrieved), (b.text, b.retrieved))

    def test_all_named_systems_construct_and_run(self):
        from scilit.pipeline import SYSTEMS

        for name in SYSTEMS:
            self.assertTrue(hasattr(make_system(name, self.index, self.llm).ask("Which dataset was GustCast trained on?"), "text"))


class Verifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = build_index(CORPUS)
        cls.v = VerificationAgent(OfflineLLM())
        cls.chunks = [cls.index.by_id["P001:results"]]

    def test_accepts_supported_claim(self):
        c = Claim("GlacierNet achieves IoU of 0.81, compared with 0.74 for U-Net.", ["P001:results"])
        self.assertTrue(self.v.check([c], self.chunks)[0].ok)

    def test_rejects_wrong_number(self):
        c = Claim("GlacierNet achieves IoU of 0.91, compared with 0.74 for U-Net.", ["P001:results"])
        r = self.v.check([c], self.chunks)[0]
        self.assertFalse(r.ok)
        self.assertIn("0.91", r.reason)

    def test_rejects_missing_or_unretrieved_citation(self):
        self.assertFalse(self.v.check([Claim("x", [])], self.chunks)[0].ok)
        self.assertFalse(self.v.check([Claim("x", ["P999:results"])], self.chunks)[0].ok)


class Extraction(unittest.TestCase):
    def test_fields_are_grounded_in_quotes(self):
        index = build_index(CORPUS)
        out = ExtractionAgent(index, OfflineLLM()).run("P007")
        self.assertEqual(out["value"]["value"], "52.4%")
        self.assertEqual(out["baseline"]["value"], "RetroTemplate")

    def test_hallucinated_fields_are_dropped(self):
        index = build_index(CORPUS)

        class Liar(OfflineLLM):
            def extract(self, paper_id, chunks):
                return {"value": {"value": "99.9", "chunk_id": "P007:results", "quote": "99.9"}}

        self.assertEqual(ExtractionAgent(index, Liar()).run("P007"), {})


if __name__ == "__main__":
    unittest.main()
