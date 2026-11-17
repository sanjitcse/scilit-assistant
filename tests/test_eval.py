import json
import tempfile
import unittest
from pathlib import Path

from scilit.eval.metrics import answer_correct, citation_scores, percentile, prf
from scilit.eval.run_eval import main
from scilit.types import Answer, Claim


class Metrics(unittest.TestCase):
    item = {"answerable": True, "gold_keywords": ["0.81"], "evidence": ["P001:results"]}

    def test_correctness_requires_keywords_and_non_abstention(self):
        ok = Answer("q", [Claim("IoU of 0.81", ["P001:results"])], False)
        self.assertTrue(answer_correct(ok, self.item))
        self.assertFalse(answer_correct(Answer("q", [], True), self.item))

    def test_abstaining_on_unanswerable_is_correct(self):
        item = {"answerable": False, "gold_keywords": [], "evidence": []}
        self.assertTrue(answer_correct(Answer("q", [], True), item))
        self.assertFalse(answer_correct(Answer("q", [Claim("made up", [])], False), item))

    def test_citation_precision_recall(self):
        a = Answer("q", [Claim("t", ["P001:results", "P002:results"])], False)
        self.assertEqual(citation_scores(a, self.item), (0.5, 1.0))

    def test_percentile_and_prf(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(prf(2, 4, 2), (0.5, 1.0, 2 / 3))


class Harness(unittest.TestCase):
    def test_end_to_end_offline_run_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            main(["--repeats", "2", "--out", d, "--systems", "closed_book", "full"])
            run = next(Path(d).iterdir())
            res = json.loads((run / "results.json").read_text())
            self.assertEqual(res["systems"]["full"]["reproducibility"], 1.0)
            self.assertGreater(res["systems"]["full"]["accuracy"], res["systems"]["closed_book"]["accuracy"])
            self.assertEqual(len(res["manifest"]["data_sha256"]), 3)
            self.assertTrue((run / "traces_full.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
