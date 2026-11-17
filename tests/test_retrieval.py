import unittest

from scilit.pipeline import build_index
from scilit.retrieval import LexicalReranker, rrf_merge
from tests.helpers import CORPUS, load_qa


class Retrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = build_index(CORPUS)
        cls.qa = [q for q in load_qa() if q["answerable"]]

    def test_bm25_finds_gold_chunk_in_top5_for_every_answerable_question(self):
        for q in self.qa:
            ids = [c.chunk_id for c, _ in self.index.search(q["question"], 5, "bm25")]
            self.assertTrue(set(q["evidence"]) <= set(ids), q["question"])

    def test_hybrid_is_deterministic(self):
        a = self.index.search("wind gust nowcasting limitation", 5)
        b = self.index.search("wind gust nowcasting limitation", 5)
        self.assertEqual([c.chunk_id for c, _ in a], [c.chunk_id for c, _ in b])

    def test_rrf_rewards_agreement(self):
        c = self.index.chunks
        merged = rrf_merge([[(c[0], 1), (c[1], 1)], [(c[1], 1), (c[2], 1)]])
        self.assertEqual(merged[0][0].chunk_id, c[1].chunk_id)

    def test_reranker_improves_top1_over_dense_on_task_described_questions(self):
        rr = LexicalReranker(self.index)
        qs = [q for q in self.qa if q["type"].endswith("_by_task")]

        def top1(q, rerank):
            hits = self.index.search(q["question"], 20, "dense")
            hits = rr.rerank(q["question"], hits, 1) if rerank else hits[:1]
            return hits[0][0].chunk_id in q["evidence"]

        self.assertGreaterEqual(sum(top1(q, True) for q in qs), sum(top1(q, False) for q in qs))


if __name__ == "__main__":
    unittest.main()
