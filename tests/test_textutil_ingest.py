import unittest

from scilit.ingest import chunk_paper, load_chunks
from scilit.textutil import entity_terms, numbers, sentences, terms
from tests.helpers import CORPUS


class TextUtil(unittest.TestCase):
    def test_terms_drop_stopwords_and_stem(self):
        self.assertEqual(terms("What does GlacierNet achieve?"), ["glaciernet", "achiev"])

    def test_numbers(self):
        self.assertEqual(numbers("IoU of 0.81, compared with 0.74 for U-Net"), ["0.81", "0.74"])

    def test_sentences_keep_decimals_together(self):
        self.assertEqual(len(sentences("It reaches 0.81 on X. Removing Y lowers it.")), 2)

    def test_entity_terms(self):
        self.assertEqual(entity_terms("What F1 score does OrbitalNet reach?"), ["f1", "orbitalnet"])


class Ingest(unittest.TestCase):
    def test_one_chunk_per_section_with_stable_ids(self):
        chunks = load_chunks(CORPUS)
        self.assertEqual(len(chunks), 32)
        self.assertIn("P001:results", {c.chunk_id for c in chunks})

    def test_long_sections_are_split_on_sentence_boundaries(self):
        paper = {"paper_id": "X", "title": "T", "sections": {"body": " ".join(f"Sentence number {i} is here." for i in range(60))}}
        chunks = chunk_paper(paper, max_words=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.text.endswith(".") for c in chunks))
        self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))


if __name__ == "__main__":
    unittest.main()
