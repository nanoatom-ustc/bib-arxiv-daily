from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import CandidatePaper, LibraryPaper
from recommender import Recommender


class FakeEmbedder:
    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.vectors[text] for text in texts], axis=0)


class RecommenderTest(unittest.TestCase):
    def test_recommend_scores_by_top_neighbors_and_filters_existing_identity(self) -> None:
        library_papers = [
            LibraryPaper(
                title="Graph Neural Networks",
                abstract="Message passing networks for graphs.",
                source_file="data/lib.bib",
                doi="10.1000/gnn",
            ),
            LibraryPaper(
                title="Diffusion Models",
                abstract="Generative diffusion on images.",
                source_file="data/lib.bib",
            ),
        ]
        candidate_papers = [
            CandidatePaper(
                title="Graph Signal Learning",
                abstract="Learning on graph-structured data.",
                authors=("A. Author",),
                entry_id="http://arxiv.org/abs/2501.00001v1",
                pdf_url="http://arxiv.org/pdf/2501.00001v1",
                published=datetime(2025, 1, 1),
                arxiv_id="2501.00001v1",
            ),
            CandidatePaper(
                title="Graph Neural Networks",
                abstract="A duplicate candidate already present in the bib corpus.",
                authors=("B. Author",),
                entry_id="http://arxiv.org/abs/2501.00002v1",
                pdf_url="http://arxiv.org/pdf/2501.00002v1",
                published=datetime(2025, 1, 2),
                doi="10.1000/gnn",
            ),
        ]

        vectors = {
            library_papers[0].embedding_text: np.array([1.0, 0.0]),
            library_papers[1].embedding_text: np.array([0.0, 1.0]),
            candidate_papers[0].embedding_text: np.array([0.8, 0.2]),
        }
        recommender = Recommender(
            embedder=FakeEmbedder(vectors),
            top_k_neighbors=2,
            max_results=5,
        )

        recommendations, stats = recommender.recommend(library_papers, candidate_papers)

        self.assertEqual(1, len(recommendations))
        self.assertEqual(2, stats.input_candidate_count)
        self.assertEqual(1, stats.after_dedup_filter_count)
        self.assertEqual(0, stats.threshold_filtered_count)
        self.assertEqual(1, stats.final_recommendation_count)
        self.assertEqual("Graph Signal Learning", recommendations[0].candidate.title)
        self.assertAlmostEqual(0.6063, recommendations[0].score, places=4)
        self.assertEqual("Graph Neural Networks", recommendations[0].neighbors[0].title)


if __name__ == "__main__":
    unittest.main()
