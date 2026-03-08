from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arxiv_fetcher import ArxivFetcher


class FakeFeedParser:
    def parse(self, url: str):
        self.last_url = url
        return SimpleNamespace(
            feed=SimpleNamespace(title="arXiv query results"),
            entries=[
                {"id": "oai:arXiv.org:2501.00001", "arxiv_announce_type": "new"},
                {"id": "oai:arXiv.org:2501.00002", "arxiv_announce_type": "replace"},
                {"id": "oai:arXiv.org:2501.00003", "arxiv_announce_type": "new"},
            ],
        )


class FakeArxivResult:
    def __init__(self, entry_id: str, title: str):
        self.entry_id = f"http://arxiv.org/abs/{entry_id}v1"
        self.title = title
        self.summary = f"Abstract for {title}"
        self.authors = [SimpleNamespace(name="Author One"), SimpleNamespace(name="Author Two")]
        self.pdf_url = f"http://arxiv.org/pdf/{entry_id}v1"
        self.categories = ["cs.LG"]
        self.published = datetime(2025, 1, 1)
        self.doi = None


class FakeArxivClient:
    def __init__(self, *args, **kwargs):
        self.seen_batches = []

    def results(self, search):
        self.seen_batches.append(tuple(search.id_list))
        return [FakeArxivResult(item, f"Paper {item}") for item in search.id_list]


class FakeArxivModule:
    class Search:
        def __init__(self, id_list):
            self.id_list = list(id_list)

    def __init__(self):
        self.created_client = None

    def Client(self, *args, **kwargs):
        self.created_client = FakeArxivClient(*args, **kwargs)
        return self.created_client


class ArxivFetcherTest(unittest.TestCase):
    def test_fetch_new_papers_uses_only_new_rss_entries(self) -> None:
        fake_feedparser = FakeFeedParser()
        fake_arxiv = FakeArxivModule()
        fetcher = ArxivFetcher(
            categories=("cs.LG", "cs.AI"),
            max_candidates=10,
            feedparser_module=fake_feedparser,
            arxiv_module=fake_arxiv,
        )

        papers, stats = fetcher.fetch_new_papers()

        self.assertEqual("https://rss.arxiv.org/atom/cs.LG+cs.AI", fake_feedparser.last_url)
        self.assertEqual(2, len(papers))
        self.assertEqual(2, stats.rss_new_count)
        self.assertEqual(2, stats.rss_unique_count)
        self.assertEqual(2, stats.fetched_candidate_count)
        self.assertEqual(("2501.00001", "2501.00003"), fake_arxiv.created_client.seen_batches[0])
        self.assertEqual("2501.00001v1", papers[0].arxiv_id)
        self.assertEqual("2501.00003v1", papers[1].arxiv_id)


if __name__ == "__main__":
    unittest.main()
