from __future__ import annotations

from datetime import datetime
import logging

import feedparser

from models import ArxivFetchStats, CandidatePaper
from utils import chunked, clean_text, extract_arxiv_id


LOGGER = logging.getLogger(__name__)


class ArxivFetcher:
    def __init__(self, categories: tuple[str, ...], max_candidates: int, feedparser_module=feedparser, arxiv_module=None):
        self.categories = categories
        self.max_candidates = max_candidates
        self._feedparser = feedparser_module
        self._arxiv_module = arxiv_module

    def _get_arxiv_module(self):
        if self._arxiv_module is None:
            import arxiv

            self._arxiv_module = arxiv
        return self._arxiv_module

    def fetch_new_papers(self) -> tuple[list[CandidatePaper], ArxivFetchStats]:
        if not self.categories:
            raise ValueError("At least one arXiv category must be configured")

        query = "+".join(self.categories)
        feed_url = f"https://rss.arxiv.org/atom/{query}"
        feed = self._feedparser.parse(feed_url)
        title = getattr(getattr(feed, "feed", None), "title", "")
        if isinstance(title, str) and "Feed error for query" in title:
            raise ValueError(f"Invalid arXiv RSS query: {query}")

        paper_ids = []
        for entry in getattr(feed, "entries", []):
            if entry.get("arxiv_announce_type", "new") != "new":
                continue
            entry_id = clean_text(entry.get("id"))
            if not entry_id:
                continue
            paper_ids.append(entry_id.removeprefix("oai:arXiv.org:"))

        unique_ids = list(dict.fromkeys(paper_ids))[: self.max_candidates]
        LOGGER.info("Fetched %s new arXiv ids from RSS feed", len(unique_ids))
        fetch_stats = ArxivFetchStats(
            rss_new_count=len(paper_ids),
            rss_unique_count=len(unique_ids),
            fetched_candidate_count=0,
        )
        if not unique_ids:
            return [], fetch_stats

        arxiv_module = self._get_arxiv_module()
        client = arxiv_module.Client(num_retries=5, delay_seconds=3)
        candidates: list[CandidatePaper] = []

        for batch in chunked(unique_ids, 20):
            search = arxiv_module.Search(id_list=batch)
            for result in client.results(search):
                candidates.append(self._convert_result(result))

        candidates.sort(
            key=lambda item: item.published or datetime.min,
            reverse=True,
        )
        fetch_stats = ArxivFetchStats(
            rss_new_count=len(paper_ids),
            rss_unique_count=len(unique_ids),
            fetched_candidate_count=len(candidates[: self.max_candidates]),
        )
        return candidates[: self.max_candidates], fetch_stats

    def _convert_result(self, result) -> CandidatePaper:
        title = clean_text(getattr(result, "title", ""))
        abstract = clean_text(getattr(result, "summary", ""))
        authors = tuple(clean_text(author.name) for author in getattr(result, "authors", []) if clean_text(author.name))
        entry_id = clean_text(getattr(result, "entry_id", ""))
        pdf_url = clean_text(getattr(result, "pdf_url", "")) or None
        categories = tuple(getattr(result, "categories", []) or ())
        doi = clean_text(getattr(result, "doi", "")) or None
        published = getattr(result, "published", None)
        arxiv_id = extract_arxiv_id(entry_id, pdf_url)
        if arxiv_id is None:
            arxiv_id = entry_id.rsplit("/", maxsplit=1)[-1]

        return CandidatePaper(
            title=title,
            abstract=abstract,
            authors=authors,
            entry_id=entry_id or arxiv_id,
            pdf_url=pdf_url,
            published=published,
            categories=categories,
            doi=doi,
            arxiv_id=arxiv_id,
        )
