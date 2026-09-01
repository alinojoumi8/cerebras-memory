"""Snippet windowing, reranker passage strategy, and lexical term selection."""

from __future__ import annotations

import re

from store import _STOPWORDS, KnowledgeStore, _query_tokens


LONG = (
    "Overview of the ingest pipeline. " * 20
    + "The reranker threshold is configured in config.json under reranker.max_length. "
    + "Trailing discussion of unrelated matters. " * 20
)


def _contains(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text.casefold()) is not None


def test_snippet_centres_on_the_match_not_the_head_of_the_chunk():
    """The old window anchored on min(find(term)) over every token.

    Short tokens like "i" or "the" matched inside some word within the first few
    characters, so the window collapsed to the head of the chunk and the match
    was never shown.
    """

    query = "how do I configure the reranker threshold"
    snippet = KnowledgeStore._snippet(LONG, query, length=300)

    assert _contains(snippet, "reranker")
    assert _contains(snippet, "threshold")
    assert len(snippet) <= 300 + 2  # plus ellipses
    assert snippet.startswith("…")  # i.e. it did not start at offset 0


def test_snippet_ignores_stopwords_when_choosing_the_window():
    # "the" occurs in the very first sentence; anchoring on it would return the
    # head. The content-bearing term appears far later.
    snippet = KnowledgeStore._snippet(LONG, "the reranker", length=250)
    assert _contains(snippet, "reranker")


def test_snippet_prefers_the_window_covering_the_most_distinct_terms():
    content = (
        "alpha padding padding padding. " * 30
        + "alpha beta gamma together here. "
        + "padding padding padding. " * 30
    )
    snippet = KnowledgeStore._snippet(content, "alpha beta gamma", length=200)
    assert _contains(snippet, "alpha")
    assert _contains(snippet, "beta")
    assert _contains(snippet, "gamma")


def test_snippet_falls_back_to_the_head_when_nothing_matches():
    snippet = KnowledgeStore._snippet(LONG, "completely absent vocabulary", length=200)
    assert snippet.endswith("…")
    assert snippet.startswith("Overview of the ingest pipeline")


def test_short_content_is_returned_whole():
    assert KnowledgeStore._snippet("tiny body", "anything") == "tiny body"


def test_rerank_passage_deliberately_keeps_the_head_biased_strategy():
    """Documented, measured decision - not an oversight.

    Routing the centred window into the cross-encoder moved MRR@8 from 0.7604
    to 0.6386 on the gold set, because the reranker scores one anchor per
    document and the chunk opening is the only place a section heading appears.
    """

    query = "how do I configure the reranker threshold"
    passage = KnowledgeStore._rerank_passage(LONG, query, 300)
    centred = KnowledgeStore._snippet(LONG, query, length=300)

    assert passage != centred
    assert passage.startswith("Overview of the ingest pipeline")


def test_fts_query_drops_stopwords_so_they_cannot_eat_the_candidate_budget():
    built = KnowledgeStore._fts_query("what did we decide about the retry budget")
    assert built is not None
    terms = set(re.findall(r'"([^"]+)"', built))
    assert "retry" in terms and "budget" in terms and "decide" in terms
    assert not terms & {"what", "did", "we", "about", "the"}


def test_fts_query_keeps_stopwords_when_that_is_all_there_is():
    built = KnowledgeStore._fts_query("what is the")
    assert built is not None
    assert set(re.findall(r'"([^"]+)"', built)) == {"what", "is", "the"}


def test_fts_query_sheds_common_terms_first_when_the_cap_bites():
    rare, common = "zzrare", "zzcommon"
    tokens = [f"tok{index}" for index in range(30)]
    query = " ".join([common, *tokens, rare])
    rarity = {common: 50_000, rare: 2}
    rarity.update({token: 1_000 for token in tokens})

    built = KnowledgeStore._fts_query(query, rarity=rarity)
    kept = set(re.findall(r'"([^"]+)"', built))

    assert len(kept) == 24
    assert rare in kept
    assert common not in kept


def test_fts_query_is_none_for_a_query_with_no_tokens():
    assert KnowledgeStore._fts_query("!!! ???") is None


def test_query_tokens_are_unique_and_case_folded():
    assert _query_tokens("Retry RETRY budget") == ["retry", "budget"]
    assert "the" in _STOPWORDS
