"""Passage selection: each page is cut to its most relevant verbatim chunks, without an LLM call,
and the vectors it computes are reused by the vector index."""

from intelligence_collector.agents.passage_selector import (
    MAX_PASSAGES_PER_PAGE,
    PASSAGE_SEPARATOR,
    PassageSelector,
)
from intelligence_collector.storage.embedding_cache import CachingEmbeddings
from intelligence_collector.storage.evidence_index import chunk_texts
from intelligence_collector.tools.page_content import CHUNK_CHARS

SQLITE = "The SqliteSaver checkpointer writes every checkpoint into a local SQLite file."
REDIS = "The RedisSaver stores checkpoints in Redis for shared deployments."


def _filler(part):
    # Distinct paragraphs: clean_page drops lines repeated three times as boilerplate.
    return " ".join(f"Unrelated background sentence {part}.{n}." for n in range(60))


PAGE = "\n\n".join(["# Persistence", _filler(1), SQLITE, _filler(2), REDIS, _filler(3)])


class KeywordEmbeddings:
    """Deterministic embeddings: one dimension per keyword; counts every call and text."""

    KEYWORDS = ("sqlite", "redis", "background", "checkpoint")

    def __init__(self, fail=False):
        self.fail, self.documents, self.queries = fail, [], []

    def _vector(self, text):
        text = text.casefold()
        return [float(text.count(word)) for word in self.KEYWORDS]

    def embed_documents(self, texts):
        if self.fail:
            raise ConnectionError("embedding provider unavailable")
        self.documents.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return self._vector(text)


def _row(url="https://docs.test/persistence", page=PAGE):
    return {"url": url, "title": "Persistence", "content": "Tavily snippet.", "raw_content": page}


def test_keeps_the_chunks_closest_to_the_query_verbatim_and_in_page_order():
    embeddings = KeywordEmbeddings()
    [row] = PassageSelector(embeddings).read_pages("Where does SQLite store checkpoints?", [_row()])

    passages = row["content"].split(PASSAGE_SEPARATOR)
    assert len(passages) == MAX_PASSAGES_PER_PAGE
    assert all(passage in PAGE for passage in passages)  # verbatim page text, no title header
    assert any(SQLITE in passage for passage in passages)
    assert PAGE.index(passages[0]) < PAGE.index(passages[1])
    assert len(row["content"]) <= MAX_PASSAGES_PER_PAGE * (CHUNK_CHARS + len(PASSAGE_SEPARATOR))
    assert row["snippet"] == "Tavily snippet." and row["raw_content"] == PAGE
    assert embeddings.queries == ["Where does SQLite store checkpoints?"]


def test_one_embedding_call_ranks_every_page_of_a_search_and_short_pages_are_kept_whole():
    embeddings = KeywordEmbeddings()
    short = "# Short page\n\n" + "A complete short page about Redis checkpoints. " * 15
    rows = [
        _row("https://docs.test/a"),
        _row("https://docs.test/short", short),
        _row("https://docs.test/b"),
        {"url": "https://docs.test/none", "content": "snippet only"},
    ]

    read = PassageSelector(embeddings).read_pages("redis", rows)

    assert [row["url"] for row in read] == [row["url"] for row in rows]
    assert len(embeddings.documents) == 1  # all long pages of the search in one call
    assert len(embeddings.documents[0]) == 2 * len(chunk_texts("Persistence", PAGE))
    assert read[1]["content"] == short.strip()  # fits in the passage budget: nothing to rank
    assert read[3]["content"] == "snippet only" and read[3]["raw_content"] == ""
    assert all(REDIS in read[i]["content"] for i in (0, 2))


def test_without_embeddings_or_when_embedding_fails_word_overlap_ranks_the_chunks():
    for selector in (PassageSelector(), PassageSelector(KeywordEmbeddings(fail=True))):
        [row] = selector.read_pages("SqliteSaver SQLite file", [_row()])
        assert SQLITE in row["content"]


def test_the_index_reuses_the_vectors_selection_computed():
    inner = KeywordEmbeddings()
    cached = CachingEmbeddings(inner)
    PassageSelector(cached).read_pages("sqlite", [_row()])
    texts = chunk_texts("Persistence", PAGE)

    assert cached.embed_documents(texts) == [inner._vector(text) for text in texts]
    assert len(inner.documents) == 1  # indexing embedded nothing new
    cached.embed_documents(texts)  # a vector is handed out once, then embedded again
    assert len(inner.documents) == 2


def test_the_cache_is_bounded():
    cached = CachingEmbeddings(KeywordEmbeddings(), max_entries=2)
    cached.remember(["a", "b", "c"], [[1.0], [2.0], [3.0]])
    assert cached.embed_documents(["b", "c"]) == [[2.0], [3.0]]
    assert cached.inner.documents == []
