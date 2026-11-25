from labor_rag.retrieval.base import BaseRetriever, RetrievalResult
from labor_rag.retrieval.query_process import (
    ArticleRef,
    QueryAwareRetriever,
    QueryProcessor,
    fold,
    parse_article_refs,
)

CHUNKS = [
    {"chunk_id": "d:article-24", "chunk_level": "article", "article_number": "24",
     "content": "Điều 24. Thử việc. Thời gian thử việc do hai bên thỏa thuận."},
    {"chunk_id": "d:article-25", "chunk_level": "article", "article_number": "25",
     "content": "Điều 25. Thời gian thử việc không quá 60 ngày. Tối đa bao nhiêu ngày."},
    {"chunk_id": "d:article-107-clause-1", "chunk_level": "clause", "article_number": "107", "clause_number": "1",
     "content": "Làm thêm giờ là khoảng thời gian làm việc ngoài thời giờ làm việc bình thường."},
    {"chunk_id": "d:article-107-clause-2", "chunk_level": "clause", "article_number": "107", "clause_number": "2",
     "content": "Làm thêm giờ phải được sự đồng ý của người lao động, không quá 200 giờ."},
]


class StubRetriever(BaseRetriever):
    def __init__(self):
        self.queries = []

    def retrieve(self, query, *, top_k=5):
        self.queries.append(query)
        return [RetrievalResult("d:article-24", "x", 0.1, {})][:top_k]


def test_fold_strips_vietnamese_diacritics():
    assert fold("Điều 35 Đơn phương chấm dứt") == "dieu 35 don phuong cham dut"


def test_parse_article_refs_variants():
    assert parse_article_refs("Điều 35 quy định gì?") == (ArticleRef("35"),)
    assert parse_article_refs("Khoản 2 Điều 107 nói gì") == (ArticleRef("107", "2"),)
    assert parse_article_refs("điều 98 khoản 1") == (ArticleRef("98", "1"),)
    assert parse_article_refs("điều kiện thử việc") == ()


def test_parse_article_refs_deduplicates_and_keeps_clause_form():
    refs = parse_article_refs("Khoản 8 Điều 3 và Điều 3")
    assert ArticleRef("3", "8") in refs


def test_restores_diacritics_from_corpus_ngrams():
    processor = QueryProcessor(c["content"] for c in CHUNKS)
    processed = processor.process("thoi gian thu viec toi da bao nhieu ngay")
    assert processed.diacritics_restored
    assert "thử việc" in processed.search_text


def test_does_not_touch_queries_that_already_have_diacritics():
    processor = QueryProcessor(c["content"] for c in CHUNKS)
    processed = processor.process("Thời gian thử việc tối đa là bao lâu?")
    assert not processed.diacritics_restored


def test_english_query_is_left_alone():
    processor = QueryProcessor(c["content"] for c in CHUNKS)
    processed = processor.process("How many days of annual leave do I get")
    assert not processed.diacritics_restored


def test_expands_abbreviations_and_everyday_terms():
    processor = QueryProcessor(c["content"] for c in CHUNKS)
    processed = processor.process("NLĐ bị đuổi việc thì sao")
    assert "người lao động" in processed.search_text
    assert any("đơn phương chấm dứt" in e for e in processed.expansions)


def test_does_not_expand_term_already_present():
    processor = QueryProcessor(c["content"] for c in CHUNKS)
    processed = processor.process("Làm thêm giờ được trả bao nhiêu")
    assert processed.expansions == ()


def test_thuong_luong_is_not_mistaken_for_bonus():
    processor = QueryProcessor(c["content"] for c in CHUNKS)
    processed = processor.process("Thương lượng tập thể là gì")
    assert not any("thưởng" in e for e in processed.expansions)


def test_retriever_pins_article_and_uses_rewritten_text_for_base():
    base = StubRetriever()
    retriever = QueryAwareRetriever(base, CHUNKS)
    results = retriever.retrieve("Điều 25 quy định gì", top_k=3)
    assert results[0].chunk_id == "d:article-25"
    assert results[0].metadata["retrieval_method"] == "article_lookup"
    assert [r.chunk_id for r in results].count("d:article-24") == 1


def test_retriever_resolves_clause_and_clause_only_articles():
    retriever = QueryAwareRetriever(StubRetriever(), CHUNKS)
    clause = retriever.retrieve("Khoản 2 Điều 107", top_k=3)
    assert clause[0].chunk_id == "d:article-107-clause-2"
    whole = retriever.retrieve("Điều 107", top_k=5)
    assert {r.chunk_id for r in whole[:2]} == {"d:article-107-clause-1", "d:article-107-clause-2"}


def test_unknown_article_returns_nothing_instead_of_lookalikes():
    retriever = QueryAwareRetriever(StubRetriever(), CHUNKS)
    assert retriever.retrieve("Điều 300 quy định gì?", top_k=5) == []


def test_lookup_can_be_disabled():
    retriever = QueryAwareRetriever(StubRetriever(), CHUNKS, article_lookup=False)
    results = retriever.retrieve("Điều 300 quy định gì?", top_k=5)
    assert results and results[0].chunk_id == "d:article-24"


def test_top_k_is_respected():
    retriever = QueryAwareRetriever(StubRetriever(), CHUNKS)
    assert len(retriever.retrieve("Điều 107", top_k=1)) == 1
