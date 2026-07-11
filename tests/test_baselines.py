import math

import pytest

from topocf_rag.baselines import BM25Config, PerQuestionBM25, tokenize_english


def test_tokenizer_nfkc_casefolds_and_uses_english_lexical_tokens() -> None:
    assert tokenize_english("ＡLPHA's co-operate_REL -> Beta42") == (
        "alpha's",
        "co",
        "operate",
        "rel",
        "beta42",
    )


def test_bm25_uses_positive_lucene_idf() -> None:
    index = PerQuestionBM25.fit(
        {"relevant": "alpha alpha", "other": "beta"},
        config=BM25Config(k1=1.5, b=0.0),
    )
    scores = index.score("alpha")
    expected_idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected_score = expected_idf * 2 * 2.5 / (2 + 1.5)
    assert scores["relevant"] == pytest.approx(expected_score)
    assert scores["other"] == 0.0
    assert scores["relevant"] > 0.0


def test_candidate_order_does_not_change_scores() -> None:
    first = PerQuestionBM25.fit({"a": "alpha beta", "b": "beta"})
    second = PerQuestionBM25.fit({"b": "beta", "a": "alpha beta"})
    assert first.score("alpha beta") == second.score("alpha beta")
    assert first.candidate_ids == ("a", "b")


def test_question_local_indexes_are_isolated() -> None:
    question_one = PerQuestionBM25.fit({"p": "alpha", "n": "beta"})
    before = question_one.score("alpha")

    # Terms and document frequencies for another question never enter q1.
    question_two = PerQuestionBM25.fit(
        {"x": "alpha alpha alpha", "y": "alpha gamma", "z": "gamma"}
    )
    assert question_two.score("gamma")["z"] > 0.0
    assert question_one.score("alpha") == before


def test_bag_of_words_topology_reordering_is_an_expected_tie() -> None:
    index = PerQuestionBM25.fit(
        {
            "forward": "doc alpha rel title mention doc beta",
            "reverse": "doc beta rel title mention doc alpha",
        }
    )
    scores = index.score("alpha beta")
    assert scores["forward"] == scores["reverse"]


def test_bm25_rejects_invalid_configuration_and_empty_universe() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        PerQuestionBM25.fit({})
    with pytest.raises(ValueError, match="k1"):
        BM25Config(k1=0)
    with pytest.raises(ValueError, match="between"):
        BM25Config(b=1.1)
