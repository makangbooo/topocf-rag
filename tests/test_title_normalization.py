import pytest

from topocf_rag.title_normalization import (
    contains_title_mention,
    mentioning_sentence_indices,
    normalize_title,
)


def test_normalize_title_is_conservative_and_deterministic() -> None:
    assert normalize_title("  Alpha_Beta  ") == "alpha beta"
    assert normalize_title("Ａｌｐｈａ\tBeta") == "alpha beta"
    assert normalize_title("Work (film) - Part II") == "work (film) - part ii"


def test_normalize_title_rejects_invalid_values() -> None:
    with pytest.raises(TypeError):
        normalize_title(7)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        normalize_title(" \t ")


def test_title_mention_is_casefolded_and_boundary_safe() -> None:
    assert contains_title_mention("ALPHA beta was cited.", "Alpha_Beta")
    assert contains_title_mention("An account of York's history.", "York")
    assert not contains_title_mention("A village in Yorkshire.", "York")
    assert not contains_title_mention("Version alpha2 was released.", "Alpha")


def test_title_mention_preserves_punctuation() -> None:
    assert contains_title_mention("The subject is Work (film).", "Work (film)")
    assert not contains_title_mention("The subject is Work film.", "Work (film)")


def test_mentioning_sentence_indices_preserves_sentence_positions() -> None:
    sentences = ["No reference here.", "Target title appears.", "TARGET_TITLE again."]
    assert mentioning_sentence_indices(sentences, "Target Title") == (1, 2)


def test_mentioning_sentence_indices_rejects_non_string_sentence() -> None:
    with pytest.raises(TypeError):
        mentioning_sentence_indices(["valid", None], "Title")  # type: ignore[list-item]
