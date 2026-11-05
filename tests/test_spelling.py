"""
Tests for Module M4: Spelling Correction & Levenshtein Edit Distance
"""

import pytest
import nltk
from nltk.metrics import edit_distance as nltk_dist

try:
    import rapidfuzz.distance.Levenshtein as rf_dist
except ImportError:
    rf_dist = None

from nlp.spelling import levenshtein_dp, validate_levenshtein, SpellingCorrector


@pytest.mark.parametrize("s1,s2", [
    ("kitten", "sitting"),
    ("router", "routter"),
    ("password", "pasword"),
    ("wireless", "wireles"),
    ("connect", "conncet"),
    ("gateway", "getaway"),
    ("default", "defalt"),
    ("firmware", "framware"),
    ("antenna", "antena"),
    ("internet", "internt"),
    ("", "hello"),
    ("world", ""),
    ("same", "same"),
    ("a", "b"),
    ("execution", "intention")
])
def test_levenshtein_against_nltk_and_rapidfuzz(s1, s2):
    scratch_d, dp_matrix, ops = levenshtein_dp(s1, s2)
    expected_nltk = nltk_dist(s1, s2)

    # 1. Test against NLTK
    assert scratch_d == expected_nltk

    # 2. Test against RapidFuzz if available
    if rf_dist:
        assert scratch_d == rf_dist.distance(s1, s2)

    # 3. Test matrix dimensions
    assert len(dp_matrix) == len(s1) + 1
    assert len(dp_matrix[0]) == len(s2) + 1
    assert dp_matrix[len(s1)][len(s2)] == scratch_d

    # 4. Verify backtrace operation costs sum to distance
    total_cost = sum(op["cost"] for op in ops)
    assert total_cost == scratch_d


def test_validate_levenshtein_helper():
    res = validate_levenshtein("routter", "router")
    assert res["is_consistent"] is True
    assert res["scratch_distance"] == 1
    assert res["nltk_distance"] == 1


def test_spelling_corrector():
    corrector = SpellingCorrector()
    corrector.fit_corpus([
        "Aurora router default password internet connection gateway reset reboot firmware"
    ])

    # Single word correction
    word, dist = corrector.correct_word("routter")
    assert word == "router"
    assert dist == 1

    word, dist = corrector.correct_word("pasword")
    assert word == "password"
    assert dist == 1

    # Query correction with Did You Mean
    res = corrector.correct_query("How do I chnage the defalt admin pasword for my routter?")
    assert res["was_corrected"] is True
    assert "password" in res["corrected"].lower()
    assert "router" in res["corrected"].lower()
    assert res["did_you_mean"] is not None


def test_spelling_corrector_preserves_error_codes():
    corrector = SpellingCorrector()
    # Error codes and IP addresses must not be altered
    res = corrector.correct_query("Fix error E-101 at 192.168.1.1")
    assert "E-101" in res["corrected"]
    assert "192.168.1.1" in res["corrected"]

