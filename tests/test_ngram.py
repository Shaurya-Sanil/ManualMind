"""
Tests for Module M5: N-Gram Language Modeling & Autocomplete
"""

import pytest
from nlp.ngram import ScratchNgramLM


def test_ngram_probabilities_and_laplace_smoothing():
    sentences = [
        "reset the router to factory settings",
        "connect the router to the modem",
        "reset the password in settings"
    ]
    lm = ScratchNgramLM()
    lm.train(sentences)

    # Known vocabulary checks
    p_router = lm.unigram_prob_laplace("router")
    assert p_router > 0.0

    # Unseen word should have non-zero probability due to Laplace smoothing
    p_unseen = lm.unigram_prob_laplace("unseenword123")
    assert p_unseen > 0.0
    assert p_router > p_unseen

    # Bigram smoothing
    p_the_router = lm.bigram_prob_laplace("the", "router")
    p_the_unseen = lm.bigram_prob_laplace("the", "unseen")
    assert p_the_router > p_the_unseen

    # Trigram smoothing
    p_reset_the_router = lm.trigram_prob_laplace("reset", "the", "router")
    assert p_reset_the_router > 0.0


def test_stupid_backoff_scores():
    sentences = [
        "reset the router",
        "update the router firmware"
    ]
    lm = ScratchNgramLM(backoff_alpha=0.4)
    lm.train(sentences)

    score_seen = lm.score_stupid_backoff("reset", "the", "router")
    score_unseen = lm.score_stupid_backoff("reset", "the", "unicorn")
    assert score_seen > score_unseen
    assert score_unseen > 0.0


def test_cross_check_counts_against_count_vectorizer():
    sample_corpus = [
        "connect the blue wan cable to your modem",
        "connect the yellow lan cable to your computer",
        "power on the router and modem now"
    ]
    lm = ScratchNgramLM()
    lm.train(sample_corpus)

    # Cross-check bigrams (n=2)
    verif = lm.verify_against_count_vectorizer(sample_corpus, n=2)
    assert verif["matching_count_accuracy"] == 100.0
    assert verif["sklearn_unique_ngrams"] > 0
    assert verif["scratch_unique_ngrams"] == verif["sklearn_unique_ngrams"]


def test_perplexity_calculation():
    sentences = [
        "the router is online with white led indicator",
        "error code E-101 indicates wan link failure",
        "navigate to settings wireless basic settings"
    ]
    lm = ScratchNgramLM()
    lm.train(sentences)

    pp_uni = lm.compute_perplexity("the router is online", n=1)
    pp_bi = lm.compute_perplexity("the router is online", n=2)
    pp_tri = lm.compute_perplexity("the router is online", n=3)

    assert pp_uni > 0 and pp_uni != float("inf")
    assert pp_bi > 0 and pp_bi != float("inf")
    assert pp_tri > 0 and pp_tri != float("inf")


def test_autocomplete_suggestions():
    sentences = [
        "how to reset the router",
        "how to reset the password",
        "how to update firmware",
        "how to configure guest network"
    ]
    lm = ScratchNgramLM()
    lm.train(sentences)

    # Prefix with trailing space: predicts next word
    next_words = lm.autocomplete("how to reset ", max_suggestions=3)
    assert len(next_words) >= 1
    assert any("the" in s for s in next_words)

    # Prefix extending to 'the ': predicts following noun
    full_words = lm.autocomplete("how to reset the ", max_suggestions=3)
    assert any("router" in s or "password" in s for s in full_words)

    # Prefix with partial word: completes word
    completions = lm.autocomplete("how to conf", max_suggestions=3)
    assert any("configure" in c for c in completions)
