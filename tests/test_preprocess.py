"""
Tests for Module M2: Preprocessing, Normalization & Byte Pair Encoding
"""

import pytest
from nlp.preprocess import (
    expand_contractions, normalise_text, ScratchBPE,
    compare_tokenizers, compare_stopwords, compare_stemming_lemmatization
)


def test_expand_contractions():
    assert expand_contractions("won't") == "will not"
    assert expand_contractions("can't") == "cannot"
    assert expand_contractions("it's working") == "it is working"
    assert expand_contractions("they didn't connect") == "they did not connect"


def test_normalise_text():
    raw = "  Resetting  the   Aurora\u00A0AX3000   router's password...  "
    norm = normalise_text(raw, lowercase=True, remove_punctuation=False)
    assert "resetting the aurora ax3000 router is password..." in norm
    assert "  " not in norm


def test_normalise_preserves_technical_symbols():
    text = "Error code E-101 at 192.168.1.1 on Settings > Wireless"
    norm = normalise_text(text, lowercase=False, remove_punctuation=True, preserve_technical_symbols=True)
    assert "E-101" in norm
    assert "192.168.1.1" in norm
    assert ">" in norm


def test_scratch_bpe_training_and_tokenization():
    corpus = [
        "low lower lowest",
        "new newer newest",
        "wide wider widest"
    ]
    bpe = ScratchBPE(num_merges=10)
    bpe.train(corpus)

    assert len(bpe.merges) > 0
    # Learned subwords reconstruct the original word faithfully
    tokens = bpe.tokenize_word("lower")
    assert isinstance(tokens, list)
    assert len(tokens) >= 1
    assert "".join(tokens).replace("</w>", "") == "lower"


def test_compare_tokenizers():
    text = "Reset the Wi-Fi router password at 192.168.1.1."
    res = compare_tokenizers(text)

    assert "char_tokens" in res
    assert "nltk_tokens" in res
    assert "spacy_tokens" in res
    assert "wordpiece_tokens" in res
    assert "bpe_from_scratch_tokens" in res

    assert len(res["char_tokens"]) == len(text)
    assert len(res["nltk_tokens"]) >= 6
    assert len(res["spacy_tokens"]) >= 6


def test_compare_stopwords():
    text = "How do I connect the blue WAN cable to my modem"
    res = compare_stopwords(text)

    assert res["original_token_count"] > res["nltk_token_count"]
    assert "the" in res["removed_stopwords"]
    assert "my" in res["removed_stopwords"]
    assert "wan" not in res["removed_stopwords"]
    assert "cable" not in res["removed_stopwords"]


def test_stemming_vs_lemmatization_reduction():
    text = "connecting connected connections connects resets resetting reset"
    res = compare_stemming_lemmatization(text)

    assert res["vocab_reduction_porter"] > 0
    assert res["vocab_reduction_wordnet"] > 0
    assert len(res["porter_stems"]) == len(res["tokens"])
    assert len(res["wordnet_lemmas"]) == len(res["tokens"])
