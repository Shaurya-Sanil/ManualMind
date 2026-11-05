"""
Tests for Module M6: POS Tagging, Chunking & Dependency Parsing
Includes the required hand-worked HMM Viterbi unit test.
"""

import math
import pytest
from nlp.tagging import (
    RuleBasedPOSTagger, ScratchHMMPOSTagger, CRFTaggerWrapper,
    get_all_pos_taggers, tag_sentence_all_models, extract_chunks,
    render_dependency_tree_svg
)


def test_hmm_viterbi_hand_worked_example():
    """
    Unit test of HMM Viterbi decoding on a classic hand-worked pedagogical example:
    States: NOUN, VERB
    Sentence: 'can fish'
    Transitions & Emissions designed such that:
    P(can | VERB) = 0.4, P(can | NOUN) = 0.1
    P(fish | NOUN) = 0.8, P(fish | VERB) = 0.1
    P(NOUN | start) = 0.8, P(VERB | start) = 0.2
    P(NOUN | VERB) = 0.5, P(VERB | NOUN) = 0.9
    
    Hand calculation:
    V[0, NOUN] = 0.8 * 0.1 = 0.08
    V[0, VERB] = 0.2 * 0.4 = 0.08
    V[1, NOUN] = max(0.08*0.1, 0.08*0.5) * 0.8 = 0.040 * 0.8 = 0.032 (prev = VERB)
    V[1, VERB] = max(0.08*0.9, 0.08*0.5) * 0.1 = 0.072 * 0.1 = 0.0072 (prev = NOUN)
    
    Best terminal state: NOUN (0.032 > 0.0072) -> Backtrace gives ['VERB', 'NOUN'].
    """
    hmm = ScratchHMMPOSTagger()
    hmm.tags = ["NOUN", "VERB"]
    hmm.tag2idx = {"NOUN": 0, "VERB": 1}
    hmm.idx2tag = {0: "NOUN", 1: "VERB"}
    hmm.num_tags = 2

    # Set exact log probabilities
    hmm.initial_log_probs = [math.log(0.8), math.log(0.2)]
    hmm.transition_log_probs = [
        [math.log(0.1), math.log(0.9)],  # From NOUN to [NOUN, VERB]
        [math.log(0.5), math.log(0.5)]   # From VERB to [NOUN, VERB]
    ]
    hmm.emission_log_probs = {
        ("NOUN", "can"): math.log(0.1),
        ("VERB", "can"): math.log(0.4),
        ("NOUN", "fish"): math.log(0.8),
        ("VERB", "fish"): math.log(0.1),
    }
    hmm.vocab = {"can", "fish"}

    # Run Viterbi
    decoded = hmm.viterbi(["can", "fish"])
    assert decoded == [("can", "VERB"), ("fish", "NOUN")]


def test_rule_based_pos_tagger():
    tagger = RuleBasedPOSTagger()
    tokens = ["the", "router", "is", "resetting", "quickly"]
    tagged = tagger.tag(tokens)

    tags = [t for _, t in tagged]
    assert tags[0] == "DET"
    assert tags[1] == "NOUN"
    assert tags[2] == "VERB"
    assert tags[3] == "VERB"  # -ing suffix
    assert tags[4] == "ADV"   # -ly suffix


def test_tag_sentence_all_models():
    res = tag_sentence_all_models("Disconnect the power adapter and check the internet light")
    assert "tokens" in res
    assert "rule_based_tags" in res
    assert "hmm_tags" in res
    assert "crf_tags" in res
    assert "spacy_tags" in res

    assert len(res["rule_based_tags"]) == len(res["tokens"])
    assert len(res["hmm_tags"]) == len(res["tokens"])
    assert len(res["spacy_tags"]) == len(res["tokens"])


def test_extract_chunks():
    sentence = "The blue WAN port connects the broadband modem quickly"
    res = extract_chunks(sentence)
    assert "nltk_np_chunks" in res
    assert "nltk_vp_chunks" in res
    assert "spacy_noun_chunks" in res

    # "The blue WAN port" or "broadband modem" should be identified as NP
    assert len(res["nltk_np_chunks"]) >= 1


def test_render_dependency_tree_svg():
    svg = render_dependency_tree_svg("Update the router firmware")
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "ROOT" in svg
    assert "firmware" in svg

