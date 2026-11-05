"""
Unit Tests for Query Understanding Engine (M8)
Tests linguistic levels, semantic frame, clause decomposition,
pronoun resolution, and lexical & syntactic Ambiguity Guard.
"""

import pytest
from nlp.tagging import get_spacy_nlp
from nlp.understanding import (
    analyze_morphology,
    extract_semantic_frame,
    decompose_clauses,
    ConversationSession,
    resolve_pronouns,
    AmbiguityGuard,
    evaluate_ambiguity_guard,
    analyze_query_full,
    DOMAIN_SENSES
)


def test_morphology_analysis():
    """Verify morphology analyzer detects inflections and affixes correctly."""
    m_ing = analyze_morphology("resetting", "reset", "VERB")
    assert m_ing["affix"] == "-ing"
    assert m_ing["form"] == "present_participle"

    m_ed = analyze_morphology("connected", "connect", "VERB")
    assert m_ed["affix"] == "-ed"
    assert m_ed["form"] == "past_tense"

    m_s = analyze_morphology("ports", "port", "NOUN")
    assert m_s["affix"] == "-s"
    assert m_s["form"] == "plural_or_3rd_person"

    m_base = analyze_morphology("router", "router", "NOUN")
    assert m_base["affix"] == "none"
    assert m_base["form"] == "base"


def test_semantic_frame_extraction():
    """Verify extraction of action, object, modality, negation, and modifiers."""
    nlp = get_spacy_nlp()
    doc = nlp("You should not reboot the router in the office if the LED is red.")
    frame = extract_semantic_frame(doc)

    assert frame["action"] in ["reboot", "rebooting"]
    assert "router" in frame["object"].lower()
    assert frame["negation"] is True
    assert frame["modality"] == "should"
    assert any("office" in mod for mod in frame["modifiers"])
    assert len(frame["conditions"]) > 0


def test_clause_decomposition_conditional():
    """Verify splitting conditional clauses into sub-questions."""
    nlp = get_spacy_nlp()
    query = "Set up a guest network if more than five devices are connected."
    doc = nlp(query)
    decomp = decompose_clauses(query, doc)

    assert decomp["is_compound"] is True
    assert len(decomp["conditions"]) >= 1
    assert any("if" in c.lower() for c in decomp["conditions"])
    assert len(decomp["sub_questions"]) >= 2
    assert "Set up a guest network" in decomp["sub_questions"][0]


def test_clause_decomposition_coordinated():
    """Verify splitting 'and' coordinated queries into two sub-questions."""
    nlp = get_spacy_nlp()
    query = "Configure the guest network and change the administrator password."
    doc = nlp(query)
    decomp = decompose_clauses(query, doc)

    assert decomp["is_compound"] is True
    assert len(decomp["sub_questions"]) == 2
    assert "guest network" in decomp["sub_questions"][0].lower()
    assert "administrator password" in decomp["sub_questions"][1].lower()


def test_multi_turn_pronoun_resolution():
    """Verify anaphora resolution (it, that, them) across session history."""
    session = ConversationSession("test_session_123")
    session.add_turn(
        query="Where is the WAN port located?",
        answer="The blue Gigabit WAN port is on the rear panel.",
        extracted_entities=[{"text": "WAN port", "label": "COMPONENT"}]
    )

    res = resolve_pronouns("How do I plug a cable into it?", session)
    assert res["has_pronoun"] is True
    assert "it" in res["anaphora_detected"]
    assert res["resolved_entity"] == "WAN port"
    assert "WAN port" in res["rewritten_query"]
    assert "it" not in res["rewritten_query"].split()


def test_ambiguity_guard_lexical():
    """Verify lexical ambiguity triggers clarification options for domain homonyms."""
    guard = AmbiguityGuard()
    res = guard.check_lexical_ambiguity("How do I reset the device?")
    assert res is not None
    assert res["is_ambiguous"] is True
    assert res["ambiguity_type"] == "lexical"
    assert "reset" in res["trigger_words"]
    assert len(res["clarification_options"]) >= 2
    assert "Soft Reset" in res["clarification_options"][0]

    # Port ambiguity
    res_port = guard.check_lexical_ambiguity("Which port should I connect?")
    assert res_port is not None
    assert "port" in res_port["trigger_words"]
    assert any("WAN" in opt for opt in res_port["clarification_options"])


def test_ambiguity_guard_syntactic():
    """Verify PP-attachment ambiguity detection on preposition chains."""
    nlp = get_spacy_nlp()
    guard = AmbiguityGuard()

    # Domain PP-attachment
    q = "Update the firmware on the router with the red light"
    doc = nlp(q)
    res = guard.check_syntactic_ambiguity(q, doc)
    assert res is not None
    assert res["is_ambiguous"] is True
    assert res["ambiguity_type"] == "syntactic"
    assert len(res["clarification_options"]) == 2

    # Unambiguous baseline
    q_clear = "What is the default IP address?"
    doc_clear = nlp(q_clear)
    res_clear = guard.analyze(q_clear, doc_clear)
    assert res_clear["is_ambiguous"] is False


def test_ambiguity_benchmark_suite():
    """Verify built-in Ambiguity Test Suite meets benchmark standards."""
    eval_res = evaluate_ambiguity_guard()
    assert eval_res["total_queries"] >= 15
    assert eval_res["precision"] >= 0.80
    assert eval_res["recall"] >= 0.80
    assert eval_res["f1"] >= 0.80


def test_analyze_query_full_integration():
    """Verify the full understanding pipeline returns all linguistic levels."""
    analysis = analyze_query_full("How do I configure Wi-Fi?", session_id="full_test_sid")
    assert "normalised_text" in analysis
    assert "tokens" in analysis
    assert "lemmas" in analysis
    assert "morphology" in analysis
    assert "pos_tags" in analysis
    assert "chunks" in analysis
    assert "entities" in analysis
    assert "dependency_tree_svg" in analysis
    assert "semantic_frame" in analysis
    assert "pronoun_resolution" in analysis
    assert "decomposition" in analysis
    assert "ambiguity" in analysis

    assert len(analysis["tokens"]) > 0
    assert "<svg" in analysis["dependency_tree_svg"]
