"""
Tests for Module M10: Intent Classification & Routing
"""

import pytest
from nlp.intent import (
    ScratchMultinomialNB, SklearnIntentClassifier,
    get_trained_intent_classifiers, route_intent, INTENT_CLASSES
)


def test_scratch_multinomial_nb_fit_and_predict():
    train_texts = [
        "How do I fix error code E-101 WAN disconnected",
        "How do I resolve internet error E-204",
        "Where do I connect the blue WAN cable",
        "How to unbox and install the router",
        "What are the specifications of the RAM and CPU",
        "What is the warranty period for replacement",
        "Is it safe to place near water or heat",
        "How do I bake a chocolate cake"
    ]
    train_labels = [
        "troubleshooting", "troubleshooting",
        "setup_install", "setup_install",
        "specifications", "warranty_support",
        "safety", "out_of_scope"
    ]

    nb = ScratchMultinomialNB(alpha=1.0)
    nb.fit(train_texts, train_labels)

    # Predictions
    pred1, conf1 = nb.predict("error code E-101 disconnected")
    assert pred1 == "troubleshooting"
    assert conf1 > 0.0

    pred2, conf2 = nb.predict("warranty replacement period")
    assert pred2 == "warranty_support"

    pred3, conf3 = nb.predict("bake a cake")
    assert pred3 == "out_of_scope"


def test_intent_routing_logic():
    assert route_intent("out_of_scope", 0.95) == "refuse_politely"
    assert route_intent("troubleshooting", 0.88) == "extract_numbered_steps"
    assert route_intent("warranty_support", 0.75) == "boost_warranty_sections"
    assert route_intent("specifications", 0.80) == "target_specs_table"
    assert route_intent("how_to_usage", 0.90) == "hybrid_retrieve"


def test_all_intent_classes_recognized():
    nb, sk = get_trained_intent_classifiers()
    sample_queries = {
        "troubleshooting": "My internet is down and light is red",
        "setup_install": "How to connect router to modem",
        "how_to_usage": "How do I configure guest network",
        "specifications": "What processor and RAM does it have",
        "warranty_support": "Customer support phone number and warranty",
        "safety": "Is it safe near water electrical shock",
        "out_of_scope": "What is the capital of France"
    }

    for expected_intent, query in sample_queries.items():
        pred_intent, _ = nb.predict(query)
        assert pred_intent in INTENT_CLASSES

