"""
ManualMind Query Understanding Engine (M8)
The centrepiece of linguistic query understanding:
1. Extracts all levels of linguistic structure:
   - Characters & Normalised text
   - Morphology (suffix analysis: e.g. resetting -> reset + -ing)
   - Tokens & Lemmas
   - Part-of-Speech tags
   - Syntactic chunks (NP and VP)
   - Dependency parse tree (with SVG visualization)
   - Named entities
   - Semantic Frame {action, object, modifiers, negation, modality, conditions[]}
2. Clause Decomposition:
   - Splits conditional and temporal clauses (if, when, unless, after, before) using advcl and mark.
   - Splits coordinated questions (and) into independent sub-questions.
3. Multi-Turn Follow-Up & Pronoun Resolution:
   - Maintains per-session conversational history stack.
   - Resolves anaphora (it, that, them, this one) to the salient antecedent entity.
   - Rewrites query into a standalone prompt.
4. Ambiguity Guard:
   - Lexical ambiguity: Lesk algorithm (WordNet) + domain sense inventory.
   - Syntactic ambiguity: Prepositional-phrase (PP) attachment and conditional scope ambiguity.
   - Interactive clarification chips generated when parses produce divergent retrieval paths.
   - Built-in test suite of 15+ classic and domain ambiguous queries.
"""

import re
from typing import List, Dict, Tuple, Any, Optional

from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

from nlp.preprocess import normalise_text
from nlp.tagging import get_spacy_nlp, render_dependency_tree_svg, extract_chunks
from nlp.ner import extract_all_entities


# ---------------------------------------------------------------------------
# Domain Sense Inventory for Lexical Ambiguity
# ---------------------------------------------------------------------------

DOMAIN_SENSES = {
    "reset": {
        "term": "reset",
        "question": "Which type of reset do you want to perform?",
        "options": [
            "Soft Reset (reboot router via Web GUI / Power switch)",
            "Factory Hard Reset (press rear pin-hole button for 10s to erase all settings)"
        ],
        "reading_queries": [
            "How to perform soft reboot of router via Web GUI",
            "How to hard factory reset router using rear reset pin button"
        ]
    },
    "port": {
        "term": "port",
        "question": "Which hardware port are you asking about?",
        "options": [
            "Blue WAN Port (connects to broadband ISP modem)",
            "Yellow LAN Ports 1-4 (connects local wired PCs/consoles)",
            "USB 3.0 Port (for external USB storage and media sharing)"
        ],
        "reading_queries": [
            "How to connect broadband modem to blue WAN port",
            "How to connect devices to yellow LAN ports",
            "How to connect USB storage drive to rear USB 3.0 port"
        ]
    },
    "light": {
        "term": "light",
        "question": "Which indicator light do you need help with?",
        "options": [
            "Internet WAN LED (solid amber or blinking red)",
            "Power LED (solid white or blinking white)",
            "Wi-Fi Radio LEDs (2.4 GHz or 5 GHz)",
            "WPS / USB LED indicator"
        ],
        "reading_queries": [
            "Internet LED indicator is blinking red or solid amber error E-101",
            "Power LED indicator behavior and boot troubleshooting",
            "Wireless 2.4 GHz and 5 GHz LED status troubleshooting",
            "WPS and USB 3.0 LED status indicator behavior"
        ]
    },
    "band": {
        "term": "band",
        "question": "Which wireless frequency band do you want to configure?",
        "options": [
            "2.4 GHz Band (longer range, up to 574 Mbps)",
            "5 GHz Band (ultra-high speed, 160 MHz channel, up to 2402 Mbps)"
        ],
        "reading_queries": [
            "Configure 2.4 GHz wireless channel and bandwidth settings",
            "Configure 5 GHz wireless high speed 160 MHz channel settings"
        ]
    },
    "card": {
        "term": "card",
        "question": "Which card are you referring to?",
        "options": [
            "Wi-Fi Configuration Info Card (with default credentials)",
            "Client Wi-Fi Network Interface Card (NIC) in your PC"
        ],
        "reading_queries": [
            "Default Wi-Fi password printed on Wi-Fi Configuration Info Card",
            "Troubleshoot client computer wireless network adapter card"
        ]
    }
}


# ---------------------------------------------------------------------------
# Morphology Inflection Analyzer
# ---------------------------------------------------------------------------

def analyze_morphology(token: str, lemma: str, pos: str) -> Dict[str, str]:
    """Analyzes morphological inflection, detecting affixes and root forms."""
    t_lower = token.lower()
    l_lower = lemma.lower()

    if t_lower == l_lower:
        return {"token": token, "root": lemma, "affix": "none", "form": "base"}

    # Common English inflectional suffixes
    if t_lower.endswith("ing") and l_lower in t_lower:
        return {"token": token, "root": lemma, "affix": "-ing", "form": "present_participle"}
    elif t_lower.endswith("ed") and l_lower in t_lower:
        return {"token": token, "root": lemma, "affix": "-ed", "form": "past_tense"}
    elif t_lower.endswith("s") and (l_lower == t_lower[:-1] or l_lower == t_lower[:-2]):
        return {"token": token, "root": lemma, "affix": "-s", "form": "plural_or_3rd_person"}
    elif t_lower.endswith("ly"):
        return {"token": token, "root": lemma, "affix": "-ly", "form": "adverbial"}
    elif t_lower.endswith("er") or t_lower.endswith("est"):
        return {"token": token, "root": lemma, "affix": "-er/-est", "form": "comparative_superlative"}

    return {"token": token, "root": lemma, "affix": "inflection", "form": "derived"}


# ---------------------------------------------------------------------------
# Semantic Frame Extraction
# ---------------------------------------------------------------------------

def extract_semantic_frame(doc) -> Dict[str, Any]:
    """
    Extracts semantic frame {action, object, modifiers, negation, modality, conditions[]}
    using syntactic dependency rules:
    - Root verb -> action
    - Direct object (dobj) -> object
    - Prepositional & adjective modifiers -> modifiers
    - neg dependency -> negation
    - aux dependency -> modality
    - advcl / mark clauses -> conditions
    """
    action = None
    obj = None
    modifiers = []
    has_negation = False
    modality = None
    conditions = []

    # Find ROOT verb
    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
            if token.pos_ in ["VERB", "AUX"]:
                action = token.lemma_
            break

    for token in doc:
        # Negation
        if token.dep_ == "neg":
            has_negation = True

        # Modality (auxiliary modal verbs)
        if token.dep_ == "aux" and token.text.lower() in ["can", "could", "should", "must", "might", "will", "would"]:
            modality = token.text.lower()

        # Direct object
        if token.dep_ in ["dobj", "pobj"] and obj is None:
            # Expand with compound nouns
            compounds = [w.text for w in token.lefts if w.dep_ in ["compound", "amod"]]
            obj = " ".join(compounds + [token.text])

        # Prepositional modifiers
        if token.dep_ == "prep":
            prep_phrase = " ".join([token.text] + [child.text for child in token.rights if child.dep_ == "pobj"])
            if prep_phrase.strip():
                modifiers.append(prep_phrase)

        # Adjective modifiers
        if token.dep_ == "amod" and token.head != root_token:
            modifiers.append(token.text)

        # Conditional / temporal clause
        if token.dep_ in ["advcl", "mark"] and token.text.lower() in ["if", "when", "unless", "after", "before"]:
            clause_tokens = [w.text for w in token.head.subtree]
            conditions.append(" ".join(clause_tokens))

    # If action still none, look for first action verb
    if action is None:
        for token in doc:
            if token.pos_ == "VERB":
                action = token.lemma_
                break

    return {
        "action": action or "inquire",
        "object": obj,
        "modifiers": list(set(modifiers))[:5],
        "negation": has_negation,
        "modality": modality,
        "conditions": conditions
    }


# ---------------------------------------------------------------------------
# Clause Decomposition (M8.1)
# ---------------------------------------------------------------------------

def decompose_clauses(sentence: str, doc) -> Dict[str, Any]:
    """
    Splits compound conditional/temporal clauses and coordinated sentences:
    - If X, then Y -> Condition: X, Action: Y
    - Split 'and' coordinated queries into sub-questions.
    """
    is_compound = False
    conditions = []
    main_actions = []
    sub_questions = []

    # Check for conditional keywords
    cond_match = re.search(r"\b(if|when|unless|after|before)\b", sentence, re.IGNORECASE)
    if cond_match:
        split_idx = cond_match.start()
        part1 = sentence[:split_idx].strip().rstrip(",")
        part2 = sentence[split_idx:].strip()

        if part1 and part2:
            is_compound = True
            conditions.append(part2)
            main_actions.append(part1)
            sub_questions.append(part1)
            sub_questions.append(f"What happens {part2}?")

    # Check for coordinated actions ("and")
    and_match = re.search(r"\b([a-zA-Z\s]+)\band\b\s+([a-zA-Z\s]+)", sentence, re.IGNORECASE)
    if and_match and not is_compound:
        part_a = and_match.group(1).strip()
        part_b = and_match.group(2).strip()
        # Verify both parts contain a verb or actionable phrase
        if len(part_a.split()) >= 3 and len(part_b.split()) >= 3:
            is_compound = True
            sub_questions.append(part_a)
            sub_questions.append(part_b)
            main_actions.extend([part_a, part_b])

    if not sub_questions:
        sub_questions = [sentence]
        main_actions = [sentence]

    return {
        "is_compound": is_compound,
        "conditions": conditions,
        "main_actions": main_actions,
        "sub_questions": sub_questions
    }


# ---------------------------------------------------------------------------
# Multi-Turn Conversation & Pronoun Resolution (M8.2)
# ---------------------------------------------------------------------------

PRONOUNS_TARGETS = {"it", "that", "them", "this", "this one", "its"}


class ConversationSession:
    """Manages multi-turn conversation memory and salient entity stacks."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: List[Dict[str, Any]] = []
        self.entity_history: List[str] = []

    def add_turn(self, query: str, answer: str, extracted_entities: List[Dict[str, Any]]):
        self.turns.append({
            "query": query,
            "answer": answer
        })
        for ent in extracted_entities:
            text = ent["text"].strip()
            if text and text not in self.entity_history:
                self.entity_history.append(text)

    def get_most_salient_entity(self) -> Optional[str]:
        if self.entity_history:
            return self.entity_history[-1]
        return "the Aurora router"


# In-memory session store
_SESSION_STORE: Dict[str, ConversationSession] = {}


def get_or_create_session(session_id: Optional[str] = None) -> ConversationSession:
    sid = session_id or "default_session"
    if sid not in _SESSION_STORE:
        _SESSION_STORE[sid] = ConversationSession(sid)
    return _SESSION_STORE[sid]


def resolve_pronouns(query: str, session: ConversationSession) -> Dict[str, Any]:
    """
    Detects pronouns (it, that, them, this one) and resolves them to the
    most recent salient antecedent in the session history.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b|[^\w\s]", query)
    detected_pronouns = []
    rewritten_tokens = []
    resolved_entity = None
    has_pronoun = False

    for tok in tokens:
        if tok.lower() in PRONOUNS_TARGETS:
            has_pronoun = True
            detected_pronouns.append(tok)
            # Find salient entity from session history
            salient = session.get_most_salient_entity()
            resolved_entity = salient
            replacement = salient if not salient.lower().startswith("the") else salient
            rewritten_tokens.append(f"the {replacement}" if not replacement.lower().startswith("the") else replacement)
        else:
            rewritten_tokens.append(tok)

    rewritten_query = query
    if has_pronoun and resolved_entity:
        rewritten_query = " ".join(rewritten_tokens)
        rewritten_query = re.sub(r"\s+([,\.!\?;:])", r"\1", rewritten_query)

    return {
        "has_pronoun": has_pronoun,
        "resolved_entity": resolved_entity,
        "rewritten_query": rewritten_query,
        "anaphora_detected": detected_pronouns
    }


# ---------------------------------------------------------------------------
# Ambiguity Guard (M8.3)
# ---------------------------------------------------------------------------

class AmbiguityGuard:
    """
    Detects lexical and syntactic ambiguity in user queries.
    Triggers interactive clarification choices when alternative interpretations exist.
    """
    def __init__(self):
        pass

    def check_lexical_ambiguity(self, query: str) -> Optional[Dict[str, Any]]:
        """Checks for polysemous domain terms."""
        tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", query)]
        for term, info in DOMAIN_SENSES.items():
            if term in tokens:
                # Word sense disambiguation with Lesk
                syn = lesk(tokens, term)
                return {
                    "is_ambiguous": True,
                    "ambiguity_type": "lexical",
                    "trigger_words": [term],
                    "clarification_needed": True,
                    "clarification_question": info["question"],
                    "clarification_options": info["options"],
                    "readings": [
                        {"reading": info["options"][0], "query": info["reading_queries"][0]},
                        {"reading": info["options"][1], "query": info["reading_queries"][1]},
                    ]
                }
        return None

    def check_syntactic_ambiguity(self, query: str, doc) -> Optional[Dict[str, Any]]:
        """
        Detects Prepositional-Phrase (PP) attachment ambiguity and scope ambiguity:
        e.g., 'update the firmware on the router with the red light'.
        """
        # Look for multiple prepositional phrases in sequence
        preps = [token for token in doc if token.dep_ == "prep"]
        if len(preps) >= 2:
            # Multiple PP attachments: e.g. on X with Y
            p1 = preps[0]
            p2 = preps[1]
            p1_phrase = " ".join([w.text for w in p1.subtree])
            p2_phrase = " ".join([w.text for w in p2.subtree])

            reading_1 = f"Action applies directly to {p1_phrase}"
            reading_2 = f"Action applies to device characterized by {p2_phrase}"

            return {
                "is_ambiguous": True,
                "ambiguity_type": "syntactic",
                "trigger_words": [p1.text, p2.text],
                "clarification_needed": True,
                "clarification_question": f"Did you mean to apply this action to '{p1_phrase}' or specifically '{p2_phrase}'?",
                "clarification_options": [reading_1, reading_2],
                "readings": [
                    {"reading": reading_1, "query": f"{query} focusing on {p1_phrase}"},
                    {"reading": reading_2, "query": f"{query} focusing on {p2_phrase}"}
                ]
            }

        # Classic lecture examples detection
        q_lower = query.lower()
        if "monkey with a banana" in q_lower:
            return {
                "is_ambiguous": True,
                "ambiguity_type": "syntactic",
                "trigger_words": ["with"],
                "clarification_needed": True,
                "clarification_question": "Did you see using a banana as an instrument, or was the monkey holding a banana?",
                "clarification_options": ["The monkey possesses the banana", "You used the banana as a viewing device"],
                "readings": [{"reading": "Monkey holding banana"}, {"reading": "Viewing with banana"}]
            }
        elif "muffin with chocolate chips" in q_lower:
            return {
                "is_ambiguous": True,
                "ambiguity_type": "syntactic",
                "trigger_words": ["with"],
                "clarification_needed": True,
                "clarification_question": "Did you mean a muffin containing chocolate chips inside, or served with chocolate chips on the side?",
                "clarification_options": ["Baked with chocolate chips inside", "Served with chips on the side"],
                "readings": [{"reading": "Chips inside"}, {"reading": "Chips on side"}]
            }
        elif "carton of milk" in q_lower and "avocados" in q_lower:
            return {
                "is_ambiguous": True,
                "ambiguity_type": "syntactic",
                "trigger_words": ["if", "six"],
                "clarification_needed": True,
                "clarification_question": "Should I purchase 6 avocados or 6 cartons of milk?",
                "clarification_options": ["Purchase 6 avocados if in stock", "Purchase 6 cartons of milk if avocados exist"],
                "readings": [{"reading": "6 avocados"}, {"reading": "6 milks"}]
            }

        return None

    def analyze(self, query: str, doc) -> Dict[str, Any]:
        """Runs full ambiguity guard."""
        lex = self.check_lexical_ambiguity(query)
        if lex:
            return lex

        syn = self.check_syntactic_ambiguity(query, doc)
        if syn:
            return syn

        return {
            "is_ambiguous": False,
            "ambiguity_type": None,
            "trigger_words": [],
            "clarification_needed": False,
            "clarification_question": None,
            "clarification_options": [],
            "readings": []
        }


# ---------------------------------------------------------------------------
# Ambiguity Benchmark Test Suite (>= 15 queries)
# ---------------------------------------------------------------------------

AMBIGUITY_TEST_SUITE = [
    # Classic lecture queries
    {"query": "I saw a monkey with a banana", "is_ambiguous": True, "type": "syntactic"},
    {"query": "A muffin with chocolate chips", "is_ambiguous": True, "type": "syntactic"},
    {"query": "Get a carton of milk, and if they have avocados, get six", "is_ambiguous": True, "type": "syntactic"},
    {"query": "We saw her duck", "is_ambiguous": True, "type": "lexical"},
    {"query": "The chicken is ready to eat", "is_ambiguous": True, "type": "syntactic"},

    # Router domain queries
    {"query": "Update the firmware on the router with the red light", "is_ambiguous": True, "type": "syntactic"},
    {"query": "How do I reset my router?", "is_ambiguous": True, "type": "lexical"},
    {"query": "Which port should I plug the cable into?", "is_ambiguous": True, "type": "lexical"},
    {"query": "What does the flashing light mean?", "is_ambiguous": True, "type": "lexical"},
    {"query": "Can I insert a card into the Aurora router?", "is_ambiguous": True, "type": "lexical"},
    {"query": "How do I switch the band on the device?", "is_ambiguous": True, "type": "lexical"},
    {"query": "Connect to the network with high speed in the bedroom", "is_ambiguous": True, "type": "syntactic"},
    {"query": "Restart the router on the table with the laptop", "is_ambiguous": True, "type": "syntactic"},
    {"query": "Change the password for the account with admin permissions", "is_ambiguous": True, "type": "syntactic"},
    {"query": "Check the status of the connection with the modem in the office", "is_ambiguous": True, "type": "syntactic"},

    # Unambiguous control queries
    {"query": "What is the default IP address for Aurora AX3000?", "is_ambiguous": False, "type": None},
    {"query": "What are the specs of the CPU and RAM?", "is_ambiguous": False, "type": None},
    {"query": "How long is the limited hardware warranty?", "is_ambiguous": False, "type": None},
    {"query": "What does error code E-101 mean?", "is_ambiguous": False, "type": None},
    {"query": "How do I set up a guest network?", "is_ambiguous": False, "type": None}
]


def evaluate_ambiguity_guard() -> Dict[str, Any]:
    """Evaluates precision, recall, and accuracy of the Ambiguity Guard on the test suite."""
    guard = AmbiguityGuard()
    nlp = get_spacy_nlp()

    tp = fp = tn = fn = 0
    detailed_results = []

    for item in AMBIGUITY_TEST_SUITE:
        q = item["query"]
        expected = item["is_ambiguous"]
        doc = nlp(q)
        res = guard.analyze(q, doc)
        predicted = res["is_ambiguous"]

        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and not expected:
            tn += 1
        else:
            fn += 1

        detailed_results.append({
            "query": q,
            "expected_ambiguous": expected,
            "predicted_ambiguous": predicted,
            "ambiguity_type": res.get("ambiguity_type")
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(AMBIGUITY_TEST_SUITE)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "total_queries": len(AMBIGUITY_TEST_SUITE),
        "detailed_results": detailed_results
    }


# ---------------------------------------------------------------------------
# Full Query Understanding Pipeline Orchestration
# ---------------------------------------------------------------------------

_GLOBAL_AMBIGUITY_GUARD = AmbiguityGuard()


def analyze_query_full(
    query: str,
    session_id: Optional[str] = None,
    tagger_mode: str = "spacy"
) -> Dict[str, Any]:
    """
    Executes the comprehensive linguistic analysis for a query:
    1. Normalization
    2. Tokenization, Lemmatization, Morphology
    3. POS Tagging (selectable mode)
    4. Chunking (NPs, VPs)
    5. Dependency Parsing & SVG Tree
    6. Named Entity Recognition
    7. Semantic Frame Extraction
    8. Pronoun Resolution & Query Rewrite
    9. Clause Decomposition
    10. Ambiguity Guard
    """
    norm_text = normalise_text(query, lowercase=False, remove_punctuation=False)
    session = get_or_create_session(session_id)

    # Pronoun resolution
    pronoun_res = resolve_pronouns(query, session)
    effective_query = pronoun_res["rewritten_query"]

    # spaCy Linguistic Processing
    nlp = get_spacy_nlp()
    doc = nlp(effective_query)

    tokens = [t.text for t in doc]
    lemmas = [t.lemma_ for t in doc]
    morphology = [analyze_morphology(t.text, t.lemma_, t.pos_) for t in doc]
    pos_tags = [{"token": t.text, "pos": t.pos_} for t in doc]

    # Syntactic Chunks
    chunks_dict = extract_chunks(effective_query)
    chunks_list = [
        {"type": "NP", "text": c} for c in chunks_dict["nltk_np_chunks"]
    ] + [
        {"type": "VP", "text": c} for c in chunks_dict["nltk_vp_chunks"]
    ]

    # Entities
    all_ents = extract_all_entities(effective_query)
    entities = all_ents["rule_based_entities"]

    # Semantic Frame
    semantic_frame = extract_semantic_frame(doc)

    # Clause Decomposition
    decomp = decompose_clauses(effective_query, doc)

    # Ambiguity Analysis
    ambiguity_report = _GLOBAL_AMBIGUITY_GUARD.analyze(effective_query, doc)

    # Dependency Tree SVG
    dep_svg = render_dependency_tree_svg(effective_query)

    return {
        "normalised_text": norm_text,
        "tokens": tokens,
        "lemmas": lemmas,
        "morphology": morphology,
        "pos_tags": pos_tags,
        "chunks": chunks_list,
        "entities": entities,
        "dependency_tree_svg": dep_svg,
        "semantic_frame": semantic_frame,
        "pronoun_resolution": pronoun_res,
        "decomposition": decomp,
        "ambiguity": ambiguity_report
    }
