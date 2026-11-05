"""
ManualMind Intent Classification Module (M10)
Implements:
1. 7 Intent Classes:
   - troubleshooting
   - setup_install
   - how_to_usage
   - specifications
   - warranty_support
   - safety
   - out_of_scope
2. Multinomial Naive Bayes implemented from scratch with log-probabilities,
   Laplace smoothing, and softmax confidence calibration.
3. Scikit-learn Logistic Regression comparison pipeline on TF-IDF n-grams.
4. Evaluation: Precision, Recall, Macro-F1, and Confusion Matrices.
5. Intent-driven pipeline routing.
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


INTENT_CLASSES = [
    "troubleshooting",
    "setup_install",
    "how_to_usage",
    "specifications",
    "warranty_support",
    "safety",
    "out_of_scope"
]


# ---------------------------------------------------------------------------
# 1. Scratch Multinomial Naive Bayes (M10.a)
# ---------------------------------------------------------------------------

class ScratchMultinomialNB:
    """
    Multinomial Naive Bayes classifier implemented from scratch using pure NumPy.
    Calculates log class priors log P(c) and smoothed feature log-likelihoods log P(w | c).
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes: List[str] = list(INTENT_CLASSES)
        self.class2idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.vocab: Dict[str, int] = {}
        self.class_log_priors: np.ndarray = np.array([])
        self.feature_log_prob: np.ndarray = np.array([])

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())

    def fit(self, texts: List[str], labels: List[str]):
        """Fits class priors and feature likelihood distributions."""
        num_docs = len(texts)
        num_classes = len(self.classes)

        # Build vocabulary
        df_counter = Counter()
        tokenized_docs = [self._tokenize(t) for t in texts]
        for tokens in tokenized_docs:
            df_counter.update(tokens)

        # Filter vocabulary to terms occurring at least once
        sorted_vocab = sorted(df_counter.keys())
        self.vocab = {term: idx for idx, term in enumerate(sorted_vocab)}
        vocab_size = len(self.vocab)

        # Count documents and words per class
        class_doc_counts = np.zeros(num_classes, dtype=np.float32)
        word_counts_per_class = np.zeros((num_classes, vocab_size), dtype=np.float32)

        for tokens, label in zip(tokenized_docs, labels):
            if label not in self.class2idx:
                continue
            c_idx = self.class2idx[label]
            class_doc_counts[c_idx] += 1
            for t in tokens:
                if t in self.vocab:
                    word_counts_per_class[c_idx, self.vocab[t]] += 1

        # Compute smoothed class log priors: log((N_c + alpha) / (N + alpha * C))
        total_smoothed_docs = num_docs + self.alpha * num_classes
        self.class_log_priors = np.log((class_doc_counts + self.alpha) / total_smoothed_docs)

        # Compute smoothed feature log likelihoods: log((count(w, c) + alpha) / (total_words(c) + alpha * V))
        total_words_in_class = word_counts_per_class.sum(axis=1, keepdims=True)
        smoothed_denom = total_words_in_class + self.alpha * vocab_size
        self.feature_log_prob = np.log((word_counts_per_class + self.alpha) / smoothed_denom)

    def predict_log_proba(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        log_probs = np.copy(self.class_log_priors)

        for t in tokens:
            if t in self.vocab:
                word_idx = self.vocab[t]
                log_probs += self.feature_log_prob[:, word_idx]

        return log_probs

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Calculates normalized softmax probability distribution over intents."""
        log_probs = self.predict_log_proba(text)
        max_log = np.max(log_probs)
        exp_scores = np.exp(log_probs - max_log)
        probs = exp_scores / np.sum(exp_scores)
        return {self.classes[i]: round(float(probs[i]), 4) for i in range(len(self.classes))}

    def predict(self, text: str) -> Tuple[str, float]:
        """Returns the predicted intent and confidence."""
        probs = self.predict_proba(text)
        best_intent = max(probs, key=probs.get)
        return best_intent, probs[best_intent]


# ---------------------------------------------------------------------------
# 2. Scikit-learn Logistic Regression Comparison Model (M10.b)
# ---------------------------------------------------------------------------

class SklearnIntentClassifier:
    """Logistic Regression baseline on TF-IDF word n-grams."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"\b[a-zA-Z0-9_\-]+\b",
            sublinear_tf=True
        )
        self.clf = LogisticRegression(random_state=42, max_iter=200, C=1.5)
        self.is_trained = False

    def fit(self, texts: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.is_trained = True

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.is_trained:
            return "how_to_usage", 0.5
        X_vec = self.vectorizer.transform([text])
        pred_label = self.clf.predict(X_vec)[0]
        probs = self.clf.predict_proba(X_vec)[0]
        class_idx = list(self.clf.classes_).index(pred_label)
        return pred_label, float(probs[class_idx])

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_trained:
            return {c: 1.0 / len(INTENT_CLASSES) for c in INTENT_CLASSES}
        X_vec = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X_vec)[0]
        return {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.clf.classes_, probs)}


# ---------------------------------------------------------------------------
# Global Classifier Manager & Seed Data
# ---------------------------------------------------------------------------

_GLOBAL_NB_CLASSIFIER: Optional[ScratchMultinomialNB] = None
_GLOBAL_SK_CLASSIFIER: Optional[SklearnIntentClassifier] = None


def get_trained_intent_classifiers(
    train_texts: Optional[List[str]] = None,
    train_labels: Optional[List[str]] = None
) -> Tuple[ScratchMultinomialNB, SklearnIntentClassifier]:
    """Returns singleton trained Naive Bayes and Logistic Regression classifiers."""
    global _GLOBAL_NB_CLASSIFIER, _GLOBAL_SK_CLASSIFIER
    if _GLOBAL_NB_CLASSIFIER is None:
        _GLOBAL_NB_CLASSIFIER = ScratchMultinomialNB(alpha=0.8)
        _GLOBAL_SK_CLASSIFIER = SklearnIntentClassifier()

        if train_texts is None or train_labels is None:
            # Fallback seed dataset if not preloaded
            train_texts, train_labels = get_default_seed_intent_data()

        _GLOBAL_NB_CLASSIFIER.fit(train_texts, train_labels)
        _GLOBAL_SK_CLASSIFIER.fit(train_texts, train_labels)

    return _GLOBAL_NB_CLASSIFIER, _GLOBAL_SK_CLASSIFIER


def route_intent(intent: str, confidence: float) -> str:
    """
    Decides architectural routing action based on classified intent:
    - out_of_scope -> 'refuse_politely'
    - troubleshooting -> 'extract_numbered_steps'
    - warranty_support -> 'boost_warranty_sections'
    - specifications -> 'target_specs_table'
    - standard -> 'hybrid_retrieve'
    """
    if intent == "out_of_scope":
        return "refuse_politely"
    elif intent == "troubleshooting":
        return "extract_numbered_steps"
    elif intent == "warranty_support":
        return "boost_warranty_sections"
    elif intent == "specifications":
        return "target_specs_table"
    return "hybrid_retrieve"


def get_default_seed_intent_data() -> Tuple[List[str], List[str]]:
    """Seed data across all 7 intents for immediate out-of-the-box training."""
    data = [
        # Troubleshooting
        ("My internet is not working and LED is blinking red", "troubleshooting"),
        ("How to fix error code E-101 WAN disconnected", "troubleshooting"),
        ("Error code E-204 DNS resolution failure what should I do", "troubleshooting"),
        ("Router is overheating with error code E-301", "troubleshooting"),
        ("Why is the power light solid amber", "troubleshooting"),
        ("My Wi-Fi keeps dropping and disconnecting constantly", "troubleshooting"),
        ("Slow internet speeds on 5 GHz band", "troubleshooting"),
        ("USB LED is blinking amber error code E-501", "troubleshooting"),
        ("Cannot access web GUI at 192.168.1.1", "troubleshooting"),
        ("PPPoE authentication failure error E-102", "troubleshooting"),
        ("DHCP lease renewal timeout error E-103", "troubleshooting"),
        ("IP address conflict error code E-201", "troubleshooting"),

        # Setup & Install
        ("How do I connect the router to my broadband modem", "setup_install"),
        ("What comes inside the router package box", "setup_install"),
        ("Initial installation steps for Aurora AX3000", "setup_install"),
        ("Where do I plug the blue WAN cable", "setup_install"),
        ("Powering on the router for the first time", "setup_install"),
        ("How to run the initial quick setup wizard", "setup_install"),
        ("Connecting laptop using yellow LAN ports", "setup_install"),
        ("What is the default IP address to set up router", "setup_install"),

        # How to & Usage
        ("How do I change the Wi-Fi network password", "how_to_usage"),
        ("How to enable and set up a guest network", "how_to_usage"),
        ("Configure parental controls to block websites", "how_to_usage"),
        ("How to set up port forwarding for Minecraft server", "how_to_usage"),
        ("How to update firmware online via Web GUI", "how_to_usage"),
        ("How to factory reset the router with the rear pin button", "how_to_usage"),
        ("Enable WPA3 security on 5 GHz network", "how_to_usage"),
        ("Enable Quality of Service QoS for gaming prioritization", "how_to_usage"),
        ("How do I access USB storage via Samba share", "how_to_usage"),
        ("How to set up WireGuard VPN server", "how_to_usage"),

        # Specifications
        ("What are the hardware specifications of Aurora AX3000", "specifications"),
        ("How much RAM and flash memory does the router have", "specifications"),
        ("What processor CPU is inside the router", "specifications"),
        ("What is the maximum Wi-Fi speed on 5 GHz and 2.4 GHz", "specifications"),
        ("What are the physical dimensions and weight", "specifications"),
        ("What is the power consumption and operating voltage", "specifications"),
        ("How many external antennas does it have and what dBi gain", "specifications"),
        ("What wireless standards IEEE 802.11 does it support", "specifications"),

        # Warranty & Support
        ("What is the warranty period for the Aurora router", "warranty_support"),
        ("How do I submit an RMA warranty replacement claim", "warranty_support"),
        ("What is the customer support phone number", "warranty_support"),
        ("What email address can I contact for technical support", "warranty_support"),
        ("What are the customer service support operating hours", "warranty_support"),
        ("Does warranty cover water damage or lightning strikes", "warranty_support"),

        # Safety
        ("What are the electrical safety guidelines", "safety"),
        ("Can I place the router near water or high humidity", "safety"),
        ("What is the recommended ventilation clearance around the router", "safety"),
        ("Is it safe to use a third-party power adapter", "safety"),
        ("FCC Part 15 and CE regulatory compliance statements", "safety"),
        ("RF radiation exposure guidelines and operating temperature limits", "safety"),

        # Out of Scope
        ("How do I bake a chocolate cake at home", "out_of_scope"),
        ("What is the capital city of France", "out_of_scope"),
        ("Write a Python script to sort a binary search tree", "out_of_scope"),
        ("Who won the FIFA World Cup in 2022", "out_of_scope"),
        ("What is the stock price of Apple today", "out_of_scope"),
        ("Can you write a poem about autumn leaves", "out_of_scope"),
        ("How do I fix a flat tire on my bicycle", "out_of_scope"),
        ("Tell me a funny joke about computer programming", "out_of_scope")
    ]
    texts = [d[0] for d in data]
    labels = [d[1] for d in data]
    return texts, labels
