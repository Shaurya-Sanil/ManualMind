"""
ManualMind Spelling Correction & Edit Distance Module (M4)
Implements:
1. Levenshtein edit distance from scratch with dynamic programming matrix
   and optimal backtrace returning the exact operations (match, insert, delete, substitute).
2. Validation against nltk.edit_distance and rapidfuzz.distance.Levenshtein.
3. Spelling corrector against the manual vocabulary with frequency prior P(w).
4. Generates 'Did you mean...' suggestions for the UI.
"""

import re
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional

import nltk
from nltk.metrics import edit_distance as nltk_edit_distance

try:
    import rapidfuzz.distance.Levenshtein as rapidfuzz_lev
except ImportError:
    rapidfuzz_lev = None


def levenshtein_dp(s1: str, s2: str) -> Tuple[int, List[List[int]], List[Dict[str, Any]]]:
    """
    Computes Levenshtein edit distance between strings s1 (source) and s2 (target)
    using dynamic programming.
    
    Returns:
        distance: Minimum edit distance (int)
        dp_matrix: (len(s1)+1) x (len(s2)+1) integer matrix
        operations: List of optimal edit operations ordered from start to end
    """
    m, n = len(s1), len(s2)
    # Initialize DP matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # Match
            else:
                cost_sub = dp[i - 1][j - 1] + 1
                cost_del = dp[i - 1][j] + 1
                cost_ins = dp[i][j - 1] + 1
                dp[i][j] = min(cost_sub, cost_del, cost_ins)

    distance = dp[m][n]

    # Backtrace to recover optimal alignment operations
    i, j = m, n
    rev_ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            rev_ops.append({
                "op": "match",
                "char_src": s1[i - 1],
                "char_tgt": s2[j - 1],
                "cost": 0,
                "cell": [i, j]
            })
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            rev_ops.append({
                "op": "substitute",
                "char_src": s1[i - 1],
                "char_tgt": s2[j - 1],
                "cost": 1,
                "cell": [i, j]
            })
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            rev_ops.append({
                "op": "delete",
                "char_src": s1[i - 1],
                "char_tgt": "",
                "cost": 1,
                "cell": [i, j]
            })
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            rev_ops.append({
                "op": "insert",
                "char_src": "",
                "char_tgt": s2[j - 1],
                "cost": 1,
                "cell": [i, j]
            })
            j -= 1
        else:
            # Fallback if multiple equal paths
            if i > 0 and j > 0:
                i -= 1
                j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1

    rev_ops.reverse()
    return distance, dp, rev_ops


def validate_levenshtein(s1: str, s2: str) -> Dict[str, Any]:
    """Validates the scratch implementation against NLTK and RapidFuzz."""
    dist_scratch, dp_matrix, ops = levenshtein_dp(s1, s2)
    dist_nltk = nltk_edit_distance(s1, s2)
    dist_rf = rapidfuzz_lev.distance(s1, s2) if rapidfuzz_lev else dist_nltk

    return {
        "word1": s1,
        "word2": s2,
        "scratch_distance": dist_scratch,
        "nltk_distance": dist_nltk,
        "rapidfuzz_distance": dist_rf,
        "is_consistent": (dist_scratch == dist_nltk == dist_rf),
        "operations": ops,
        "dp_matrix": dp_matrix
    }


class SpellingCorrector:
    """
    Manual-domain spelling corrector using minimum edit distance with word frequency priors.
    P(c|w) is proportional to P(w) * P(c|w edit distance).
    """
    def __init__(self, vocab_counter: Optional[Counter] = None):
        self.vocab: Counter = vocab_counter or Counter()
        # Seed with common router domain vocabulary
        seed_words = [
            "aurora", "router", "wireless", "network", "password", "default", "gateway",
            "reset", "reboot", "firmware", "update", "internet", "ethernet", "wan", "lan",
            "port", "cable", "modem", "band", "frequency", "channel", "bandwidth",
            "guest", "parental", "controls", "security", "encryption", "wpa2", "wpa3",
            "indicator", "status", "power", "amber", "white", "blinking", "solid",
            "error", "code", "troubleshooting", "disconnect", "connect", "connection",
            "speed", "gigabit", "hardware", "specifications", "warranty", "support"
        ]
        for w in seed_words:
            self.vocab[w] += 10

    def fit_corpus(self, texts: List[str]):
        """Populates vocabulary and token counts from the ingested manual text."""
        for text in texts:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            self.vocab.update(words)

    def correct_word(self, word: str, max_distance: int = 2) -> Tuple[str, int]:
        """
        Finds the most likely intended word in the manual vocabulary.
        Returns (corrected_word, edit_distance).
        """
        w_lower = word.lower()
        if w_lower in self.vocab or len(w_lower) <= 2:
            return word, 0

        # Don't spell-correct digits, error codes, IP addresses
        if re.match(r"^(e-\d+|v\d+|\d+)", w_lower):
            return word, 0

        best_word = word
        min_dist = max_distance + 1
        max_score = -1e9

        total_words = sum(self.vocab.values()) or 1

        for candidate, count in self.vocab.items():
            if abs(len(candidate) - len(w_lower)) > max_distance:
                continue

            dist, _, _ = levenshtein_dp(w_lower, candidate)
            if dist <= max_distance:
                freq_prob = count / total_words
                # Bayesian scoring: prioritize lower distance and higher word prior
                score = (1.0 / (dist + 0.1)) * (freq_prob ** 0.3)
                if score > max_score:
                    max_score = score
                    min_dist = dist
                    best_word = candidate

        if min_dist <= max_distance and best_word.lower() != w_lower:
            # Preserve original capitalization
            if word.isupper():
                return best_word.upper(), min_dist
            elif word[0].isupper():
                return best_word.capitalize(), min_dist
            return best_word, min_dist

        return word, 0

    def correct_query(self, query: str) -> Dict[str, Any]:
        """
        Tokenizes query, spell-corrects unknown tokens, and formats
        a corrected query and 'Did you mean...' suggestion.
        """
        tokens = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b|\b[a-zA-Z]+-\d+\b|\b[a-zA-Z0-9_]+\b|[^\w\s]", query)
        corrected_tokens = []
        corrections_made = []

        for token in tokens:
            if re.match(r"^[a-zA-Z]{3,}$", token):
                corr, dist = self.correct_word(token)
                if dist > 0:
                    corrections_made.append({
                        "original": token,
                        "corrected": corr,
                        "distance": dist
                    })
                    corrected_tokens.append(corr)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)

        corrected_query = ""
        for i, tok in enumerate(corrected_tokens):
            if i > 0 and not re.match(r"^[,\.!\?;:>\)]", tok) and not re.match(r"^[\(<\"]", corrected_tokens[i - 1]):
                corrected_query += " "
            corrected_query += tok

        has_corrections = len(corrections_made) > 0
        did_you_mean = corrected_query if has_corrections else None

        return {
            "original": query,
            "corrected": corrected_query,
            "was_corrected": has_corrections,
            "did_you_mean": did_you_mean,
            "edits": corrections_made
        }
