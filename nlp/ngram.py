"""
ManualMind N-Gram Language Model Module (M5)
Implements:
1. Unigram, Bigram, and Trigram language models trained from scratch on the manual.
2. Probability estimation with Laplace (add-1) smoothing and Stupid Backoff (alpha = 0.4).
3. Perplexity computation on held-out text.
4. Real-time autocomplete suggestions and next-word generation.
5. Verification of n-gram counts against scikit-learn's CountVectorizer.
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional

from sklearn.feature_extraction.text import CountVectorizer


class ScratchNgramLM:
    """
    N-gram Language Model implemented in pure Python.
    Supports unigram, bigram, and trigram modeling with Laplace smoothing
    and Stupid Backoff for out-of-vocabulary / unseen context handling.
    """
    def __init__(self, backoff_alpha: float = 0.4):
        self.alpha = backoff_alpha
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocab = set()
        self.total_tokens = 0
        self.context_to_words = defaultdict(list)

    def _tokenize(self, text: str) -> List[str]:
        """Lowercases and cleans text into word tokens."""
        return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())

    def train(self, sentences: List[str]):
        """Trains n-gram frequency distributions on a list of sentences or manual paragraphs."""
        self.unigrams.clear()
        self.bigrams.clear()
        self.trigrams.clear()
        self.vocab.clear()
        self.total_tokens = 0
        self.context_to_words.clear()

        for sent in sentences:
            tokens = self._tokenize(sent)
            if not tokens:
                continue

            # Add start and end sentence boundary markers
            padded = ["<s>", "<s>"] + tokens + ["</s>"]
            
            for tok in tokens:
                self.unigrams[tok] += 1
                self.vocab.add(tok)
                self.total_tokens += 1

            for i in range(len(padded) - 1):
                bg = (padded[i], padded[i + 1])
                self.bigrams[bg] += 1

            for i in range(len(padded) - 2):
                tg = (padded[i], padded[i + 1], padded[i + 2])
                self.trigrams[tg] += 1
                # Index for fast autocomplete lookup
                ctx = (padded[i], padded[i + 1])
                if padded[i + 2] not in ["</s>", "<s>"]:
                    self.context_to_words[ctx].append(padded[i + 2])

    def unigram_prob_laplace(self, word: str) -> float:
        """P(w) with Laplace (add-1) smoothing."""
        vocab_size = len(self.vocab) or 1
        count = self.unigrams.get(word.lower(), 0)
        return (count + 1) / (self.total_tokens + vocab_size)

    def bigram_prob_laplace(self, w1: str, w2: str) -> float:
        """P(w2 | w1) with Laplace smoothing."""
        vocab_size = len(self.vocab) or 1
        c_bg = self.bigrams.get((w1.lower(), w2.lower()), 0)
        c_ug = self.unigrams.get(w1.lower(), 0)
        return (c_bg + 1) / (c_ug + vocab_size)

    def trigram_prob_laplace(self, w1: str, w2: str, w3: str) -> float:
        """P(w3 | w1, w2) with Laplace smoothing."""
        vocab_size = len(self.vocab) or 1
        c_tg = self.trigrams.get((w1.lower(), w2.lower(), w3.lower()), 0)
        c_bg = self.bigrams.get((w1.lower(), w2.lower()), 0)
        return (c_tg + 1) / (c_bg + vocab_size)

    def score_stupid_backoff(self, w1: str, w2: str, w3: str) -> float:
        """
        Stupid Backoff score S(w3 | w1, w2):
        If trigram exists: C(w1, w2, w3) / C(w1, w2)
        Else if bigram exists: alpha * C(w2, w3) / C(w2)
        Else: alpha^2 * C(w3) / N
        """
        w1, w2, w3 = w1.lower(), w2.lower(), w3.lower()
        c_tg = self.trigrams.get((w1, w2, w3), 0)
        c_bg_ctx = self.bigrams.get((w1, w2), 0)
        if c_tg > 0 and c_bg_ctx > 0:
            return c_tg / c_bg_ctx

        c_bg = self.bigrams.get((w2, w3), 0)
        c_ug_ctx = self.unigrams.get(w2, 0)
        if c_bg > 0 and c_ug_ctx > 0:
            return self.alpha * (c_bg / c_ug_ctx)

        c_ug = self.unigrams.get(w3, 0)
        return (self.alpha ** 2) * max(1, c_ug) / max(1, self.total_tokens)

    def compute_perplexity(self, test_text: str, n: int = 3) -> float:
        """
        Computes perplexity on held-out text:
        PP = exp( - (1/N) * sum( log P(w_i | context) ) )
        """
        tokens = self._tokenize(test_text)
        if not tokens:
            return float("inf")

        log_prob_sum = 0.0
        m = len(tokens)
        padded = ["<s>", "<s>"] + tokens + ["</s>"]

        if n == 1:
            for tok in tokens:
                p = self.unigram_prob_laplace(tok)
                log_prob_sum += math.log(max(p, 1e-12))
        elif n == 2:
            for i in range(len(padded) - 1):
                p = self.bigram_prob_laplace(padded[i], padded[i + 1])
                log_prob_sum += math.log(max(p, 1e-12))
            m = len(padded) - 1
        else:  # n == 3
            for i in range(len(padded) - 2):
                p = self.trigram_prob_laplace(padded[i], padded[i + 1], padded[i + 2])
                log_prob_sum += math.log(max(p, 1e-12))
            m = len(padded) - 2

        if m == 0:
            return float("inf")

        cross_entropy = - (log_prob_sum / m)
        return round(math.exp(cross_entropy), 2)

    def autocomplete(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Provides live autocomplete predictions for a typed prefix.
        If prefix ends with a partial word, completes it;
        if prefix ends with a space, predicts likely next words.
        """
        raw_prefix = prefix
        ends_with_space = raw_prefix.endswith(" ")
        tokens = self._tokenize(prefix)

        if not tokens:
            # Return most frequent manual keywords
            return [w for w, _ in self.unigrams.most_common(max_suggestions) if w not in ["the", "a", "is", "and"]]

        if not ends_with_space:
            # Partial word completion
            partial = tokens[-1].lower()
            matching = [w for w in self.vocab if w.startswith(partial) and w != partial]
            matching.sort(key=lambda w: self.unigrams.get(w, 0), reverse=True)
            completed = []
            for w in matching[:max_suggestions]:
                prev_text = raw_prefix[:raw_prefix.rfind(tokens[-1])]
                completed.append(prev_text + w)
            return completed

        # Context-based next word prediction
        w2 = tokens[-1].lower()
        w1 = tokens[-2].lower() if len(tokens) >= 2 else "<s>"

        candidates = Counter()
        # Look at trigram contexts
        for (tw1, tw2, tw3), count in self.trigrams.items():
            if tw1 == w1 and tw2 == w2 and tw3 not in ["</s>", "<s>"]:
                candidates[tw3] += count * 4

        # Look at bigram contexts
        for (bw1, bw2), count in self.bigrams.items():
            if bw1 == w2 and bw2 not in ["</s>", "<s>"]:
                candidates[bw2] += count * 2

        # Fallback to frequent unigrams
        for w, count in self.unigrams.most_common(20):
            if w not in ["the", "a", "an", "is", "and", "of", "to"]:
                candidates[w] += 1

        top_words = [word for word, _ in candidates.most_common(max_suggestions)]
        return [raw_prefix.rstrip() + " " + word for word in top_words]

    def verify_against_count_vectorizer(self, sample_texts: List[str], n: int = 2) -> Dict[str, Any]:
        """
        Cross-checks from-scratch n-gram counts against scikit-learn CountVectorizer.
        Returns match status and top n-grams from both implementations.
        """
        # Scikit-learn CountVectorizer
        cv = CountVectorizer(ngram_range=(n, n), token_pattern=r"\b[a-zA-Z0-9_\-]+\b")
        X = cv.fit_transform(sample_texts)
        cv_feature_names = cv.get_feature_names_out()
        cv_counts = X.toarray().sum(axis=0)
        cv_map = dict(zip(cv_feature_names, cv_counts))

        # From-scratch model counts for the same texts
        scratch_lm = ScratchNgramLM()
        scratch_lm.train(sample_texts)

        scratch_map = {}
        if n == 1:
            for w, c in scratch_lm.unigrams.items():
                scratch_map[w] = c
        elif n == 2:
            for (w1, w2), c in scratch_lm.bigrams.items():
                if w1 not in ["<s>", "</s>"] and w2 not in ["<s>", "</s>"]:
                    scratch_map[f"{w1} {w2}"] = c
        elif n == 3:
            for (w1, w2, w3), c in scratch_lm.trigrams.items():
                if w1 not in ["<s>", "</s>"] and w2 not in ["<s>", "</s>"] and w3 not in ["<s>", "</s>"]:
                    scratch_map[f"{w1} {w2} {w3}"] = c

        # Compare common keys
        matched_keys = 0
        total_keys = len(cv_map)
        for k, v in cv_map.items():
            if scratch_map.get(k) == v:
                matched_keys += 1

        accuracy = (matched_keys / total_keys) * 100 if total_keys > 0 else 100.0

        return {
            "order_n": n,
            "sklearn_unique_ngrams": total_keys,
            "scratch_unique_ngrams": len(scratch_map),
            "matching_count_accuracy": round(accuracy, 2),
            "top_sklearn": sorted(cv_map.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_scratch": sorted(scratch_map.items(), key=lambda x: x[1], reverse=True)[:5]
        }
