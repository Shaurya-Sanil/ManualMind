"""
ManualMind Retrieval & Text Representation Module (M9)
Implements:
1. Bag-of-Words & TF-IDF Vectorizer from scratch with cosine similarity ranking.
   Tested and verified against scikit-learn TfidfVectorizer.
2. BM25 scoring engine (BM25Okapi formula).
3. Static Word Embeddings: Gensim Word2Vec CBOW and Skip-gram with averaged embeddings.
4. Tiny Skip-gram with Negative Sampling (SGNS) implemented in pure NumPy for Lab exploration.
5. Dense Sentence Embeddings using sentence-transformers/all-MiniLM-L6-v2.
6. Hybrid Retrieval combining BM25 and Dense with Reciprocal Rank Fusion (RRF)
   and exact-match boosts for diagnostic error codes.
7. Query expansion via WordNet synsets and nearest embedding neighbours.
8. Sparse vs Dense vector representations for the NLP Lab.
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn

from nlp.ingest import ManualChunk
from nlp.patterns import find_error_codes


# ---------------------------------------------------------------------------
# 1. From-Scratch TF-IDF Vectorizer (M9.a)
# ---------------------------------------------------------------------------

class ScratchTfidfVectorizer:
    """
    Pure Python/NumPy TF-IDF vectorizer.
    Calculates term frequencies, smoothed inverse document frequency:
    idf(t) = ln((1 + N) / (1 + df(t))) + 1.0,
    applies L2 vector normalization, and computes cosine similarity.
    """
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.idf: np.ndarray = np.array([])
        self.doc_vectors: np.ndarray = np.array([])
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        self.num_docs = len(documents)
        df_counts = Counter()
        doc_tokens_list = []

        for doc in documents:
            tokens = self._tokenize(doc)
            doc_tokens_list.append(tokens)
            unique_terms = set(tokens)
            df_counts.update(unique_terms)

        # Build vocabulary sorted for determinism
        sorted_terms = sorted(df_counts.keys())
        self.vocab = {t: idx for idx, t in enumerate(sorted_terms)}
        self.inv_vocab = {idx: t for t, idx in self.vocab.items()}
        vocab_size = len(self.vocab)

        # Calculate smooth IDF
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self.vocab.items():
            df = df_counts[term]
            self.idf[idx] = math.log((1.0 + self.num_docs) / (1.0 + df)) + 1.0

        # Build TF-IDF document matrix
        self.doc_vectors = np.zeros((self.num_docs, vocab_size), dtype=np.float32)
        for i, tokens in enumerate(doc_tokens_list):
            if not tokens:
                continue
            tf_counter = Counter(tokens)
            for term, count in tf_counter.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    self.doc_vectors[i, idx] = count * self.idf[idx]

            # Row-wise L2 normalization
            norm = np.linalg.norm(self.doc_vectors[i])
            if norm > 0:
                self.doc_vectors[i] /= norm

        return self.doc_vectors

    def transform_query(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        tf_counter = Counter(tokens)

        for term, count in tf_counter.items():
            if term in self.vocab:
                idx = self.vocab[term]
                q_vec[idx] = count * self.idf[idx]

        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm
        return q_vec

    def rank(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Ranks documents using cosine similarity with the query vector."""
        if self.doc_vectors.shape[0] == 0:
            return []
        q_vec = self.transform_query(query)
        scores = np.dot(self.doc_vectors, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


def verify_scratch_tfidf_against_sklearn(documents: List[str], query: str) -> Dict[str, Any]:
    """Compares Scratch TF-IDF ranking against scikit-learn TfidfVectorizer."""
    # Scratch implementation
    scratch_vec = ScratchTfidfVectorizer()
    scratch_vec.fit_transform(documents)
    scratch_ranks = scratch_vec.rank(query, top_k=5)

    # Scikit-learn implementation
    sk_vec = TfidfVectorizer(token_pattern=r"\b[a-zA-Z0-9_\-]+\b", norm='l2', smooth_idf=True)
    sk_matrix = sk_vec.fit_transform(documents)
    sk_q = sk_vec.transform([query])
    sk_scores = (sk_matrix * sk_q.T).toarray().flatten()
    sk_top_indices = np.argsort(sk_scores)[::-1][:5]
    sk_ranks = [(int(idx), float(sk_scores[idx])) for idx in sk_top_indices]

    # Measure rank correlation / top-1 match
    top1_match = (scratch_ranks[0][0] == sk_ranks[0][0]) if scratch_ranks and sk_ranks else False

    return {
        "scratch_ranking": scratch_ranks,
        "sklearn_ranking": sk_ranks,
        "top1_match": top1_match,
        "query": query
    }


# ---------------------------------------------------------------------------
# 2. BM25 Retrieval Engine (M9.b)
# ---------------------------------------------------------------------------

class ScratchBM25:
    """
    BM25Okapi retrieval function:
    score(D, Q) = sum_{q in Q} IDF(q) * (tf(q, D) * (k1 + 1)) / (tf(q, D) + k1 * (1 - b + b * (|D| / avgdl)))
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Counter = Counter()
        self.doc_term_counts: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())

    def fit(self, documents: List[str]):
        self.num_docs = len(documents)
        self.doc_lengths = []
        self.doc_term_counts = []
        self.doc_freqs.clear()

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            term_cnt = Counter(tokens)
            self.doc_term_counts.append(term_cnt)
            for t in term_cnt.keys():
                self.doc_freqs[t] += 1

        self.avgdl = sum(self.doc_lengths) / max(1, self.num_docs)

        # Precompute Robertson-Sparck Jones IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> np.ndarray:
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.num_docs, dtype=np.float32)

        for q in q_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for i in range(self.num_docs):
                tf = self.doc_term_counts[i].get(q, 0)
                if tf > 0:
                    len_norm = 1.0 - self.b + self.b * (self.doc_lengths[i] / self.avgdl)
                    num = tf * (self.k1 + 1.0)
                    denom = tf + self.k1 * len_norm
                    scores[i] += q_idf * (num / denom)

        return scores


# ---------------------------------------------------------------------------
# 3. Tiny NumPy Skip-Gram with Negative Sampling (SGNS) (M9.c)
# ---------------------------------------------------------------------------

class TinyNumpySkipGram:
    """
    Educational NumPy implementation of Word2Vec Skip-gram with Negative Sampling.
    Features 2 embedding matrices (W_in for center words, W_out for context words),
    sigmoid forward pass, negative sampling, and stochastic gradient updates.
    """
    def __init__(self, vocab_size: int = 100, embed_dim: int = 16, lr: float = 0.05):
        np.random.seed(42)
        self.V = vocab_size
        self.D = embed_dim
        self.lr = lr
        self.W_in = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))
        self.W_out = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def train_step(self, center_idx: int, context_idx: int, neg_indices: List[int]) -> float:
        """Runs a single forward and backward gradient descent step for one pair."""
        v_c = self.W_in[center_idx]        # Shape: (D,)
        u_o = self.W_out[context_idx]      # Shape: (D,)
        u_neg = self.W_out[neg_indices]    # Shape: (K, D)

        # Positive pair prediction
        dot_pos = np.dot(u_o, v_c)
        sig_pos = float(self.sigmoid(np.array([dot_pos]))[0])
        loss = -math.log(max(sig_pos, 1e-10))

        # Negative samples prediction
        dot_negs = np.dot(u_neg, v_c)
        sig_negs = self.sigmoid(-dot_negs)
        for s in sig_negs:
            loss -= math.log(max(float(s), 1e-10))

        # Gradients
        grad_v_c = (sig_pos - 1.0) * u_o
        grad_u_o = (sig_pos - 1.0) * v_c

        for k, neg_idx in enumerate(neg_indices):
            s_neg = float(self.sigmoid(np.array([dot_negs[k]]))[0])
            grad_v_c += s_neg * u_neg[k]
            self.W_out[neg_idx] -= self.lr * s_neg * v_c

        # Update weights
        self.W_in[center_idx] -= self.lr * grad_v_c
        self.W_out[context_idx] -= self.lr * grad_u_o

        return float(loss)


# ---------------------------------------------------------------------------
# 4. Gensim Word2Vec Static Word Embeddings
# ---------------------------------------------------------------------------

class DomainWord2Vec:
    """Wrapper for Gensim CBOW and Skip-gram trained on manual chunks."""
    def __init__(self, vector_size: int = 100):
        self.vector_size = vector_size
        self.model_cbow = None
        self.model_sg = None

    def train(self, corpus_texts: List[str]):
        try:
            from gensim.models import Word2Vec
            tokenized = [re.findall(r"\b[a-zA-Z0-9_\-]+\b", t.lower()) for t in corpus_texts]
            tokenized = [t for t in tokenized if len(t) > 2]
            if not tokenized:
                tokenized = [["aurora", "ax3000", "router", "wireless", "settings", "password"]]

            # Train CBOW (sg=0)
            self.model_cbow = Word2Vec(
                tokenized, vector_size=self.vector_size, window=5, min_count=1, sg=0, seed=42, epochs=30
            )
            # Train Skip-gram (sg=1)
            self.model_sg = Word2Vec(
                tokenized, vector_size=self.vector_size, window=5, min_count=1, sg=1, seed=42, epochs=30
            )
        except Exception:
            pass

    def get_nearest_neighbours(self, word: str, top_n: int = 6) -> Dict[str, List[Dict[str, float]]]:
        w_lower = word.lower()
        cbow_res = []
        sg_res = []

        if self.model_cbow and w_lower in self.model_cbow.wv:
            for w, sim in self.model_cbow.wv.most_common(top_n):
                cbow_res.append({"word": w, "similarity": round(float(sim), 4)})
            try:
                for w, sim in self.model_cbow.wv.most_similar(w_lower, topn=top_n):
                    cbow_res.append({"word": w, "similarity": round(float(sim), 4)})
            except Exception:
                pass

        if self.model_sg and w_lower in self.model_sg.wv:
            try:
                for w, sim in self.model_sg.wv.most_similar(w_lower, topn=top_n):
                    sg_res.append({"word": w, "similarity": round(float(sim), 4)})
            except Exception:
                pass

        return {
            "cbow": cbow_res[:top_n],
            "skipgram": sg_res[:top_n]
        }

    def encode_sentence(self, text: str, mode: str = "skipgram") -> np.ndarray:
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
        model = self.model_sg if mode == "skipgram" else self.model_cbow
        if model is None or not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)

        vectors = [model.wv[t] for t in tokens if t in model.wv]
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)

        avg_vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm > 0:
            avg_vec /= norm
        return avg_vec


# ---------------------------------------------------------------------------
# 5. Dense Sentence Embeddings (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

_SENTENCE_TRANSFORMER_MODEL = None


def get_sentence_transformer():
    global _SENTENCE_TRANSFORMER_MODEL
    if _SENTENCE_TRANSFORMER_MODEL is None:
        try:
            import torch
            torch.set_num_threads(1)
            from sentence_transformers import SentenceTransformer
            _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")
            _SENTENCE_TRANSFORMER_MODEL = None
    return _SENTENCE_TRANSFORMER_MODEL


# ---------------------------------------------------------------------------
# 6. Hybrid Retrieval Manager
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Orchestrates multi-method retrieval:
    - Sparse TF-IDF & BM25
    - Static Word Embeddings (CBOW / Skip-Gram averaged vectors)
    - Dense Transformers (all-MiniLM-L6-v2)
    - Reciprocal Rank Fusion (RRF) with Error Code Exact Match Boosting
    """
    def __init__(self):
        self.chunks: List[ManualChunk] = []
        self.doc_texts: List[str] = []
        self.tfidf = ScratchTfidfVectorizer()
        self.bm25 = ScratchBM25()
        self.w2v = DomainWord2Vec()
        self.dense_embeddings: Optional[np.ndarray] = None
        self.w2v_cbow_embeddings: Optional[np.ndarray] = None
        self.w2v_sg_embeddings: Optional[np.ndarray] = None
        self.is_indexed: bool = False

    def index_chunks(self, chunks: List[ManualChunk]):
        """Indexes manual chunks across all retrieval representations."""
        self.chunks = chunks
        self.doc_texts = [f"{c.section_path}\n{c.content}" for c in chunks]

        # 1. Fit TF-IDF & BM25
        self.tfidf.fit_transform(self.doc_texts)
        self.bm25.fit(self.doc_texts)

        # 2. Train Word2Vec on chunks
        self.w2v.train(self.doc_texts)
        self.w2v_cbow_embeddings = np.array([self.w2v.encode_sentence(t, "cbow") for t in self.doc_texts])
        self.w2v_sg_embeddings = np.array([self.w2v.encode_sentence(t, "skipgram") for t in self.doc_texts])

        # 3. Dense sentence embeddings (guarded for low-memory container stability)
        try:
            st_model = get_sentence_transformer()
            if st_model is not None:
                import torch
                import gc
                with torch.no_grad():
                    self.dense_embeddings = st_model.encode(
                        self.doc_texts,
                        batch_size=8,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                gc.collect()
            else:
                self.dense_embeddings = None
        except Exception as e:
            logger.warning(f"Dense sentence encoding skipped: {e}")
            self.dense_embeddings = None

        self.is_indexed = True

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        rrf_k: int = 60
    ) -> List[Tuple[ManualChunk, float, int]]:
        """
        Retrieves top_k chunks using the specified mode:
        'hybrid', 'bm25', 'dense', 'tfidf', 'w2v_cbow', 'w2v_skipgram'.
        Returns list of (ManualChunk, score, rank).
        """
        if not self.is_indexed or not self.chunks:
            return []

        N = len(self.chunks)
        query_error_codes = find_error_codes(query)

        if mode == "tfidf":
            ranked_pairs = self.tfidf.rank(query, top_k=N)
            top_ranked = ranked_pairs[:top_k]
            return [(self.chunks[idx], score, rank + 1) for rank, (idx, score) in enumerate(top_ranked)]

        elif mode == "bm25":
            bm25_scores = self.bm25.score(query)
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            return [(self.chunks[idx], float(bm25_scores[idx]), rank + 1) for rank, idx in enumerate(top_indices)]

        elif mode == "dense":
            st_model = get_sentence_transformer()
            q_emb = st_model.encode([query], normalize_embeddings=True)[0]
            sims = np.dot(self.dense_embeddings, q_emb)
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [(self.chunks[idx], float(sims[idx]), rank + 1) for rank, idx in enumerate(top_indices)]

        elif mode == "w2v_cbow":
            q_vec = self.w2v.encode_sentence(query, "cbow")
            sims = np.dot(self.w2v_cbow_embeddings, q_vec)
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [(self.chunks[idx], float(sims[idx]), rank + 1) for rank, idx in enumerate(top_indices)]

        elif mode == "w2v_skipgram":
            q_vec = self.w2v.encode_sentence(query, "skipgram")
            sims = np.dot(self.w2v_sg_embeddings, q_vec)
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [(self.chunks[idx], float(sims[idx]), rank + 1) for rank, idx in enumerate(top_indices)]

        # Default: Hybrid with Reciprocal Rank Fusion (RRF)
        # 1. BM25 ranks
        bm25_scores = self.bm25.score(query)
        bm25_ranked = np.argsort(bm25_scores)[::-1]
        bm25_rank_dict = {doc_idx: rank for rank, doc_idx in enumerate(bm25_ranked)}

        # 2. Dense ranks (fallback to TF-IDF ranks if dense embeddings unavailable)
        st_model = get_sentence_transformer()
        if st_model is not None and self.dense_embeddings is not None:
            try:
                import torch
                with torch.no_grad():
                    q_emb = st_model.encode([query], normalize_embeddings=True)[0]
                dense_scores = np.dot(self.dense_embeddings, q_emb)
                dense_ranked = np.argsort(dense_scores)[::-1]
                dense_rank_dict = {doc_idx: rank for rank, doc_idx in enumerate(dense_ranked)}
            except Exception:
                tfidf_pairs = self.tfidf.rank(query, top_k=N)
                dense_rank_dict = {doc_idx: rank for rank, (doc_idx, _) in enumerate(tfidf_pairs)}
        else:
            tfidf_pairs = self.tfidf.rank(query, top_k=N)
            dense_rank_dict = {doc_idx: rank for rank, (doc_idx, _) in enumerate(tfidf_pairs)}

        # 3. Reciprocal Rank Fusion
        rrf_scores = np.zeros(N, dtype=np.float32)
        for i in range(N):
            r_bm25 = bm25_rank_dict[i]
            r_dense = dense_rank_dict[i]
            score = (1.0 / (rrf_k + r_bm25 + 1)) + (1.0 / (rrf_k + r_dense + 1))
            
            # Exact Match Boost for Error Codes (e.g. E-204)
            if query_error_codes:
                chunk_text = self.doc_texts[i].upper()
                for code in query_error_codes:
                    if code in chunk_text:
                        score += 0.05  # Significant RRF boost to ensure exact troubleshooting section wins

            rrf_scores[i] = score

        top_indices = np.argsort(rrf_scores)[::-1][:top_k]
        return [(self.chunks[idx], float(rrf_scores[idx]), rank + 1) for rank, idx in enumerate(top_indices)]


# ---------------------------------------------------------------------------
# 7. Sparse vs Dense Representations for the Lab (M9)
# ---------------------------------------------------------------------------

def compare_sparse_vs_dense(sentence: str) -> Dict[str, Any]:
    """
    Visualizes the lecture concept of Sparse vs Dense representations on the same sentence:
    - Sparse: high-dimensional one-hot bag-of-words / n-gram features with exact non-zero entries.
    - Dense: continuous 384-dimensional dense semantic embedding.
    """
    words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", sentence.lower())
    
    # Sparse feature representations
    sparse_features = []
    for i, w in enumerate(words):
        sparse_features.append({
            "feature_name": f"cur_word={w}",
            "value": 1.0,
            "type": "unigram"
        })
        if i > 0:
            sparse_features.append({
                "feature_name": f"bigram={words[i-1]}_{w}",
                "value": 1.0,
                "type": "bigram"
            })

    # Dense embedding representation
    st_model = get_sentence_transformer()
    dense_vec = st_model.encode([sentence], normalize_embeddings=True)[0]
    dense_preview = [round(float(v), 4) for v in dense_vec[:16]]

    return {
        "sentence": sentence,
        "sparse_features": sparse_features[:20],
        "sparse_dimension": 5000,
        "dense_vector_preview": dense_preview,
        "dense_dimension": len(dense_vec)
    }


def expand_query_wordnet(query: str, max_synonyms: int = 2) -> str:
    """Expands query tokens with WordNet synonyms."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    expanded = list(tokens)
    for tok in tokens:
        syns = wn.synsets(tok)
        added = 0
        for s in syns:
            for l in s.lemmas():
                name = l.name().lower().replace("_", " ")
                if name != tok and name not in expanded:
                    expanded.append(name)
                    added += 1
                    if added >= max_synonyms:
                        break
            if added >= max_synonyms:
                break
    return " ".join(expanded)

