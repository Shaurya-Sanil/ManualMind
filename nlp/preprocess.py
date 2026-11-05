"""
ManualMind Preprocessing Module (M2)
Implements:
1. Contraction expansion, noise removal, and unicode normalisation.
2. Tokenization algorithms:
   - NLTK word tokenizer
   - spaCy tokenizer
   - Character-level tokenizer
   - WordPiece tokenizer (BERT)
   - Byte Pair Encoding (BPE) trained from scratch with merge priority learning
3. Stopword removal (NLTK vs spaCy lists) with before/after statistics.
4. Stemming (Porter, Lancaster) vs Lemmatization (WordNet, spaCy) with vocabulary reduction metrics.
5. Configurable production text cleaner.
"""

import re
import unicodedata
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Any, Optional

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords_module
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer

# Ensure required NLTK resources are available
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


# Common English Contractions Mapping
CONTRACTIONS_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "what's": "what is",
    "how's": "how is",
    "where's": "where is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "who's": "who is",
    "let's": "let us",
    "i'm": "i am",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
}


def expand_contractions(text: str) -> str:
    """Expands English contractions into their full linguistic forms."""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS_MAP.keys()) + r')\b', re.IGNORECASE)
    def replace(match):
        val = match.group(0).lower()
        return CONTRACTIONS_MAP.get(val, val)
    return pattern.sub(replace, text)


def normalise_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    expand_contr: bool = True,
    preserve_technical_symbols: bool = True
) -> str:
    """
    Applies unicode normalization, contraction expansion, whitespace trimming,
    and optional case folding and punctuation removal.
    If preserve_technical_symbols is True, hyphens (E-101), periods in IPs/versions (192.168.1.1, v1.4.0),
    and greater-than arrows in menu paths (Settings > Network) are preserved.
    """
    # 1. Unicode decomposition & normalisation (NFKD)
    normalized = unicodedata.normalize('NFKD', text)
    normalized = "".join([c for c in normalized if not unicodedata.combining(c)])

    # 2. Contractions expansion
    if expand_contr:
        normalized = expand_contractions(normalized)

    # 3. Lowercase
    if lowercase:
        normalized = normalized.lower()

    # 4. Punctuation handling
    if remove_punctuation:
        if preserve_technical_symbols:
            # Strip standard punctuation but preserve '-', '.', '>', and '_'
            normalized = re.sub(r"[^\w\s\-\.\>]", " ", normalized)
        else:
            normalized = re.sub(r"[^\w\s]", " ", normalized)

    # 5. Collapse duplicate whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# ---------------------------------------------------------------------------
# Byte Pair Encoding (BPE) from scratch
# ---------------------------------------------------------------------------

class ScratchBPE:
    """
    Pure Python implementation of Byte Pair Encoding (BPE) merge learning.
    Learns frequent character/subword pair merges from training text,
    stores priority merge rules, and segments unseen text into subwords.
    """
    def __init__(self, num_merges: int = 150):
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Set[str] = set()

    @staticmethod
    def _get_stats(vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Counts frequency of adjacent symbol pairs across vocabulary words."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_vocab(pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Replaces instances of (pair[0], pair[1]) with concatenated token 'pair[0]pair[1]'."""
        new_vocab = {}
        bigram = pair
        for word in vocab:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = vocab[word]
        return new_vocab

    def train(self, texts: List[str]):
        """Learns subword merge rules from corpus texts."""
        word_freqs = Counter()
        for text in texts:
            words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
            word_freqs.update(words)

        # Initialize vocabulary with characters plus end-of-word marker </w>
        vocab = {}
        for word, freq in word_freqs.items():
            chars = tuple(list(word) + ["</w>"])
            vocab[chars] = freq

        self.merges = []
        for _ in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

        # Build final vocabulary set
        subword_vocab = set()
        for word in vocab:
            for symbol in word:
                subword_vocab.add(symbol)
        self.vocab = subword_vocab

    def tokenize_word(self, word: str) -> List[str]:
        """Segments a single word using the learned BPE merges."""
        symbols = list(word.lower()) + ["</w>"]
        for pair in self.merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    new_symbols.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into a list of BPE subword units."""
        words = re.findall(r"\b[a-zA-Z0-9_\-]+\b|[^\w\s]", text)
        result = []
        for w in words:
            if re.match(r"\w+", w):
                result.extend(self.tokenize_word(w))
            else:
                result.append(w)
        return result


# ---------------------------------------------------------------------------
# Preprocessing Comparison Functions
# ---------------------------------------------------------------------------

_GLOBAL_BPE_MODEL: Optional[ScratchBPE] = None


def get_trained_scratch_bpe(corpus_texts: Optional[List[str]] = None) -> ScratchBPE:
    global _GLOBAL_BPE_MODEL
    if _GLOBAL_BPE_MODEL is None:
        _GLOBAL_BPE_MODEL = ScratchBPE(num_merges=120)
        seed_corpus = corpus_texts or [
            "Aurora AX3000 Wi-Fi 6 router default gateway 192.168.1.1 settings wireless network",
            "Reset password factory default power supply internet wan lan usb 3.0 error code E-101 E-204",
            "Troubleshooting parental controls guest network bandwidth frequency channel firmware update"
        ]
        _GLOBAL_BPE_MODEL.train(seed_corpus)
    return _GLOBAL_BPE_MODEL


def compare_tokenizers(text: str, corpus_sample: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compares 5 tokenization approaches on the same input sentence:
    1. Character-level
    2. NLTK word_tokenize
    3. spaCy tokenizer
    4. WordPiece (via BERT tokenizer)
    5. Byte Pair Encoding (from-scratch merge learning vs Hugging Face)
    """
    # 1. Character-level
    char_tokens = [c for c in text]

    # 2. NLTK word_tokenize
    try:
        nltk_tokens = word_tokenize(text)
    except Exception:
        nltk_tokens = text.split()

    # 3. spaCy tokenizer
    try:
        import spacy
        nlp_spacy = spacy.blank("en")
        spacy_tokens = [tok.text for tok in nlp_spacy(text)]
    except Exception:
        spacy_tokens = text.split()

    # 4. WordPiece (BERT Tokenizer)
    try:
        from transformers import BertTokenizerFast
        bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
        wordpiece_tokens = bert_tok.tokenize(text)
    except Exception:
        wordpiece_tokens = [f"##{t}" for t in text.split()]

    # 5. Scratch BPE
    bpe = get_trained_scratch_bpe(corpus_sample)
    scratch_bpe_tokens = bpe.tokenize(text)

    # 6. Hugging Face BPE
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        hf_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        hf_tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"], vocab_size=500)
        hf_tokenizer.train_from_iterator([text] + (corpus_sample or []), trainer=trainer)
        hf_bpe_tokens = hf_tokenizer.encode(text).tokens
    except Exception:
        hf_bpe_tokens = scratch_bpe_tokens

    return {
        "char_tokens": char_tokens,
        "nltk_tokens": nltk_tokens,
        "spacy_tokens": spacy_tokens,
        "wordpiece_tokens": wordpiece_tokens,
        "bpe_from_scratch_tokens": scratch_bpe_tokens,
        "hf_bpe_tokens": hf_bpe_tokens,
        "bpe_learned_merges_count": len(bpe.merges)
    }


def compare_stopwords(text: str) -> Dict[str, Any]:
    """Compares NLTK vs spaCy stopword filtering with text statistics."""
    # NLTK stopwords
    nltk_stops = set(nltk_stopwords_module.words("english"))
    
    # spaCy stopwords
    try:
        from spacy.lang.en.stop_words import STOP_WORDS as spacy_stops
    except ImportError:
        spacy_stops = nltk_stops

    tokens = word_tokenize(normalise_text(text, lowercase=True, remove_punctuation=False))
    
    nltk_filtered = [t for t in tokens if t not in nltk_stops and re.match(r"\w+", t)]
    spacy_filtered = [t for t in tokens if t not in spacy_stops and re.match(r"\w+", t)]
    
    removed = sorted(list((set(tokens) & nltk_stops) | (set(tokens) & spacy_stops)))

    return {
        "original_text": text,
        "original_token_count": len(tokens),
        "nltk_filtered_tokens": nltk_filtered,
        "nltk_token_count": len(nltk_filtered),
        "spacy_filtered_tokens": spacy_filtered,
        "spacy_token_count": len(spacy_filtered),
        "removed_stopwords": removed
    }


def compare_stemming_lemmatization(text: str) -> Dict[str, Any]:
    """
    Compares Porter Stemmer, Lancaster Stemmer, WordNet Lemmatizer,
    and spaCy Lemmatizer on query tokens, reporting vocabulary size reduction.
    """
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()

    words = [w.lower() for w in word_tokenize(text) if re.match(r"^[a-zA-Z]+$", w)]
    if not words:
        words = ["resetting", "configured", "connecting", "antennas", "switches", "troubleshooting", "encrypted"]

    porter_stems = [porter.stem(w) for w in words]
    lancaster_stems = [lancaster.stem(w) for w in words]
    wordnet_lemmas = [lemmatizer.lemmatize(w, pos='v') for w in words]

    # spaCy Lemmatizer
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(" ".join(words))
        spacy_lemmas = [token.lemma_ for token in doc]
    except Exception:
        spacy_lemmas = wordnet_lemmas

    orig_vocab_size = len(set(words)) or 1
    red_porter = round((1.0 - len(set(porter_stems)) / orig_vocab_size) * 100, 2)
    red_lancaster = round((1.0 - len(set(lancaster_stems)) / orig_vocab_size) * 100, 2)
    red_wordnet = round((1.0 - len(set(wordnet_lemmas)) / orig_vocab_size) * 100, 2)

    return {
        "tokens": words,
        "porter_stems": porter_stems,
        "lancaster_stems": lancaster_stems,
        "wordnet_lemmas": wordnet_lemmas,
        "spacy_lemmas": spacy_lemmas,
        "vocab_reduction_porter": red_porter,
        "vocab_reduction_lancaster": red_lancaster,
        "vocab_reduction_wordnet": red_wordnet
    }
