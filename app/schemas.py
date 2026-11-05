"""
ManualMind Pydantic Schemas
Defines request and response data models for the FastAPI application,
ensuring strict type validation and documentation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core Chat & QA Pipeline Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The user's question or message")
    session_id: Optional[str] = Field(None, description="Per-session ID for conversation history & pronoun resolution")
    tagger_mode: Optional[str] = Field("spacy", description="POS tagger to use: 'spacy', 'hmm', 'crf', or 'regex'")
    retrieval_mode: Optional[str] = Field("hybrid", description="Retrieval mode: 'hybrid', 'bm25', 'dense', 'tfidf', 'w2v_cbow', 'w2v_skipgram'")
    generative_mode: bool = Field(False, description="Whether to run optional flan-t5 generative rewrite")
    clarification_selection: Optional[str] = Field(None, description="User selection if previously prompted with clarification options")


class Citation(BaseModel):
    manual: str
    page: int
    section_path: str
    chunk_id: str
    evidence_snippet: str
    score: float


class ChunkResult(BaseModel):
    chunk_id: str
    manual: str
    page: int
    section_path: str
    score: float
    rank: int
    content: str
    has_numbered_steps: bool = False


class SpellingResult(BaseModel):
    original: str
    corrected: str
    was_corrected: bool
    did_you_mean: Optional[str] = None
    edits: List[Dict[str, Any]] = []


class PronounResolutionResult(BaseModel):
    has_pronoun: bool
    resolved_entity: Optional[str] = None
    rewritten_query: str
    anaphora_detected: List[str] = []


class SemanticFrame(BaseModel):
    action: Optional[str] = None
    object: Optional[str] = None
    modifiers: List[str] = []
    negation: bool = False
    modality: Optional[str] = None
    conditions: List[str] = []


class LinguisticAnalysis(BaseModel):
    tokens: List[str]
    lemmas: List[str]
    morphology: List[Dict[str, str]]
    pos_tags: List[Dict[str, str]]
    chunks: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    dependency_tree_svg: Optional[str] = None
    semantic_frame: SemanticFrame


class AmbiguityReport(BaseModel):
    is_ambiguous: bool
    ambiguity_type: Optional[str] = None  # 'lexical', 'syntactic', or None
    trigger_words: List[str] = []
    readings: List[Dict[str, Any]] = []
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    clarification_options: List[str] = []


class IntentResult(BaseModel):
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    routing_action: str


class ClauseDecomposition(BaseModel):
    is_compound: bool
    sub_questions: List[str] = []
    conditions: List[str] = []
    main_actions: List[str] = []


class AnswerPayload(BaseModel):
    answer_text: str
    steps: List[str] = []
    citations: List[Citation] = []
    confidence: float
    is_extractive: bool = True
    generative_text: Optional[str] = None
    low_confidence: bool = False
    suggested_questions: List[str] = []
    refusal_reason: Optional[str] = None


class AskResponse(BaseModel):
    session_id: str
    query: str
    rewritten_query: str
    answer: AnswerPayload
    normalised_text: str
    spelling: SpellingResult
    pronoun_resolution: PronounResolutionResult
    linguistic_analysis: LinguisticAnalysis
    ambiguity: AmbiguityReport
    intent: IntentResult
    decomposition: ClauseDecomposition
    retrieval: List[ChunkResult]
    timings_ms: Dict[str, float]


# ---------------------------------------------------------------------------
# STT & Audio Schemas
# ---------------------------------------------------------------------------

class STTResponse(BaseModel):
    transcript: str
    duration_seconds: float
    detected_language: str
    confidence: float


# ---------------------------------------------------------------------------
# Manual Metadata Schemas
# ---------------------------------------------------------------------------

class ManualInfo(BaseModel):
    id: str
    name: str
    filename: str
    total_pages: int
    total_sections: int
    total_chunks: int
    vocabulary_size: int
    is_active: bool
    file_size_bytes: int


class ManualListResponse(BaseModel):
    manuals: List[ManualInfo]
    active_manual_id: str


# ---------------------------------------------------------------------------
# Autocomplete & Suggestions
# ---------------------------------------------------------------------------

class AutocompleteRequest(BaseModel):
    prefix: str = Field(..., max_length=200)
    max_suggestions: int = Field(5, ge=1, le=10)


class AutocompleteResponse(BaseModel):
    prefix: str
    suggestions: List[str]


# ---------------------------------------------------------------------------
# Lab Tools Request & Response Schemas
# ---------------------------------------------------------------------------

class LabTokenizationRequest(BaseModel):
    text: str


class LabTokenizationResponse(BaseModel):
    nltk_tokens: List[str]
    spacy_tokens: List[str]
    char_tokens: List[str]
    wordpiece_tokens: List[str]
    bpe_from_scratch_tokens: List[str]
    hf_bpe_tokens: List[str]
    bpe_learned_merges_count: int


class LabStopwordsRequest(BaseModel):
    text: str


class LabStopwordsResponse(BaseModel):
    original_text: str
    original_token_count: int
    nltk_filtered_tokens: List[str]
    nltk_token_count: int
    spacy_filtered_tokens: List[str]
    spacy_token_count: int
    removed_stopwords: List[str]


class LabStemLemmatizeRequest(BaseModel):
    text: str


class LabStemLemmatizeResponse(BaseModel):
    tokens: List[str]
    porter_stems: List[str]
    lancaster_stems: List[str]
    wordnet_lemmas: List[str]
    spacy_lemmas: List[str]
    vocab_reduction_porter: float
    vocab_reduction_lancaster: float
    vocab_reduction_wordnet: float


class LabRegexRequest(BaseModel):
    text: str


class LabRegexResponse(BaseModel):
    matches: Dict[str, List[str]]


class LabLevenshteinRequest(BaseModel):
    word1: str
    word2: str


class LabLevenshteinResponse(BaseModel):
    word1: str
    word2: str
    distance: int
    operations: List[Dict[str, Any]]
    dp_matrix: List[List[int]]
    backtrace_path: List[List[int]]
    nltk_distance: int
    rapidfuzz_distance: int


class LabNgramRequest(BaseModel):
    prompt: str
    n: int = 3
    num_words: int = 5


class LabNgramResponse(BaseModel):
    generated_text: str
    unigram_perplexity: float
    bigram_perplexity: float
    trigram_perplexity: float
    next_word_probabilities: List[Dict[str, Any]]


class LabPOSTaggingRequest(BaseModel):
    sentence: str


class LabPOSTaggingResponse(BaseModel):
    tokens: List[str]
    rule_based_tags: List[str]
    hmm_tags: List[str]
    crf_tags: List[str]
    spacy_tags: List[str]


class LabChunkingRequest(BaseModel):
    sentence: str


class LabChunkingResponse(BaseModel):
    nltk_np_chunks: List[str]
    nltk_vp_chunks: List[str]
    spacy_noun_chunks: List[str]


class LabNERRequest(BaseModel):
    sentence: str


class LabNERResponse(BaseModel):
    rule_based_entities: List[Dict[str, Any]]
    crf_entities: List[Dict[str, Any]]
    transformer_baseline_entities: List[Dict[str, Any]]


class LabTfidfRequest(BaseModel):
    query: str
    top_k: int = 5


class LabTfidfResponse(BaseModel):
    query_terms: List[str]
    top_terms: List[Dict[str, float]]
    scratch_scores: List[Dict[str, Any]]
    sklearn_scores: List[Dict[str, Any]]


class LabVectorNeighboursRequest(BaseModel):
    word: str
    top_n: int = 6


class LabVectorNeighboursResponse(BaseModel):
    word: str
    gensim_cbow_neighbours: List[Dict[str, float]]
    gensim_skipgram_neighbours: List[Dict[str, float]]
    scratch_skipgram_neighbours: List[Dict[str, float]]


class LabSparseDenseRequest(BaseModel):
    sentence: str


class LabSparseDenseResponse(BaseModel):
    sentence: str
    sparse_features: List[Dict[str, Any]]
    sparse_dimension: int
    dense_vector_preview: List[float]
    dense_dimension: int


class LabIntentRequest(BaseModel):
    query: str


class LabIntentResponse(BaseModel):
    query: str
    naive_bayes_scratch: Dict[str, Any]
    logistic_regression_sklearn: Dict[str, Any]


# ---------------------------------------------------------------------------
# Method Evolution Lab Schemas
# ---------------------------------------------------------------------------

class EraResult(BaseModel):
    era_id: int
    era_name: str
    description: str
    methodology: str
    retrieved_section: str
    answer_text: str
    confidence_score: float
    latency_ms: float


class MethodEvolutionResponse(BaseModel):
    query: str
    eras: List[EraResult]
    historical_benchmark_accuracies: Dict[str, float]
