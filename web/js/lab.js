/**
 * ManualMind NLP Laboratory Controller (web/js/lab.js)
 * Implements interactive handlers for all 13 syllabus tools:
 * 1. Tokenizers
 * 2. Stopwords
 * 3. Stemming vs Lemmatization
 * 4. Regex Pattern Extraction
 * 5. Levenshtein Dynamic Programming Matrix & Backtrace
 * 6. N-Gram Language Model
 * 7. POS Taggers Side-by-Side
 * 8. Syntactic Chunking
 * 9. Named Entity Recognition
 * 10. TF-IDF Matrix & Ranking
 * 11. Word2Vec Nearest Neighbours
 * 12. Sparse vs Dense Feature Representations
 * 13. Intent Classifier Comparison
 */

// 1. Tokenizer Comparison
async function runLabTokenization() {
  const text = document.getElementById("lab-tok-input").value;
  const container = document.getElementById("lab-tok-results");
  container.innerHTML = "Processing tokenizers...";

  try {
    const res = await fetch(`/api/lab/tokenization?text=${encodeURIComponent(text)}`);
    const data = await res.json();
    container.innerHTML = `
      <table class="frame-table" style="margin-top: 0.5rem;">
        <tr><th>Character-level</th><td>${data.char_tokens.slice(0, 30).join(" ")} ...</td></tr>
        <tr><th>NLTK word_tokenize</th><td>${data.nltk_tokens.map(t => `<span class="entity-pill">${t}</span>`).join(" ")}</td></tr>
        <tr><th>spaCy Tokenizer</th><td>${data.spacy_tokens.map(t => `<span class="entity-pill">${t}</span>`).join(" ")}</td></tr>
        <tr><th>WordPiece (BERT)</th><td>${data.wordpiece_tokens.map(t => `<span class="entity-pill" style="background:#EEF2FF">${t}</span>`).join(" ")}</td></tr>
        <tr><th>Byte Pair Encoding (Scratch)</th><td>${data.bpe_from_scratch_tokens.map(t => `<span class="entity-pill" style="background:#ECFDF5; border-color:#10B981">${t}</span>`).join(" ")} <small>(${data.bpe_learned_merges_count} merges)</small></td></tr>
        <tr><th>HuggingFace BPE</th><td>${data.hf_bpe_tokens.map(t => `<span class="entity-pill">${t}</span>`).join(" ")}</td></tr>
      </table>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 2. Stopword Filtering
async function runLabStopwords() {
  const text = document.getElementById("lab-stop-input").value;
  const container = document.getElementById("lab-stop-results");
  container.innerHTML = "Filtering stopwords...";

  try {
    const res = await fetch(`/api/lab/stopwords?text=${encodeURIComponent(text)}`);
    const data = await res.json();
    container.innerHTML = `
      <table class="frame-table" style="margin-top:0.5rem;">
        <tr><th>Original Tokens (${data.original_token_count})</th><td>${data.original_text}</td></tr>
        <tr><th>NLTK Filtered (${data.nltk_token_count})</th><td>${data.nltk_filtered_tokens.map(t => `<span class="entity-pill">${t}</span>`).join(" ")}</td></tr>
        <tr><th>spaCy Filtered (${data.spacy_token_count})</th><td>${data.spacy_filtered_tokens.map(t => `<span class="entity-pill">${t}</span>`).join(" ")}</td></tr>
        <tr><th>Removed Stopwords</th><td><span style="color:var(--danger); font-weight:600;">${data.removed_stopwords.join(", ")}</span></td></tr>
      </table>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 3. Stemming vs Lemmatization
async function runLabStemLemmatize() {
  const text = document.getElementById("lab-stem-input").value;
  const container = document.getElementById("lab-stem-results");
  container.innerHTML = "Comparing stemming and lemmatization...";

  try {
    const res = await fetch(`/api/lab/stem_lemmatize?text=${encodeURIComponent(text)}`);
    const data = await res.json();
    container.innerHTML = `
      <table class="frame-table" style="margin-top: 0.5rem;">
        <tr><th>Tokens</th><td>${data.tokens.join(" ")}</td></tr>
        <tr><th>Porter Stemmer</th><td>${data.porter_stems.join(" ")} <strong style="color:var(--accent); font-size:0.8rem;">(${data.vocab_reduction_porter}% vocab reduction)</strong></td></tr>
        <tr><th>Lancaster Stemmer</th><td>${data.lancaster_stems.join(" ")} <strong style="color:var(--accent); font-size:0.8rem;">(${data.vocab_reduction_lancaster}% vocab reduction)</strong></td></tr>
        <tr><th>WordNet Lemmatizer</th><td>${data.wordnet_lemmas.join(" ")} <strong style="color:var(--success); font-size:0.8rem;">(${data.vocab_reduction_wordnet}% vocab reduction)</strong></td></tr>
        <tr><th>spaCy Lemmatizer</th><td>${data.spacy_lemmas.join(" ")}</td></tr>
      </table>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 4. Regex Pattern Extraction
async function runLabRegex() {
  const text = document.getElementById("lab-regex-input").value;
  const container = document.getElementById("lab-regex-results");
  container.innerHTML = "Extracting technical regex patterns...";

  try {
    const res = await fetch(`/api/lab/regex?text=${encodeURIComponent(text)}`);
    const data = await res.json();
    let rows = "";
    for (const [name, matches] of Object.entries(data.matches)) {
      if (matches.length > 0) {
        rows += `<tr><th>${name}</th><td>${matches.map(m => `<span class="entity-pill" style="font-weight:700;">${m}</span>`).join(" ")}</td></tr>`;
      }
    }
    if (!rows) rows = `<tr><td colspan="2" style="color:var(--text-muted);">No regex patterns matched.</td></tr>`;
    container.innerHTML = `<table class="frame-table" style="margin-top:0.5rem;">${rows}</table>`;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 5. Levenshtein DP Matrix Visualization
async function runLabLevenshtein() {
  const w1 = document.getElementById("lab-lev-w1").value.trim();
  const w2 = document.getElementById("lab-lev-w2").value.trim();
  const container = document.getElementById("lab-lev-results");
  container.innerHTML = "Computing Levenshtein dynamic programming table...";

  try {
    const res = await fetch(`/api/lab/levenshtein?word1=${encodeURIComponent(w1)}&word2=${encodeURIComponent(w2)}`);
    const data = await res.json();

    const m = data.dp_matrix;
    const ops = data.operations;

    // Track backtrace cells for highlighting
    const backtraceCoords = new Set();
    ops.forEach(op => {
      if (op.cell) backtraceCoords.add(`${op.cell[0]},${op.cell[1]}`);
    });

    let tableHtml = `<div class="matrix-container"><table class="dp-table">`;
    tableHtml += `<tr><th></th><th>#</th>`;
    for (let j = 0; j < w2.length; j++) {
      tableHtml += `<th>${w2[j]}</th>`;
    }
    tableHtml += `</tr>`;

    for (let i = 0; i < m.length; i++) {
      const rowChar = i === 0 ? "#" : w1[i - 1];
      tableHtml += `<tr><th>${rowChar}</th>`;
      for (let j = 0; j < m[i].length; j++) {
        const isBt = backtraceCoords.has(`${i},${j}`);
        const cls = isBt ? "backtrace-cell" : "";
        tableHtml += `<td class="${cls}">${m[i][j]}</td>`;
      }
      tableHtml += `</tr>`;
    }
    tableHtml += `</table></div>`;

    let opsSummary = ops.map(o => `<strong>${o.op}</strong>(${o.char_src}➔${o.char_tgt})`).join(" ➔ ");

    container.innerHTML = `
      <div style="margin-top: 0.5rem;">
        <p><strong>Scratch Distance:</strong> ${data.distance} | <strong>NLTK Distance:</strong> ${data.nltk_distance} | <strong>RapidFuzz:</strong> ${data.rapidfuzz_distance}</p>
        <p style="font-size: 0.85rem; margin: 0.4rem 0;"><strong>Edit Operations Path:</strong> ${opsSummary}</p>
        ${tableHtml}
      </div>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 6. N-Gram Language Model
async function runLabNgram() {
  const prompt = document.getElementById("lab-ngram-input").value;
  const container = document.getElementById("lab-ngram-results");
  container.innerHTML = "Computing language model probabilities...";

  try {
    const res = await fetch(`/api/lab/ngram?prompt=${encodeURIComponent(prompt)}`);
    const data = await res.json();
    container.innerHTML = `
      <table class="frame-table" style="margin-top:0.5rem;">
        <tr><th>Prompt</th><td>${data.prompt}</td></tr>
        <tr><th>Top Next-Word Autocomplete</th><td>${data.autocomplete_suggestions.map(s => `<span class="choice-chip" style="display:inline-block; margin:2px;">${s}</span>`).join(" ")}</td></tr>
        <tr><th>Unigram Perplexity (N=1)</th><td><strong>${data.unigram_perplexity}</strong></td></tr>
        <tr><th>Bigram Perplexity (N=2)</th><td><strong>${data.bigram_perplexity}</strong></td></tr>
        <tr><th>Trigram Perplexity (N=3)</th><td><strong style="color:var(--success);">${data.trigram_perplexity}</strong> (Lowest perplexity)</td></tr>
      </table>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 7. POS Taggers Side-by-Side
async function runLabPOS() {
  const sentence = document.getElementById("lab-pos-input").value;
  const container = document.getElementById("lab-pos-results");
  container.innerHTML = "Running 4 POS taggers...";

  try {
    const res = await fetch(`/api/lab/pos?sentence=${encodeURIComponent(sentence)}`);
    const data = await res.json();

    let tableHtml = `<table class="frame-table" style="margin-top:0.5rem;"><thead><tr><th>Token</th><th>Rule-Based</th><th>Scratch HMM (Viterbi)</th><th>CRF Suite</th><th>spaCy Neural</th></tr></thead><tbody>`;
    data.tokens.forEach((tok, i) => {
      tableHtml += `
        <tr>
          <td><strong>${tok}</strong></td>
          <td><span class="entity-pill">${data.rule_based_tags[i]}</span></td>
          <td><span class="entity-pill" style="background:#ECFDF5; border-color:#10B981">${data.hmm_tags[i]}</span></td>
          <td><span class="entity-pill" style="background:#EEF2FF; border-color:#6366F1">${data.crf_tags[i]}</span></td>
          <td><span class="entity-pill" style="background:#FFFBEB; border-color:#F59E0B">${data.spacy_tags[i]}</span></td>
        </tr>
      `;
    });
    tableHtml += `</tbody></table>`;
    container.innerHTML = tableHtml;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 8. Chunking
async function runLabChunking() {
  const sentence = document.getElementById("lab-chunk-input").value;
  const container = document.getElementById("lab-chunk-results");
  container.innerHTML = "Parsing chunks...";

  try {
    const res = await fetch(`/api/lab/chunking?sentence=${encodeURIComponent(sentence)}`);
    const data = await res.json();
    container.innerHTML = `
      <table class="frame-table" style="margin-top:0.5rem;">
        <tr><th>NLTK Noun Phrases (NP)</th><td>${data.nltk_np_chunks.map(c => `<span class="entity-pill">${c}</span>`).join(" ")}</td></tr>
        <tr><th>NLTK Verb Phrases (VP)</th><td>${data.nltk_vp_chunks.map(c => `<span class="entity-pill" style="background:#ECFDF5">${c}</span>`).join(" ")}</td></tr>
        <tr><th>spaCy Noun Chunks</th><td>${data.spacy_noun_chunks.map(c => `<span class="entity-pill">${c}</span>`).join(" ")}</td></tr>
      </table>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 9. Named Entity Recognition
async function runLabNER() {
  const sentence = document.getElementById("lab-ner-input").value;
  const container = document.getElementById("lab-ner-results");
  container.innerHTML = "Extracting entities across 3 paradigms...";

  try {
    const res = await fetch(`/api/lab/ner?sentence=${encodeURIComponent(sentence)}`);
    const data = await res.json();
    container.innerHTML = `
      <div style="margin-top:0.5rem;">
        <div style="padding: 0.75rem; background: var(--bg-primary); border-radius: var(--radius-sm); margin-bottom: 0.75rem;">
          ${data.highlighted_html}
        </div>
        <table class="frame-table">
          <tr><th>Rule-Based NER</th><td>${data.rule_based_entities.map(e => `<strong>${e.text}</strong> [${e.label}]`).join(", ") || "None"}</td></tr>
          <tr><th>CRF Statistical NER</th><td>${data.crf_entities.map(e => `<strong>${e.text}</strong> [${e.label}]`).join(", ") || "None"}</td></tr>
          <tr><th>BERT Baseline (General)</th><td>${data.transformer_baseline_entities.map(e => `<strong>${e.text}</strong> [${e.label}]`).join(", ") || "None (missed domain entities)"}</td></tr>
        </table>
      </div>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 10. TF-IDF Matrix & Ranking
async function runLabTfidf() {
  const query = document.getElementById("lab-tfidf-input").value;
  const container = document.getElementById("lab-tfidf-results");
  container.innerHTML = "Building term-document matrix and ranking with cosine similarity...";

  try {
    const res = await fetch(`/api/lab/tfidf?query=${encodeURIComponent(query)}`);
    const data = await res.json();
    container.innerHTML = `
      <table class="frame-table" style="margin-top:0.5rem;">
        <tr><th>Query</th><td>${data.query}</td></tr>
        <tr><th>Scratch TF-IDF Top Matches</th><td>${data.scratch_ranking.map(r => `Doc #${r[0]} (score: ${r[1].toFixed(4)})`).join(" • ")}</td></tr>
        <tr><th>Scikit-Learn Top Matches</th><td>${data.sklearn_ranking.map(r => `Doc #${r[0]} (score: ${r[1].toFixed(4)})`).join(" • ")}</td></tr>
        <tr><th>Rank #1 Perfect Match?</th><td><strong style="color:${data.top1_match ? 'var(--success)' : 'var(--warning)'};">${data.top1_match ? 'YES (Identical Top-1 Ranking)' : 'Close rank'}</strong></td></tr>
      </table>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 11. Word2Vec Nearest Neighbours
async function runLabVectors() {
  const word = document.getElementById("lab-w2v-input").value.trim();
  const container = document.getElementById("lab-w2v-results");
  container.innerHTML = "Querying Word2Vec vector space...";

  try {
    const res = await fetch(`/api/lab/vector_neighbours?word=${encodeURIComponent(word)}`);
    const data = await res.json();
    const cbowList = data.neighbours.cbow.map(n => `<span>${n.word} <small>(${n.similarity})</small></span>`).join(" • ") || "None";
    const sgList = data.neighbours.skipgram.map(n => `<span>${n.word} <small>(${n.similarity})</small></span>`).join(" • ") || "None";

    container.innerHTML = `
      <table class="frame-table" style="margin-top:0.5rem;">
        <tr><th>Target Word</th><td><strong>${data.word}</strong></td></tr>
        <tr><th>CBOW Nearest Neighbours</th><td>${cbowList}</td></tr>
        <tr><th>Skip-Gram Nearest Neighbours</th><td>${sgList}</td></tr>
      </table>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 12. Sparse vs Dense Representations
async function runLabSparseDense() {
  const sentence = document.getElementById("lab-sd-input").value;
  const container = document.getElementById("lab-sd-results");
  container.innerHTML = "Computing sparse and dense representations...";

  try {
    const res = await fetch(`/api/lab/sparse_dense?sentence=${encodeURIComponent(sentence)}`);
    const data = await res.json();
    const sparseStr = data.sparse_features.map(f => `${f.feature_name}=1`).join(", ");
    const denseStr = data.dense_vector_preview.join(", ");

    container.innerHTML = `
      <table class="frame-table" style="margin-top:0.5rem;">
        <tr><th>Sentence</th><td>${data.sentence}</td></tr>
        <tr><th>Sparse Feature Vector (Dim: ${data.sparse_dimension})</th><td style="font-family:var(--font-mono); font-size:0.8rem;">{ ${sparseStr} ... } (mostly zeroes)</td></tr>
        <tr><th>Dense Sentence Vector (Dim: ${data.dense_dimension})</th><td style="font-family:var(--font-mono); font-size:0.8rem;">[ ${denseStr}, ... ] (continuous embeddings)</td></tr>
      </table>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

// 13. Intent Classifier Comparison
async function runLabIntent() {
  const query = document.getElementById("lab-intent-input").value;
  const container = document.getElementById("lab-intent-results");
  container.innerHTML = "Classifying query intent...";

  try {
    const res = await fetch(`/api/lab/intent?query=${encodeURIComponent(query)}`);
    const data = await res.json();
    container.innerHTML = `
      <table class="frame-table" style="margin-top:0.5rem;">
        <tr><th>Query</th><td>${data.query}</td></tr>
        <tr><th>Scratch Naive Bayes</th><td><strong>${data.naive_bayes_scratch.predicted_intent}</strong> (${Math.round(data.naive_bayes_scratch.confidence * 100)}% confidence)</td></tr>
        <tr><th>Sklearn Logistic Regression</th><td><strong>${data.logistic_regression_sklearn.predicted_intent}</strong> (${Math.round(data.logistic_regression_sklearn.confidence * 100)}% confidence)</td></tr>
      </table>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
  }
}

