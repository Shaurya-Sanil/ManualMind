/**
 * ManualMind Assistant & Inspector Controller (web/js/assistant.js)
 * Implements:
 * 1. Conversational Chat & Message Flow
 * 2. Voice STT (MediaRecorder + faster-whisper) & Voice TTS (Web Speech API)
 * 3. Live Autocomplete (N-Gram Language Model)
 * 4. Interactive Clarification Chips & Did-You-Mean
 * 5. Full NLP Inspector Visualizations (SVG Dependency Trees, Tokens, Semantic Frames)
 */

let currentSessionId = "session_" + Math.random().toString(36).substring(2, 9);
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let autocompleteDebounce = null;

document.addEventListener("DOMContentLoaded", () => {
  initChatInput();
  initVoiceSTT();
  initInspectorTabs();
});

// ---------------------------------------------------------------------------
// 1. Chat Input & Autocomplete Flow
// ---------------------------------------------------------------------------

function initChatInput() {
  const queryInput = document.getElementById("query-input");
  const sendBtn = document.getElementById("send-btn");

  sendBtn.addEventListener("click", () => submitUserQuery());
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      closeAutocomplete();
      submitUserQuery();
    }
  });

  // Autocomplete typing listener
  queryInput.addEventListener("input", () => {
    clearTimeout(autocompleteDebounce);
    const val = queryInput.value;
    if (val.trim().length >= 2) {
      autocompleteDebounce = setTimeout(() => fetchAutocomplete(val), 180);
    } else {
      closeAutocomplete();
    }
  });

  // Click outside to dismiss autocomplete
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".chat-input-area")) {
      closeAutocomplete();
    }
  });
}

async function fetchAutocomplete(prefix) {
  try {
    const res = await fetch(`/api/autocomplete?prefix=${encodeURIComponent(prefix)}&max_suggestions=4`);
    if (res.ok) {
      const data = await res.json();
      renderAutocompleteDropdown(data.suggestions);
    }
  } catch (err) {
    console.error("Autocomplete error:", err);
  }
}

function renderAutocompleteDropdown(suggestions) {
  const dropdown = document.getElementById("autocomplete-dropdown");
  if (!dropdown) return;

  if (!suggestions || suggestions.length === 0) {
    dropdown.style.display = "none";
    dropdown.innerHTML = "";
    return;
  }

  dropdown.innerHTML = "";
  suggestions.forEach(item => {
    const div = document.createElement("div");
    div.className = "autocomplete-item";
    div.innerHTML = `<span>${item}</span> <span style="opacity: 0.5; font-size: 0.75rem;">↵ Tab</span>`;
    div.addEventListener("click", () => {
      const queryInput = document.getElementById("query-input");
      queryInput.value = item;
      queryInput.focus();
      closeAutocomplete();
    });
    dropdown.appendChild(div);
  });
  dropdown.style.display = "block";
}

function closeAutocomplete() {
  const dropdown = document.getElementById("autocomplete-dropdown");
  if (dropdown) dropdown.style.display = "none";
}

// ---------------------------------------------------------------------------
// 2. Query Submission & Pipeline Execution
// ---------------------------------------------------------------------------

async function submitUserQuery(overrideText = null, clarificationSelection = null) {
  const queryInput = document.getElementById("query-input");
  const query = (overrideText || queryInput.value).trim();
  if (!query) return;

  if (!overrideText) {
    queryInput.value = "";
  }
  closeAutocomplete();

  // Append user message bubble to chat
  appendMessage("user", query);

  // Animate progress strip
  setStageActive("stage-prep");

  const isGenerative = document.getElementById("generative-toggle")?.checked || false;

  try {
    const payload = {
      query: query,
      session_id: currentSessionId,
      generative_mode: isGenerative,
      clarification_selection: clarificationSelection
    };

    setStageActive("stage-understand");
    const t0 = performance.now();
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${res.status}`);
    }

    setStageActive("stage-retrieve");
    setStageActive("stage-answer");
    const responseData = await res.json();
    setStageDoneAll();

    // Render assistant answer bubble
    renderAssistantResponse(responseData);

    // Update Live NLP Inspector with complete structured linguistic data
    updateNLPInspector(responseData);

    // Optional Text-To-Speech
    const autoSpeak = document.getElementById("auto-speak-toggle")?.checked || false;
    if (autoSpeak) {
      speakResponse(responseData.answer);
    }

  } catch (err) {
    console.error("Ask query failed:", err);
    appendMessage("assistant", `⚠️ An error occurred: ${err.message}`);
    resetStages();
  }
}

// ---------------------------------------------------------------------------
// 3. Chat Message Rendering
// ---------------------------------------------------------------------------

function appendMessage(sender, text) {
  const chatHistory = document.getElementById("chat-history");
  const bubble = document.createElement("div");
  bubble.className = `message-bubble ${sender}`;

  const content = document.createElement("div");
  content.className = "content";
  content.textContent = text;

  const meta = document.createElement("div");
  meta.className = "message-meta";
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  meta.textContent = `${sender === 'user' ? 'You' : 'ManualMind'} • ${now}`;

  bubble.appendChild(content);
  bubble.appendChild(meta);
  chatHistory.appendChild(bubble);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return bubble;
}

function renderAssistantResponse(data) {
  const chatHistory = document.getElementById("chat-history");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble assistant";

  const content = document.createElement("div");
  content.className = "content";

  // 1. Did You Mean chip if corrected
  if (data.spelling && data.spelling.was_corrected && data.spelling.did_you_mean) {
    const dym = document.createElement("div");
    dym.style.fontSize = "0.8rem";
    dym.style.color = "var(--text-muted)";
    dym.style.marginBottom = "0.5rem";
    dym.innerHTML = `Did you mean: <strong style="cursor: pointer; color: var(--accent);">${data.spelling.did_you_mean}</strong>?`;
    dym.querySelector("strong").addEventListener("click", () => {
      submitUserQuery(data.spelling.did_you_mean);
    });
    content.appendChild(dym);
  }

  // 2. Main text or generative rewrite
  const p = document.createElement("p");
  if (data.answer.generative_text) {
    p.innerHTML = `<span style="font-size:0.78rem; font-weight:700; color:var(--accent);">[FLAN-T5 Synthesis]</span> ${data.answer.generative_text}`;
  } else if (data.answer.steps && data.answer.steps.length > 0) {
    // When ordered steps are extracted, display the lead sentence cleanly without repeating raw numbered text
    const leadLine = data.answer.answer_text.split("\n")[0];
    p.textContent = leadLine;
  } else {
    p.textContent = data.answer.answer_text;
  }
  content.appendChild(p);

  // 3. Numbered steps if present
  if (data.answer.steps && data.answer.steps.length > 0) {
    const stepsList = document.createElement("div");
    stepsList.className = "steps-list";
    data.answer.steps.forEach((step, idx) => {
      const stepItem = document.createElement("div");
      stepItem.className = "step-item";
      stepItem.innerHTML = `<span class="step-num">${idx + 1}.</span> <span>${step}</span>`;
      stepsList.appendChild(stepItem);
    });
    content.appendChild(stepsList);
  }

  // 4. Clarification question & choices (Ambiguity Guard)
  if (data.ambiguity && data.ambiguity.clarification_needed && data.ambiguity.clarification_options.length > 0) {
    const clarBox = document.createElement("div");
    clarBox.className = "clarification-box";
    clarBox.innerHTML = `<strong>⚠️ Ambiguity Clarification:</strong> ${data.ambiguity.clarification_question}`;

    const optsDiv = document.createElement("div");
    optsDiv.className = "clarification-options";
    data.ambiguity.clarification_options.forEach(opt => {
      const btn = document.createElement("button");
      btn.className = "choice-chip";
      btn.textContent = `➔ ${opt}`;
      btn.addEventListener("click", () => {
        submitUserQuery(data.query, opt);
      });
      optsDiv.appendChild(btn);
    });
    clarBox.appendChild(optsDiv);
    content.appendChild(clarBox);
  }

  // 5. Evidence Citations (Expandable on Click)
  if (data.answer.citations && data.answer.citations.length > 0) {
    const citWrapper = document.createElement("div");
    citWrapper.className = "citation-wrapper";
    citWrapper.style.marginTop = "0.75rem";

    const citTitle = document.createElement("div");
    citTitle.style.fontSize = "0.74rem";
    citTitle.style.fontWeight = "600";
    citTitle.style.color = "var(--text-muted)";
    citTitle.style.marginBottom = "0.35rem";
    citTitle.textContent = "Ground-Truth Citations (click any page to expand text evidence):";
    citWrapper.appendChild(citTitle);

    const citContainer = document.createElement("div");
    citContainer.className = "citation-container";

    const evidenceDrawer = document.createElement("div");
    evidenceDrawer.className = "citation-evidence-card";
    evidenceDrawer.style.display = "none";

    let activeChip = null;

    data.answer.citations.forEach((c, idx) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "citation-chip";
      const cleanSection = c.section_path.split(">").pop().trim();
      chip.innerHTML = `📄 Page ${c.page} • ${cleanSection} <span class="cit-arrow">▾</span>`;
      
      chip.addEventListener("click", () => {
        if (activeChip === chip) {
          // Collapse currently open drawer
          evidenceDrawer.style.display = "none";
          chip.classList.remove("active");
          const arrow = chip.querySelector(".cit-arrow");
          if (arrow) arrow.textContent = "▾";
          activeChip = null;
        } else {
          // Expand evidence for clicked citation
          if (activeChip) {
            activeChip.classList.remove("active");
            const prevArrow = activeChip.querySelector(".cit-arrow");
            if (prevArrow) prevArrow.textContent = "▾";
          }
          chip.classList.add("active");
          const arrow = chip.querySelector(".cit-arrow");
          if (arrow) arrow.textContent = "▴";
          activeChip = chip;

          evidenceDrawer.innerHTML = `
            <div class="citation-evidence-header">
              <span>📖 PDF Evidence Passage • Page ${c.page}</span>
              <span class="citation-score-badge">Relevance: ${(c.score * 100).toFixed(1)}%</span>
            </div>
            <div class="citation-evidence-meta">
              <span>📁 <strong>Hierarchy:</strong> ${c.section_path}</span>
              <span>🏷️ <strong>Chunk:</strong> ${c.chunk_id || `chunk_${idx+1}`}</span>
            </div>
            <div class="citation-evidence-body">${c.evidence_snippet}</div>
          `;
          evidenceDrawer.style.display = "block";
        }
      });

      citContainer.appendChild(chip);
    });

    citWrapper.appendChild(citContainer);
    citWrapper.appendChild(evidenceDrawer);
    content.appendChild(citWrapper);
  }

  // 6. Low Confidence Alert & Suggested Questions
  if (data.answer.low_confidence || (data.answer.suggested_questions && data.answer.suggested_questions.length > 0 && data.answer.confidence < 0.6)) {
    const suggBox = document.createElement("div");
    suggBox.style.marginTop = "0.75rem";
    suggBox.style.fontSize = "0.82rem";
    suggBox.innerHTML = `<div style="color:var(--text-muted); font-weight:600; margin-bottom:0.3rem;">Suggested related questions:</div>`;
    const suggList = document.createElement("div");
    suggList.style.display = "flex";
    suggList.style.flexWrap = "wrap";
    suggList.style.gap = "0.3rem";

    data.answer.suggested_questions.forEach(sq => {
      const sqChip = document.createElement("button");
      sqChip.className = "choice-chip";
      sqChip.style.fontSize = "0.78rem";
      sqChip.textContent = sq;
      sqChip.addEventListener("click", () => submitUserQuery(sq));
      suggList.appendChild(sqChip);
    });
    suggBox.appendChild(suggList);
    content.appendChild(suggBox);
  }

  // Meta
  const meta = document.createElement("div");
  meta.className = "message-meta";
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const confPercent = Math.round(data.answer.confidence * 100);
  meta.textContent = `Confidence: ${confPercent}% • Intent: ${data.intent.intent} • ${data.timings_ms.total_pipeline_ms}ms • ${now}`;

  bubble.appendChild(content);
  bubble.appendChild(meta);
  chatHistory.appendChild(bubble);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

// ---------------------------------------------------------------------------
// 4. Progress Strip Helpers
// ---------------------------------------------------------------------------

function setStageActive(stageId) {
  const ids = ["stage-prep", "stage-understand", "stage-intent", "stage-retrieve", "stage-answer"];
  let passed = true;
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove("active");
    if (id === stageId) {
      el.classList.add("active");
      passed = false;
    } else if (passed) {
      el.classList.add("done");
    }
  });
}

function setStageDoneAll() {
  ["stage-prep", "stage-understand", "stage-intent", "stage-retrieve", "stage-answer"].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.classList.remove("active");
      el.classList.add("done");
    }
  });
}

function resetStages() {
  ["stage-prep", "stage-understand", "stage-intent", "stage-retrieve", "stage-answer"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove("active", "done");
  });
}

// ---------------------------------------------------------------------------
// 5. NLP Inspector Updates
// ---------------------------------------------------------------------------

function updateNLPInspector(data) {
  // Latency badge
  const tb = document.getElementById("timing-badge");
  if (tb) tb.textContent = `Latency: ${data.timings_ms.total_pipeline_ms} ms`;

  // Tab 1: Linguistic Levels
  // Dependency Tree SVG
  const depTree = document.getElementById("inspector-dep-tree");
  if (depTree && data.linguistic_analysis.dependency_tree_svg) {
    depTree.innerHTML = data.linguistic_analysis.dependency_tree_svg;
  }

  // Tokens & Badges
  const tokGrid = document.getElementById("inspector-tokens");
  if (tokGrid) {
    tokGrid.innerHTML = "";
    data.linguistic_analysis.tokens.forEach((tok, idx) => {
      const pos = data.linguistic_analysis.pos_tags[idx]?.pos || "NOUN";
      const morph = data.linguistic_analysis.morphology[idx]?.affix || "";
      const b = document.createElement("div");
      b.className = "token-badge";
      b.innerHTML = `
        <span class="tok">${tok}</span>
        <span class="pos">${pos}</span>
        ${morph !== 'none' ? `<span class="morph">${morph}</span>` : ''}
      `;
      tokGrid.appendChild(b);
    });
  }

  // Chunks
  const chunkContainer = document.getElementById("inspector-chunks");
  if (chunkContainer) {
    chunkContainer.innerHTML = "";
    data.linguistic_analysis.chunks.forEach(c => {
      const sp = document.createElement("span");
      sp.className = "entity-pill";
      sp.style.padding = "2px 8px";
      sp.style.background = c.type === "NP" ? "rgba(19, 64, 116, 0.1)" : "rgba(42, 157, 143, 0.1)";
      sp.style.border = "1px solid var(--border-color)";
      sp.style.borderRadius = "4px";
      sp.style.fontSize = "0.82rem";
      sp.innerHTML = `<strong>${c.type}:</strong> ${c.text}`;
      chunkContainer.appendChild(sp);
    });
  }

  // Entities
  const entContainer = document.getElementById("inspector-entities");
  if (entContainer) {
    entContainer.innerHTML = "";
    if (data.linguistic_analysis.entities.length === 0) {
      entContainer.innerHTML = `<span style="color:var(--text-muted); font-size:0.85rem;">No technical entities detected</span>`;
    } else {
      data.linguistic_analysis.entities.forEach(e => {
        const sp = document.createElement("span");
        sp.className = "entity-pill";
        sp.style.padding = "2px 8px";
        sp.style.background = "rgba(0, 119, 182, 0.12)";
        sp.style.border = "1px solid var(--accent)";
        sp.style.color = "var(--accent)";
        sp.style.borderRadius = "4px";
        sp.style.fontSize = "0.82rem";
        sp.style.fontWeight = "600";
        sp.innerHTML = `${e.text} <small>[${e.label}]</small>`;
        entContainer.appendChild(sp);
      });
    }
  }

  // Semantic Frame Table
  const sf = data.linguistic_analysis.semantic_frame;
  document.getElementById("frame-action").textContent = sf.action || "None";
  document.getElementById("frame-object").textContent = sf.object || "None";
  document.getElementById("frame-modifiers").textContent = sf.modifiers.join(", ") || "None";
  document.getElementById("frame-negation").textContent = sf.negation ? "True (Negated)" : "False";
  document.getElementById("frame-modality").textContent = sf.modality || "None";
  document.getElementById("frame-conditions").textContent = sf.conditions.join("; ") || "None";

  // Tab 2: Rewrite & Decomp
  document.getElementById("decomp-entity").textContent = data.pronoun_resolution.resolved_entity || "None";
  document.getElementById("decomp-rewritten").textContent = data.pronoun_resolution.rewritten_query;
  document.getElementById("decomp-is-compound").textContent = data.decomposition.is_compound ? "Yes (Splitting applied)" : "No";

  const subQList = document.getElementById("decomp-subquestions");
  if (subQList) {
    subQList.innerHTML = "";
    data.decomposition.sub_questions.forEach(sq => {
      const li = document.createElement("li");
      li.textContent = sq;
      subQList.appendChild(li);
    });
  }

  // Tab 3: Ambiguity Guard
  document.getElementById("amb-detected").textContent = data.ambiguity.is_ambiguous ? "Yes (Clarification Triggered)" : "No";
  document.getElementById("amb-type").textContent = data.ambiguity.ambiguity_type || "None";

  const ambReadings = document.getElementById("amb-readings-list");
  if (ambReadings) {
    ambReadings.innerHTML = "";
    if (data.ambiguity.readings && data.ambiguity.readings.length > 0) {
      data.ambiguity.readings.forEach((r, idx) => {
        const div = document.createElement("div");
        div.style.padding = "0.4rem 0.6rem";
        div.style.background = "var(--bg-primary)";
        div.style.border = "1px solid var(--border-color)";
        div.style.borderRadius = "4px";
        div.style.marginBottom = "0.4rem";
        div.style.fontSize = "0.85rem";
        div.innerHTML = `<strong>Reading ${idx + 1}:</strong> ${r.reading}`;
        ambReadings.appendChild(div);
      });
    }
  }

  // Tab 4: Intent Routing
  document.getElementById("intent-name").textContent = data.intent.intent.toUpperCase();
  document.getElementById("intent-conf").textContent = `${Math.round(data.intent.confidence * 100)}%`;
  document.getElementById("intent-route").textContent = data.intent.routing_action;

  const probBars = document.getElementById("intent-prob-bars");
  if (probBars) {
    probBars.innerHTML = "";
    for (const [cls, prob] of Object.entries(data.intent.probabilities)) {
      const pPercent = Math.round(prob * 100);
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.alignItems = "center";
      row.style.fontSize = "0.78rem";
      row.style.gap = "0.5rem";
      row.innerHTML = `
        <span style="min-width: 110px;">${cls}:</span>
        <div style="flex: 1; background: var(--border-color); height: 8px; border-radius: 4px; overflow: hidden;">
          <div style="width: ${pPercent}%; background: var(--accent); height: 100%;"></div>
        </div>
        <span style="min-width: 35px; text-align: right;">${pPercent}%</span>
      `;
      probBars.appendChild(row);
    }
  }

  // Tab 5: Retrieval & Chunks
  const retList = document.getElementById("inspector-retrieval-list");
  if (retList) {
    retList.innerHTML = "";
    data.retrieval.forEach(ch => {
      const card = document.createElement("div");
      card.style.background = "var(--bg-primary)";
      card.style.border = "1px solid var(--border-color)";
      card.style.borderRadius = "var(--radius-sm)";
      card.style.padding = "0.75rem";
      card.style.fontSize = "0.85rem";
      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-weight: 700; color: var(--accent); margin-bottom: 0.3rem;">
          <span>Rank #${ch.rank} • ${ch.chunk_id}</span>
          <span>Score: ${ch.score}</span>
        </div>
        <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.4rem;">
          📍 ${ch.section_path} (Page ${ch.page})
        </div>
        <div style="color: var(--text-secondary); line-height: 1.4;">
          ${ch.content.substring(0, 220)}...
        </div>
      `;
      retList.appendChild(card);
    });
  }

  // Tab 6: Timings
  document.getElementById("t-total").textContent = `${data.timings_ms.total_pipeline_ms} ms`;
  document.getElementById("t-prep").textContent = `${data.timings_ms.preprocessing_spelling_ms || 0} ms`;
  document.getElementById("t-ling").textContent = `${data.timings_ms.linguistic_analysis_ms || 0} ms`;
  document.getElementById("t-pronoun").textContent = `${data.timings_ms.pronoun_resolution_ms || 0} ms`;
  document.getElementById("t-intent").textContent = `${data.timings_ms.intent_classification_ms || 0} ms`;
  document.getElementById("t-ret").textContent = `${data.timings_ms.retrieval_ms || 0} ms`;
  document.getElementById("t-ans").textContent = `${data.timings_ms.answering_ms || 0} ms`;
}

function initInspectorTabs() {
  const tabs = document.querySelectorAll(".inspector-tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-content").forEach(c => {
        c.style.display = "none";
        c.classList.remove("active");
      });
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.style.display = "block";
        targetContent.classList.add("active");
      }
    });
  });
}

// ---------------------------------------------------------------------------
// 6. Voice Integration (STT & TTS)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 6. Voice Integration (STT & TTS)
// ---------------------------------------------------------------------------

let browserRecognition = null;

function initVoiceSTT() {
  const micBtn = document.getElementById("mic-btn");
  const voiceStatus = document.getElementById("voice-status-text");

  // Check for native Web Speech Recognition API (Chrome, Edge, Safari)
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition) {
    browserRecognition = new SpeechRecognition();
    browserRecognition.continuous = false;
    browserRecognition.interimResults = true;
    browserRecognition.lang = "en-US";

    let finalTranscript = "";

    browserRecognition.onstart = () => {
      isRecording = true;
      finalTranscript = "";
      micBtn.classList.add("recording");
      micBtn.title = "Listening... Click to stop";
      if (voiceStatus) voiceStatus.textContent = "🎙️ Listening... (Speak your question clearly)";
    };

    browserRecognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interim += event.results[i][0].transcript;
        }
      }
      const queryInput = document.getElementById("query-input");
      if (queryInput) {
        queryInput.value = finalTranscript || interim;
      }
      if (voiceStatus) {
        voiceStatus.textContent = `🎙️ Hearing: "${(finalTranscript || interim).substring(0, 35)}..."`;
      }
    };

    browserRecognition.onerror = (event) => {
      console.warn("Native SpeechRecognition error:", event.error);
      isRecording = false;
      micBtn.classList.remove("recording");
      micBtn.title = "Click to speak";
      if (voiceStatus) {
        voiceStatus.textContent = event.error === "no-speech" 
          ? "Voice: No speech detected. Please try again."
          : `Voice notice: ${event.error}`;
      }
    };

    browserRecognition.onend = () => {
      isRecording = false;
      micBtn.classList.remove("recording");
      micBtn.title = "Click to speak";

      const queryInput = document.getElementById("query-input");
      const textToSubmit = (finalTranscript || queryInput?.value || "").trim();

      if (textToSubmit && textToSubmit.length >= 3) {
        if (voiceStatus) voiceStatus.textContent = `✅ Transcribed: "${textToSubmit}"`;
        // Ensure auto-speak is enabled for voice turns so assistant talks back
        const autoSpeak = document.getElementById("auto-speak-toggle");
        if (autoSpeak) autoSpeak.checked = true;
        // Automatically submit the voice query
        submitUserQuery(textToSubmit);
      } else {
        if (voiceStatus) voiceStatus.textContent = "Voice: Ready (Click mic to speak)";
      }
    };

    micBtn.addEventListener("click", () => {
      if (isRecording) {
        browserRecognition.stop();
      } else {
        try {
          browserRecognition.start();
        } catch (e) {
          console.error("Failed to start browser recognition:", e);
        }
      }
    });

    if (voiceStatus) voiceStatus.textContent = "Voice: Ready (Microphone enabled)";
    return;
  }

  // Fallback: MediaRecorder + Server-side Whisper /api/stt
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (voiceStatus) voiceStatus.textContent = "Voice: Microphone not supported in this browser";
    if (micBtn) micBtn.disabled = true;
    return;
  }

  micBtn.addEventListener("click", () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      await sendAudioToSTT(audioBlob);
      stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    const micBtn = document.getElementById("mic-btn");
    micBtn.classList.add("recording");
    micBtn.title = "Click to stop recording";
    document.getElementById("voice-status-text").textContent = "Voice: Recording audio... Click mic to stop";
  } catch (err) {
    console.error("Microphone access denied:", err);
    alert("Microphone access was denied or is unavailable. Please check your browser audio permissions.");
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    const micBtn = document.getElementById("mic-btn");
    micBtn.classList.remove("recording");
    micBtn.title = "Click to speak";
    document.getElementById("voice-status-text").textContent = "Voice: Processing with Whisper...";
  }
}

async function sendAudioToSTT(audioBlob) {
  const voiceStatus = document.getElementById("voice-status-text");
  try {
    const formData = new FormData();
    formData.append("file", audioBlob, "recording.webm");

    const res = await fetch("/api/stt", {
      method: "POST",
      body: formData
    });

    if (res.ok) {
      const data = await res.json();
      const rawText = (data.transcript || "").trim();
      
      // If server returned fallback or valid text
      if (rawText && !rawText.startsWith("[")) {
        const input = document.getElementById("query-input");
        if (input) input.value = rawText;
        if (voiceStatus) voiceStatus.textContent = `Voice: Transcribed (${data.duration_seconds}s)`;
        
        // Auto-enable speech output for voice requests
        const autoSpeak = document.getElementById("auto-speak-toggle");
        if (autoSpeak) autoSpeak.checked = true;

        // Auto-submit the voice query
        submitUserQuery(rawText);
      } else {
        if (voiceStatus) voiceStatus.textContent = "Voice: Could not recognize speech. Please speak clearly.";
      }
    } else {
      throw new Error(`STT HTTP error ${res.status}`);
    }
  } catch (err) {
    console.error("STT request failed:", err);
    if (voiceStatus) voiceStatus.textContent = "Voice: Transcription failed. Please type query.";
  }
}

function speakResponse(answerPayload) {
  if (!("speechSynthesis" in window)) return;

  window.speechSynthesis.cancel(); // Stop prior speech

  let spokenText = answerPayload.answer_text;
  if (answerPayload.steps && answerPayload.steps.length > 0) {
    spokenText += " Step " + answerPayload.steps.map((s, idx) => `${idx + 1}: ${s}`).join(". Step ");
  }

  // Normalise technical units for natural TTS
  spokenText = spokenText.replace(/5\s*GHz/gi, "five gigahertz")
    .replace(/2\.4\s*GHz/gi, "two point four gigahertz")
    .replace(/160\s*MHz/gi, "one hundred sixty megahertz")
    .replace(/12\s*V/gi, "twelve volts")
    .replace(/2\.0\s*A/gi, "two point zero amperes")
    .replace(/\bE-(\d)(\d)(\d)\b/gi, "Error code E $1 $2 $3");

  const utterance = new SpeechSynthesisUtterance(spokenText);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.lang = "en-US";

  window.speechSynthesis.speak(utterance);
}

