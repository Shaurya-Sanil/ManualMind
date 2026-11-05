# ManualMind: 5-Minute Laboratory Demonstration Script

This script provides an exact, step-by-step walkthrough for presenting **ManualMind** to the examination committee. It highlights every theoretical NLP concept from the syllabus, demonstrates both text and voice interaction, and shows the live NLP Inspector and Method Evolution Lab.

---

## Pre-Demo Checklist (30 seconds before presentation)
1. Launch server locally or open the live deployment:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
   ```
2. Navigate browser to: **`http://localhost:7860`**
3. Verify that the loaded manual shows **`aurora_ax3000_manual.pdf` (33 chunks indexed)** in the top banner.
4. Ensure computer audio/microphone permissions are granted in the browser.

---

## Act 1: The Core Assistant & Typo Correction (Minute 1:00)

### Action:
1. On the **Assistant** page, type this deliberate typo-filled query into the chat input:
   ```text
   How do I rest the wireles pasword?
   ```
2. Click **Ask** (or press Enter).

### Examiner Explanation (What to say):
> *"Here, the user made multiple typos ('rest' for 'reset', 'wireles', 'pasword'). Notice the 'Did you mean' chip at the top of the response bubble. ManualMind implements Levenshtein edit distance from scratch with a dynamic programming backtrace matrix, combined with word-frequency priors from the manual. It corrected the query to 'How do I reset the wireless password?' before sending it to the retrieval engine, and extracted the exact 4-step procedure with citations to Section 4.1."*

### Inspector Demo:
- Click the **Inspect NLP** button under the assistant response.
- Point to:
  1. **Morphology & Lemmas:** Show suffix decomposition (`resetting` $\to$ `reset` + `-ing`).
  2. **Dependency Tree:** Click the **Dependency Tree** tab to show the inline SVG rendered directly from spaCy.
  3. **Semantic Frame:** Point to `{action: 'reset', object: 'wireless password'}`.

---

## Act 2: Multi-Turn Conversation & Pronoun Resolution (Minute 2:00)

### Action:
1. Ask Turn 1:
   ```text
   Where is the blue WAN port located?
   ```
2. Read answer (points to the rear panel).
3. Now immediately ask Turn 2 with an anaphoric pronoun:
   ```text
   What happens if I disconnect it?
   ```

### Examiner Explanation (What to say):
> *"Customer support queries are inherently conversational. Notice that in Turn 2, the user asked 'What happens if I disconnect it?'. A naive keyword or embedding search on 'disconnect it' would fail. Look at the NLP Inspector under Query Understanding: the Pronoun Resolution module tracked the salient entity from Turn 1 ('WAN port') in the session memory stack, detected the pronoun 'it', and rewritten the effective query as 'What happens if I disconnect the WAN port?'. The response cites error code E-101 and troubleshooting steps."*

---

## Act 3: Ambiguity Guard: Lexical & Syntactic (Minute 3:00)

### Action A (Lexical Disambiguation):
1. Type into the chat:
   ```text
   How do I reset the router?
   ```
2. Observe that the assistant pauses and displays **interactive clarification buttons**:
   - `[Soft Reset (reboot router via Web GUI / Power switch)]`
   - `[Factory Hard Reset (press rear pin-hole button for 10s to erase all settings)]`
3. Click the **Factory Hard Reset** button.

### Examiner Explanation:
> *"The word 'reset' in technical manuals is polysemous. In the lecture, we studied lexical ambiguity. Rather than guessing and risking accidental data loss for a customer, the Ambiguity Guard uses a domain sense inventory and WordNet Lesk disambiguation. When alternative readings lead to divergent retrieval targets, it asks an interactive clarification question. Clicking 'Factory Hard Reset' instantly brings up the 10-second rear button instructions."*

### Action B (Syntactic PP-Attachment Ambiguity):
1. Type:
   ```text
   Update the firmware on the router with the red light
   ```
2. Point out the syntactic clarification option displaying the two attachment parses (Action applied to the router vs device characterized by the red light).

---

## Act 4: Pattern Extraction & Adversarial Refusal (Minute 3:45)

### Action A (Exact Error Code Regex Boost):
1. Type:
   ```text
   What does error code E-101 mean?
   ```
2. Point out the instantaneous retrieval (Rank 1, 100% confidence) citing Section 8.1.
> *"Module 3's regex pattern engine boosts exact error code matches directly into the Reciprocal Rank Fusion stage, ensuring diagnostic codes never get diluted by dense embedding similarity."*

### Action B (Out-of-Scope Intent Refusal):
1. Type:
   ```text
   What is the recipe for homemade chocolate chip cookies?
   ```
2. Point out the answer:
   `"I apologize, but this question is outside the scope of the Aurora AX3000 Wi-Fi 6 Router User Manual. I can only assist with router setup, Wi-Fi configuration, hardware specifications, and troubleshooting."`
> *"Notice the Intent Classification module: our scratch Multinomial Naive Bayes classifier identified 'out_of_scope' with 100% confidence, triggering polite refusal without wasting reader compute or hallucinating."*

---

## Act 5: Voice Interaction (Speech-to-Text & Spoken Audio) (Minute 4:15)

### Action:
1. Click the **Microphone (Voice)** button.
2. Speak clearly into the microphone:
   ```text
   How do I enable the guest network?
   ```
3. Click the **Stop** button.
4. Listen to the spoken audio response synthesized via the browser's speech synthesis engine.
> *"For voice queries, faster-whisper (base.en, int8 CPU) transcribes the speech locally. The assistant reply normalizes technical units (e.g., expanding '5 GHz' to '5 gigahertz') so the text-to-speech output sounds natural."*

---

## Act 6: The Method Evolution Lab & Evaluation Page (Minute 4:45)

### Action:
1. Click **Method Evolution** in the top navigation bar.
2. Type `How do I troubleshoot internet connection errors?` and click **Compare Across Eras**.
3. Walk the examiner through the 5 eras running live side-by-side:
   - **Era 1 (Rules):** Regex keyword match.
   - **Era 2 (Statistical):** BM25 sparse lexical matching.
   - **Era 3 (Embeddings):** Word2Vec average vector cosine similarity.
   - **Era 4 (Transformers):** Dense semantic embeddings (`all-MiniLM-L6-v2`).
   - **Era 5 (Modern Hybrid RRF):** Reciprocal Rank Fusion combining dense, sparse, and exact code boosts.
4. Click **Evaluation** in the navigation bar to display the full benchmark dashboard:
   - Show the **100.0% Recall@5** on Hybrid Retrieval.
   - Show the **Confusion Matrices** comparing scratch Naive Bayes (82.9% accuracy) vs Scikit-learn Logistic Regression.
   - Show the **POS Tagger Comparison** and explain why general BERT fails on domain NER (0.0% F1) while our domain system achieves 100%.

---

## 5-Minute Wrap-Up Statement to the Examiner
> *"ManualMind demonstrates that modern conversational AI does not need to be a black-box API wrapper. By systematically building each syllabus concept — from Levenshtein DP and HMM Viterbi to Clause Decomposition, Ambiguity Guards, and Hybrid RRF — we achieve an offline, production-grade customer support assistant that runs in 527 milliseconds on pure CPU and rigorously adheres to the theoretical principles of NLP."*

