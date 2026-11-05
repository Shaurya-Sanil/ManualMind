/**
 * Method Evolution Lab Controller (web/js/evolution.js)
 * Executes the same question across 5 distinct historical paradigms of NLP:
 * Era 1: Rule-Based (1960s–1980s)
 * Era 2: Statistical & N-Grams (1990s–2000s)
 * Era 3: Distributed Word Embeddings (2013–2017)
 * Era 4: Contextualized Transformers (2018–2022)
 * Era 5: Modern Hybrid Ensemble & Synthesis (2023–Present)
 */

async function runMethodEvolution() {
  const query = document.getElementById("evolution-query-input").value.trim();
  if (!query) return;

  const grid = document.getElementById("evolution-cards-grid");
  grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: var(--text-muted);">Running question through 5 historical paradigms of NLP...</div>`;

  try {
    const formData = new FormData();
    formData.append("query", query);

    const res = await fetch("/api/evolution", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
    const data = await res.json();

    // Render 5 Era Cards
    grid.innerHTML = "";
    data.eras.forEach(era => {
      const card = document.createElement("div");
      card.className = "era-card";
      card.innerHTML = `
        <span class="era-badge">ERA ${era.era_id}</span>
        <h4>${era.era_name.split(":")[1]?.trim() || era.era_name}</h4>
        <div style="font-size: 0.8rem; color: var(--text-muted);">${era.description}</div>
        <div style="margin: 0.5rem 0; font-size: 0.88rem; flex: 1;">
          <div style="font-size: 0.75rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Retrieved Section:</div>
          <div style="color: var(--accent); font-weight: 600; margin-bottom: 0.4rem;">${era.retrieved_section}</div>
          <div style="font-size: 0.75rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Answer Extract:</div>
          <div style="color: var(--text-primary); line-height: 1.4;">${era.answer_text}</div>
        </div>
        <div class="metric-row">
          <span>Score: <strong>${Math.round(era.confidence_score * 100)}%</strong></span>
          <span>Latency: <strong>${era.latency_ms} ms</strong></span>
        </div>
      `;
      grid.appendChild(card);
    });

    // Render Historical Accuracy Comparison Chart
    renderEvolutionChart(data.historical_benchmark_accuracies);

  } catch (err) {
    grid.innerHTML = `<div style="grid-column: 1 / -1; color: var(--danger); text-align: center;">Error running evolution comparison: ${err.message}</div>`;
  }
}

function renderEvolutionChart(accuracies) {
  const container = document.getElementById("evolution-chart-container");
  if (!container || !accuracies) return;

  const entries = Object.entries(accuracies);
  const width = 800;
  const height = 220;
  const paddingLeft = 140;
  const barHeight = 24;
  const gap = 16;

  let svg = `<svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; font-family: -apple-system, sans-serif;">`;

  entries.forEach(([eraName, acc], idx) => {
    const y = 20 + idx * (barHeight + gap);
    const barWidth = (acc / 100) * (width - paddingLeft - 80);
    const color = idx === 4 ? "#2A9D8F" : (idx === 3 ? "#134074" : "#457B9D");

    // Era label
    svg += `<text x="${paddingLeft - 10}" y="${y + 16}" text-anchor="end" font-size="12" font-weight="600" fill="currentColor">${eraName}</text>`;
    // Background track
    svg += `<rect x="${paddingLeft}" y="${y}" width="${width - paddingLeft - 80}" height="${barHeight}" rx="4" fill="rgba(150, 150, 150, 0.15)"/>`;
    // Value bar
    svg += `<rect x="${paddingLeft}" y="${y}" width="${barWidth}" height="${barHeight}" rx="4" fill="${color}"/>`;
    // Percentage label
    svg += `<text x="${paddingLeft + barWidth + 8}" y="${y + 16}" font-size="12" font-weight="700" fill="${color}">${acc}%</text>`;
  });

  svg += `</svg>`;
  container.innerHTML = svg;
}
