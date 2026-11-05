/**
 * Manual Documents & Ingestion Controller (web/js/manuals.js)
 * Handles PDF, TXT, and Markdown upload, ingestion statistics,
 * and active manual switching.
 */

async function loadManualsList() {
  const container = document.getElementById("manuals-list-grid");
  if (!container) return;

  try {
    const res = await fetch("/api/manuals");
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();

    container.innerHTML = "";
    data.manuals.forEach(m => {
      const card = document.createElement("div");
      card.className = "era-card";
      card.style.background = m.is_active ? "rgba(0, 119, 182, 0.06)" : "var(--bg-surface)";
      card.style.border = m.is_active ? "2px solid var(--accent)" : "1px solid var(--border-color)";

      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h4 style="font-size:1.05rem;">${m.name}</h4>
          ${m.is_active ? '<span class="badge" style="background:var(--success); color:#fff;">ACTIVE</span>' : ''}
        </div>
        <div style="font-size:0.8rem; color:var(--text-muted); font-family:var(--font-mono);">${m.filename}</div>
        
        <div style="margin: 0.75rem 0; font-size:0.85rem; display:flex; flex-direction:column; gap:0.25rem;">
          <div>📄 <strong>Pages:</strong> ${m.total_pages || "N/A"}</div>
          <div>📑 <strong>Sections:</strong> ${m.total_sections || "N/A"}</div>
          <div>🧩 <strong>Chunks:</strong> ${m.total_chunks || "N/A"}</div>
          <div>🔤 <strong>Vocabulary:</strong> ${m.vocabulary_size ? m.vocabulary_size.toLocaleString() + ' unique terms' : "N/A"}</div>
          <div>💾 <strong>File Size:</strong> ${(m.file_size_bytes / 1024).toFixed(1)} KB</div>
        </div>

        <div style="margin-top:auto; padding-top:0.5rem;">
          ${!m.is_active ? `<button class="action-btn primary" style="width:100%; justify-content:center; padding:0.4rem;" onclick="activateManual('${m.filename}')">Set Active Manual</button>` : '<div style="font-size:0.8rem; color:var(--success); font-weight:700; text-align:center;">Currently Active</div>'}
        </div>
      `;
      container.appendChild(card);
    });

  } catch (err) {
    container.innerHTML = `<div style="color:var(--danger)">Failed to load manuals: ${err.message}</div>`;
  }
}

async function uploadManualFile() {
  const fileInput = document.getElementById("manual-file-input");
  const statusDiv = document.getElementById("upload-status");
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    statusDiv.style.color = "var(--danger)";
    statusDiv.textContent = "Please select a file to upload.";
    return;
  }

  const file = fileInput.files[0];
  statusDiv.style.color = "var(--accent)";
  statusDiv.textContent = `Uploading and ingesting '${file.name}'...`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/manuals", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Upload error ${res.status}`);
    }

    statusDiv.style.color = "var(--success)";
    statusDiv.textContent = `Success: Ingested '${file.name}' (${data.pages} pages, ${data.chunks} chunks, ${data.vocabulary_size} terms).`;
    fileInput.value = "";
    loadManualsList();
    fetchHealthAndActiveManual();
  } catch (err) {
    statusDiv.style.color = "var(--danger)";
    statusDiv.textContent = `Upload failed: ${err.message}`;
  }
}

async function activateManual(filename) {
  try {
    const res = await fetch(`/api/manuals/${encodeURIComponent(filename)}/activate`, {
      method: "POST"
    });
    if (res.ok) {
      loadManualsList();
      fetchHealthAndActiveManual();
    }
  } catch (err) {
    alert(`Could not switch manual: ${err.message}`);
  }
}

window.loadManualsList = loadManualsList;
