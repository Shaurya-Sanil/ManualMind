/**
 * ManualMind Application Core Controller (web/js/app.js)
 * Manages tab switching, theme toggling, active manual state, and global utilities.
 */

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initNavigation();
  fetchHealthAndActiveManual();
});

// Theme Management (Light / Dark Mode with localStorage persistence)
function initTheme() {
  const toggleBtn = document.getElementById("theme-toggle");
  const currentTheme = localStorage.getItem("manualmind-theme") || "light";
  document.documentElement.setAttribute("data-theme", currentTheme);
  toggleBtn.textContent = currentTheme === "dark" ? "☀️" : "🌙";

  toggleBtn.addEventListener("click", () => {
    const active = document.documentElement.getAttribute("data-theme");
    const next = active === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("manualmind-theme", next);
    toggleBtn.textContent = next === "dark" ? "☀️" : "🌙";
  });
}

// Navigation Router between Pages
function initNavigation() {
  const navButtons = document.querySelectorAll(".nav-btn");
  const pages = document.querySelectorAll(".page-view");

  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetPageId = btn.getAttribute("data-page");

      navButtons.forEach(b => b.classList.remove("active"));
      pages.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPage = document.getElementById(targetPageId);
      if (targetPage) {
        targetPage.classList.add("active");
      }

      // Trigger page specific activations
      if (targetPageId === "evaluation-page") {
        if (window.loadEvaluationResults) window.loadEvaluationResults();
      } else if (targetPageId === "manuals-page") {
        if (window.loadManualsList) window.loadManualsList();
      }
    });
  });
}

// Global API Endpoint Resolver (supports local, Vercel proxy rewrite, Render hosted, or custom ?backend=URL)
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get("backend")) {
  localStorage.setItem("manualmind-backend-url", urlParams.get("backend").replace(/\/+$/, ""));
}

// Default production Render backend fallback
const DEFAULT_REMOTE_BACKEND = "https://manualmind-backend.onrender.com";

// Auto-configure for production hosting (e.g. Vercel) without prompting the user
const isLocalEnv = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.hostname === "";
const isRenderHosted = window.location.hostname.includes("onrender.com");

if (!localStorage.getItem("manualmind-backend-url") && !isLocalEnv && !isRenderHosted) {
  // Running on Vercel or external domain: route directly to Render backend
  localStorage.setItem("manualmind-backend-url", DEFAULT_REMOTE_BACKEND);
}

window.API_BASE = localStorage.getItem("manualmind-backend-url") || "";
window.apiUrl = function(path) {
  if (!path.startsWith("/")) path = "/" + path;
  return `${window.API_BASE}${path}`;
};

// Universal fetch interceptor for remote backend deployments (e.g. Vercel)
const _nativeFetch = window.fetch;
window.fetch = function(input, init) {
  if (typeof input === "string" && input.startsWith("/api") && window.API_BASE) {
    input = `${window.API_BASE}${input}`;
  }
  return _nativeFetch(input, init);
};

// Fetch Active Manual from Backend Health Probe
async function fetchHealthAndActiveManual() {
  const pill = document.getElementById("active-manual-name");
  const dot = document.querySelector(".indicator-dot");

  try {
    const res = await fetch(window.apiUrl("/api/health"));
    if (res.ok) {
      const data = await res.json();
      if (pill && data.active_manual) {
        pill.textContent = data.active_manual;
        pill.title = "Connected to live NLP backend";
        if (dot) dot.style.background = "var(--success)";
      }
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    console.warn("ManualMind backend health check waiting: ", err);
    if (pill) {
      pill.textContent = "aurora_ax3000_manual.pdf";
      pill.title = "Connecting to backend service (Render takes ~30s if waking up)...";
      if (dot) dot.style.background = "var(--warning)";
    }
  }
}

