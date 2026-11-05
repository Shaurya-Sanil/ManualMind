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

// Fetch Active Manual from Backend Health Probe
async function fetchHealthAndActiveManual() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      const pill = document.getElementById("active-manual-name");
      if (pill && data.active_manual) {
        pill.textContent = data.active_manual;
      }
    }
  } catch (err) {
    console.warn("ManualMind backend health check waiting: ", err);
  }
}
