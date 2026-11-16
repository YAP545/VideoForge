/**
 * VideoForge – main client JS
 * Handles: language switching, nav mobile, flash auto-dismiss
 */

// ── Language switching ──────────────────────────────────────────────────────
const LANG_KEY = "vf_lang";

function getLang() {
  return localStorage.getItem(LANG_KEY) || "en";
}

function setLang(lang) {
  localStorage.setItem(LANG_KEY, lang);
  // Re-request the current page with lang param so server renders correct strings
  const url = new URL(window.location.href);
  url.searchParams.set("lang", lang);
  window.location.href = url.toString();
}

// Sync button label on load
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("lang-btn");
  if (!btn) return;
  // The button label is already set server-side; just wire up click
  btn.addEventListener("click", () => {
    const current = getLang();
    setLang(current === "en" ? "hi" : "en");
  });
});

// ── Flash auto-dismiss ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".flash[data-autodismiss]").forEach((el) => {
    setTimeout(() => {
      el.style.transition = "opacity .4s";
      el.style.opacity = "0";
      setTimeout(() => el.remove(), 400);
    }, 4000);
  });
});

// ── Mobile nav toggle ───────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("nav-toggle");
  const menu   = document.getElementById("nav-menu");
  if (!toggle || !menu) return;
  toggle.addEventListener("click", () => {
    const open = menu.classList.toggle("nav__menu--open");
    toggle.setAttribute("aria-expanded", open);
  });
});

// ── Utility: simple form validation highlight ──────────────────────────────
function markInvalid(inputEl, msg) {
  inputEl.classList.add("input--error");
  let err = inputEl.parentElement.querySelector(".form-error");
  if (!err) {
    err = document.createElement("p");
    err.className = "form-error";
    inputEl.parentElement.appendChild(err);
  }
  err.textContent = msg;
}
function clearInvalid(inputEl) {
  inputEl.classList.remove("input--error");
  const err = inputEl.parentElement.querySelector(".form-error");
  if (err) err.remove();
}
