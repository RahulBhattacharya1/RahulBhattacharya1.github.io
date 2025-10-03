// Add CSS only for mobile
(function ensureMobileCSS() {
  const href = "/assets/css/mobile-profile.css";
  if (![...document.styleSheets].some(s => s.href && s.href.endsWith("mobile-profile.css"))) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.media = "(max-width: 720px)";
    link.href = href;
    document.head.appendChild(link);
  }
})();

// Fetch and inject the HTML partial into the placeholder
(async function mountMobileProfile() {
  const root = document.getElementById("mobile-profile-root");
  if (!root) return;
  try {
    const res = await fetch("/mobile-profile.html", { credentials: "same-origin" });
    if (!res.ok) return;
    const html = await res.text();
    root.innerHTML = html;

    // Optional: lightweight behaviors — keep ARIA in sync with the new structure
    const toggle = root.querySelector("#satinA");
    const card  = root.querySelector('label.mp-card[for="satinA"]') || root.querySelector('label[for="satinA"]');
    const panel = root.querySelector("#profile-panel");
    if (toggle && card) {
      const sync = () => {
        card.setAttribute("aria-expanded", String(toggle.checked));
        if (panel) panel.setAttribute("aria-hidden", String(!toggle.checked));
      };
      toggle.addEventListener("change", sync);
      sync();
    }
    root.querySelectorAll(".chip,.btn").forEach(el => {
      el.addEventListener("click", e => {
        if (el.getAttribute("href") === "#") e.preventDefault();
        el.style.transform = "translateY(-1px) scale(1.02)";
        setTimeout(() => { el.style.transform = ""; }, 180);
      });
    });
  } catch (e) {
    // Silently ignore on failure to keep desktop lean
  }
})();
