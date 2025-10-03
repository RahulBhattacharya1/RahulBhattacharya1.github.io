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

  // === Impact tab bar animation (replay on tab switch and on open) ===
const impactTab   = root.querySelector('#tab-impact');
const impactPanel = root.querySelector('#panel-impact');
const toggle      = root.querySelector('#satinA');       // the Bio expand checkbox
const panel       = root.querySelector('#profile-panel'); // the collapsible container

if (impactTab && impactPanel) {
  const animateImpactBars = () => {
    impactPanel.querySelectorAll('.fill').forEach(el => {
      const target = getComputedStyle(el).getPropertyValue('--w').trim() || '0%';
      el.style.width = '0';        // reset
      void el.offsetWidth;         // force reflow
      requestAnimationFrame(() => { el.style.width = target; }); // animate to target
    });
  };

  // Animate when switching into Impact while the panel is already open
  impactTab.addEventListener('change', () => {
    if (impactTab.checked) animateImpactBars();
  });

  // Animate when opening Bio with Impact already selected (wait for expand transition)
  if (toggle && panel) {
    toggle.addEventListener('change', () => {
      if (!toggle.checked || !impactTab.checked) return;
      const onEnd = ev => {
        if (ev.propertyName !== 'max-height') return;
        panel.removeEventListener('transitionend', onEnd);
        animateImpactBars();
      };
      panel.addEventListener('transitionend', onEnd, { once: true });
    });
  }

  // If page loads with Impact already selected and Bio open
  if (impactTab.checked && (!toggle || toggle.checked)) {
    requestAnimationFrame(animateImpactBars);
  }
}

})();
