/* Faux Likes — reusable, backend-free, monotonic, +1 per browser */
(function () {
  // ---------- Config (tweak without touching the rest) ----------
  const CFG = {
    selector: ".btn--like",   // buttons to auto-init
    attrId: "data-like-id",   // per-post id (slug/url)
    attrPub: "data-like-pub", // YYYY-MM-DD publish date
    showCount: true,          // hide by setting false
    tickMs: 60000,            // gentle within-day tick
    storagePrefix: "faux-like:",
    clampMonotonic: true,     // never show lower than last seen
    // Growth configuration (deterministic per post)
    startMin: 12, startMax: 120,     // inclusive-ish range for start
    gpdMin: 0.8, gpdMax: 6.2,        // growth per day
    useUTC: false,                   // set true for the same day boundary worldwide
    // Optional: fixed growth override, e.g., 2.6
    fixedGrowthPerDay: null          // number | null
  };

  // ---------- Utilities ----------
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $all(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }
  function escNum(s) { return parseInt(String(s).replace(/[^\d]/g, "") || "0", 10) || 0; }
  function fmt(n) { return Number(n).toLocaleString(); }
  function sk(id, suffix) { return CFG.storagePrefix + id + (suffix ? (":" + suffix) : ""); }
  function getLS(k, d = "0") { try { return localStorage.getItem(k) ?? d; } catch { return d; } }
  function setLS(k, v) { try { localStorage.setItem(k, String(v)); } catch {} }

  // FNV-like hash + mulberry32 PRNG (deterministic)
  function hash32(str) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
    h += h << 13; h ^= h >>> 7; h += h << 3; h ^= h >>> 17; h += h << 5;
    return h >>> 0;
  }
  function mulberry32(seed) {
    return function () {
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- Baseline model (monotonic over time) ----------
  function dayStartMs(d) {
    if (CFG.useUTC) return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
    return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  }

  function baselineFor(id, pubISO, now) {
    if (!pubISO) return 0;
    const pub = CFG.useUTC ? new Date(pubISO + "T00:00:00Z") : new Date(pubISO);
    if (isNaN(pub)) return 0;

    const ageDays = Math.max(0, (now.getTime() - pub.getTime()) / 86400000);

    // Per-post deterministic params
    const seed = hash32(id + "|" + pubISO);
    const rnd  = mulberry32(seed);

    const start = CFG.startMin + Math.floor(rnd() * (CFG.startMax - CFG.startMin + 1));
    const gpd   = (CFG.fixedGrowthPerDay != null)
      ? CFG.fixedGrowthPerDay
      : (CFG.gpdMin + rnd() * (CFG.gpdMax - CFG.gpdMin));

    // Smooth within-day growth (still non-decreasing after floor)
    const frac = Math.max(0, (now.getTime() - dayStartMs(now)) / 86400000);
    const val  = start + gpd * (Math.floor(ageDays) + Math.min(1, frac));
    return Math.floor(val);
  }

  // ---------- Per-button logic ----------
  function initButton(btn) {
    const id  = btn.getAttribute(CFG.attrId) || "";
    const pub = btn.getAttribute(CFG.attrPub) || "";
    if (!id || !pub) return;

    const countEl = $(".count", btn);
    if (!countEl && CFG.showCount) return;

    const likedKey = sk(id, "liked");
    const maxKey   = sk(id, "maxShown");

    const getLocal = () => escNum(getLS(likedKey, "0"));
    const setLocal = v  => setLS(likedKey, v);
    const getMax   = () => escNum(getLS(maxKey, "0"));
    const setMax   = v  => setLS(maxKey, v);

    const apply = (n) => {
      const val = CFG.clampMonotonic ? Math.max(n, getMax()) : n;
      if (CFG.showCount) countEl.textContent = val > 0 ? fmt(val) : "";
      if (CFG.clampMonotonic && val > getMax()) setMax(val);
    };

    // First paint
    const now  = new Date();
    const base = baselineFor(id, pub, now);
    const loc  = getLocal();
    apply(base + loc);

    if (loc > 0) { btn.classList.add("liked"); btn.setAttribute("aria-pressed", "true"); }

    // Click = one local +1 per browser
    btn.addEventListener("click", () => {
      if (getLocal() > 0) return;
      setLocal(1);
      btn.classList.add("liked");
      btn.setAttribute("aria-pressed", "true");
      apply(baselineFor(id, pub, new Date()) + 1);
    });

    // Gentle within-day tick; never decreases
    if (CFG.tickMs > 0) {
      setInterval(() => {
        const b = baselineFor(id, pub, new Date());
        apply(b + getLocal());
      }, CFG.tickMs);
    }
  }

  // ---------- Auto-init ----------
  document.addEventListener("DOMContentLoaded", () => {
    $all(CFG.selector).forEach(initButton);
  });

  // Optional: expose a tiny API if you ever need manual init
  window.FauxLikes = {
    initAll: () => $all(CFG.selector).forEach(initButton),
    config: CFG
  };
})();
