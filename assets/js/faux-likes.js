(function(){
  // --- seeded hash + PRNG for per-post parameters (no library needed) ---
  function hash32(str){
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++){
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    // final avalanche
    h += h << 13; h ^= h >>> 7; h += h << 3; h ^= h >>> 17; h += h << 5;
    return h >>> 0;
  }
  function mulberry32(seed){
    return function(){
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Deterministic, non-decreasing baseline based on publish date + now
  function computeBaseline(id, pubISO, now){
    const nowMs = now.getTime();
    const pub = new Date(pubISO + "T00:00:00Z"); // day precision
    if (isNaN(pub)) return 0;
    const ageDays = Math.max(0, (nowMs - pub.getTime()) / 86400000);

    // Seeded params per post
    const seed = hash32(id + "|" + pubISO);
    const rnd = mulberry32(seed);

    // Start between 12..120
    const start = 12 + Math.floor(rnd() * 109);

    // Growth between 0.8..6.2 per day
    const gpd = 0.8 + rnd() * 5.4;

    // Smooth intra-day growth so numbers tick during the day but never fall
    const dayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const fracDay = Math.max(0, (nowMs - dayStart) / 86400000); // 0..1 within the day

    const val = start + gpd * (Math.floor(ageDays) + Math.min(1, fracDay));
    return Math.floor(val); // monotonic across time
  }

  // Local +1 on click (one per browser) to give a “reactive” feel
  function storageKey(id){ return "faux-like:" + id; }
  function getLocalClicks(id){
    try { return parseInt(localStorage.getItem(storageKey(id)) || "0", 10); }
    catch(e){ return 0; }
  }
  function setLocalClicks(id, n){
    try { localStorage.setItem(storageKey(id), String(n)); } catch(e){}
  }
  function isAlreadyLiked(id){ return getLocalClicks(id) > 0; }

  function format(n){ return n.toLocaleString(undefined); }

  function initButton(btn){
    const id = btn.getAttribute("data-id") || "";
    const pub = btn.getAttribute("data-pub") || "";
    if (!id || !pub) return;

    const countEl = btn.querySelector(".like-count");
    const now = new Date();

    const baseline = computeBaseline(id, pub, now);
    const local = getLocalClicks(id);
    const total = baseline + local;

    if (countEl) countEl.textContent = format(total);
    if (local > 0){
      btn.classList.add("liked");
      btn.setAttribute("aria-pressed", "true");
    }

    // On click: apply a single local +1 (per browser)
    btn.addEventListener("click", ()=>{
      if (isAlreadyLiked(id)) return;
      const cur = parseInt((countEl?.textContent || "0").replace(/[^\d]/g,"") || "0", 10);
      const next = cur + 1;
      if (countEl) countEl.textContent = format(next);
      setLocalClicks(id, 1);
      btn.classList.add("liked");
      btn.setAttribute("aria-pressed", "true");
      // Optional micro animation
      btn.style.transform = "translateY(-1px) scale(1.02)";
      setTimeout(()=>{ btn.style.transform = ""; }, 120);
    });
  }

  document.addEventListener("DOMContentLoaded", function(){
    document.querySelectorAll(".like-btn").forEach(initButton);
  });

  // Optional: periodic tick during session (still monotonic)
  setInterval(()=>{
    const now = new Date();
    document.querySelectorAll(".like-btn").forEach(btn=>{
      const id = btn.getAttribute("data-id") || "";
      const pub = btn.getAttribute("data-pub") || "";
      const countEl = btn.querySelector(".like-count");
      if (!id || !pub || !countEl) return;
      const base = computeBaseline(id, pub, now);
      const local = getLocalClicks(id);
      const shown = parseInt((countEl.textContent||"0").replace(/[^\d]/g,""), 10) || 0;
      const next = Math.max(shown, base + local); // never decrease
      if (next !== shown) countEl.textContent = next.toLocaleString(undefined);
    });
  }, 60 * 1000);
})();
