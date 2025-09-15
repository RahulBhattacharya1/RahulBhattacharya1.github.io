/* Faux Share Count — monotonic baseline + local +1 on successful share/copy */
(function () {
  // ---- config (tweak if you want a different pace than Like) ----
  const CFG = {
    selector: ".share-btn[data-share-id]",
    startMin: 10, startMax: 90,       // initial range
    gpdMin: 0.6, gpdMax: 4.0,         // growth per day
    tickMs: 60000,                    // gentle within-day tick
    useUTC: false                     // flip true for a global day boundary
  };

  // ---- utils ----
  function escNum(s){ return parseInt(String(s).replace(/[^\d]/g,"")||"0",10)||0; }
  function fmt(n){ return Number(n).toLocaleString(); }
  function hash32(str){
    let h=2166136261>>>0;
    for(let i=0;i<str.length;i++){ h^=str.charCodeAt(i); h=Math.imul(h,16777619); }
    h+=h<<13; h^=h>>>7; h+=h<<3; h^=h>>>17; h+=h<<5;
    return h>>>0;
  }
  function mulberry32(seed){ return function(){
    let t=seed+=0x6D2B79F5;
    t=Math.imul(t^(t>>>15), t|1);
    t^=t+Math.imul(t^(t>>>7), t|61);
    return ((t^(t>>>14))>>>0)/4294967296;
  };}

  function dayStartMs(d){
    return CFG.useUTC
      ? Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate())
      : new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  }

  // different seed suffix so Like/Share baselines don't match
  function baselineFor(id, pubISO, now){
    if(!pubISO) return 0;
    const pub = CFG.useUTC ? new Date(pubISO+"T00:00:00Z") : new Date(pubISO);
    if(isNaN(pub)) return 0;

    const ageDays = Math.max(0, (now.getTime() - pub.getTime())/86400000);
    const seed = hash32(id + "|" + pubISO + "|share");
    const rnd  = mulberry32(seed);
    const start = CFG.startMin + Math.floor(rnd()*(CFG.startMax - CFG.startMin + 1));
    const gpd   = CFG.gpdMin + rnd()*(CFG.gpdMax - CFG.gpdMin);

    const frac = Math.max(0, (now.getTime() - dayStartMs(now))/86400000);
    return Math.floor(start + gpd*(Math.floor(ageDays) + Math.min(1, frac)));
  }

  function storageKey(id){ return "faux-share:"+id; }
  function getLocal(id){ try{return parseInt(localStorage.getItem(storageKey(id))||"0",10);}catch{return 0;} }
  function setLocal(id,v){ try{localStorage.setItem(storageKey(id), String(v));}catch{} }

  async function doShare(url, title){
    // Prefer native share if available
    if (navigator.share) {
      try { await navigator.share({ title, url }); return true; } catch { return false; }
    }
    // Fallback: copy to clipboard
    try {
      await navigator.clipboard.writeText(url);
      return true;
    } catch {
      // ultra-fallback
      const ta = document.createElement("textarea");
      ta.value = url; document.body.appendChild(ta); ta.select();
      const ok = document.execCommand("copy"); document.body.removeChild(ta);
      return !!ok;
    }
  }

  function initShare(btn){
    const id  = btn.getAttribute("data-share-id") || "";
    const pub = btn.getAttribute("data-share-pub") || "";
    const countEl = btn.querySelector(".count");
    if(!id || !pub || !countEl) return;

    const now = new Date();
    const base = baselineFor(id, pub, now);
    const local = getLocal(id);
    const apply = n => { countEl.textContent = n>0 ? fmt(n) : ""; };

    apply(base + local);

    // gentle tick
    if (CFG.tickMs > 0) {
      setInterval(()=>{
        const b = baselineFor(id, pub, new Date());
        const displayed = escNum(countEl.textContent);
        const next = Math.max(displayed, b + getLocal(id));
        if (next !== displayed) apply(next);
      }, CFG.tickMs);
    }

    // click handler
    btn.addEventListener("click", async (e)=>{
      e.preventDefault();
      const url = btn.getAttribute("data-url") || window.location.href;
      const ok = await doShare(url, document.title);
      if (!ok) return;
      if (getLocal(id) > 0) return;      // only first share per browser
      setLocal(id, 1);
      apply(baselineFor(id, pub, new Date()) + 1);
      btn.classList.add("shared");       // optional hook if you want a style change
    }, { passive:false });
  }

  document.addEventListener("DOMContentLoaded", ()=>{
    document.querySelectorAll(CFG.selector).forEach(initShare);
  });
})();
