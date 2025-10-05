(function(){
  /* =========================
     Config + Data
  ========================== */
  const RBX_ROTATE_MS = 2600;
  const RBX_FADE_MS   = 450;

  const RBX_CAPTIONS = [
    "Accelerated 120+ Databricks notebooks delivering 40% faster data pipelines",
    "Optimized Azure Data Factory jobs saving 1,200+ compute hours annually",
    "Built 250+ complex Python projects reducing manual processes by 65%",
    "Migrated 180+ Power BI dashboards improving decision-making across 4,000+ users",
    "Delivered 95% faster PL/SQL queries by tuning mission-critical healthcare systems",
    "Deployed 130+ Tableau dashboards for retail with 50% more user adoption",
    "Reduced ETL runtime by 70% with Informatica optimization across finance data",
    "Engineered AI models with 92% prediction accuracy in healthcare diagnostics",
    "Automated 300+ accounting workflows saving 5,000+ staff hours yearly",
    "Scaled credit risk scoring AI model to process 25M+ transactions daily",
    "Trained 110+ machine learning models improving fraud detection accuracy by 18%"
    // ... keep the rest of your captions here
  ];

  const RBX_METRICS = [
    { v: "100+", c: "AI Models trained" },
    { v: "300+", c: "Dashboards created" },
    { v: "200+", c: "Data flows & pipelines" },
    { v: "80+",  c: "Databricks" },
    { v: "80+",  c: "Warehouses designed" }
  ];
  const RBX_METRICS_INTERVAL = 2400;
  const RBX_METRICS_FADE     = 350;

  /* =========================
     Helpers
  ========================== */
  function rbxFillBars(){
    document.querySelectorAll(".rbx-bar-fill").forEach(el=>{
      const v = el.getAttribute("data-rbx-value");
      if(!v) return;
      requestAnimationFrame(()=>{ el.style.width = v + "%"; });
    });
  }

  // === Bold numbers + units in captions (robust) ===
  const RBX_CAP_UNIT_WORDS =
    '(?:hours?|hrs?|hr|minutes?|mins?|min|seconds?|secs?|sec|ms|days?|day|yrs?|years?|transactions?|txns?|records?|rows?|reports?|models?|pipelines?|dashboards?|stores?|users?|hospitals?|accounts?|GB|TB|MB|KB|B|gb|tb|mb|kb|b|h|m|s)';

  // A) number with %, or K/M/B suffix, optional +
  // B) number (optional +) then optional one-word descriptor then a unit word
  const RBX_CAP_REGEX = new RegExp(
  [
    // number + optional decimal + optional comma grouping + optional spaces + %
    String.raw`(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*%`,  

    // number + optional decimal + K/M/B (case-insensitive), optional +
    String.raw`(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?[kmbKMB]\+?`,  

    // number + optional + + optional word + unit
    String.raw`(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\+?(?:\s+\w{1,24})?\s+` + RBX_CAP_UNIT_WORDS
  ].join('|'),
  'g'
);

  function rbxFormatCaption(text){
    return text.replace(RBX_CAP_REGEX, m => `<strong>${m}</strong>`);
  }

  /* =========================
     Caption rotation (single span)
  ========================== */
  function rbxRotateCaption(){
    const el = document.getElementById("rbx-rotating-caption");
    if(!el || !Array.isArray(RBX_CAPTIONS) || RBX_CAPTIONS.length === 0) return;

    let i = 0;
    el.innerHTML = rbxFormatCaption(RBX_CAPTIONS[i]); // initial paint

    function next(){
      el.classList.add("rbx-cap-out");
      setTimeout(()=>{
        i = (i + 1) % RBX_CAPTIONS.length;
        el.innerHTML = rbxFormatCaption(RBX_CAPTIONS[i]); // IMPORTANT: innerHTML
        el.classList.remove("rbx-cap-out");
        el.classList.add("rbx-cap-in");
        setTimeout(()=> el.classList.remove("rbx-cap-in"), RBX_FADE_MS + 20);
      }, RBX_FADE_MS);
    }

    let timer = setInterval(next, RBX_ROTATE_MS);
    el.addEventListener("mouseenter", ()=> clearInterval(timer));
    el.addEventListener("mouseleave", ()=> timer = setInterval(next, RBX_ROTATE_MS));
    el.addEventListener("focus", ()=> clearInterval(timer), true);
    el.addEventListener("blur",  ()=> timer = setInterval(next, RBX_ROTATE_MS), true);
  }

  /* =========================
     Metrics: rotate 2 at a time
  ========================== */
  function rbxRotateMetrics(){
    const row = document.getElementById("rbx-metrics-row");
    const m1v = document.getElementById("rbx-m1-val");
    const m1c = document.getElementById("rbx-m1-cap");
    const m2v = document.getElementById("rbx-m2-val");
    const m2c = document.getElementById("rbx-m2-cap");
    const ind = document.getElementById("rbx-metrics-ind");
    if(!row || !m1v || !m1c || !m2v || !m2c || !ind) return;

    let i = 0; // left index of the pair

    function render(){
      const j = (i + 1) % RBX_METRICS.length;
      row.classList.add("rbx-fade-out");
      setTimeout(()=>{
        m1v.textContent = RBX_METRICS[i].v;
        m1c.textContent = RBX_METRICS[i].c;
        m2v.textContent = RBX_METRICS[j].v;
        m2c.textContent = RBX_METRICS[j].c;
        ind.textContent = (i+1) + "/" + (j+1);
        row.classList.remove("rbx-fade-out");
        i = j;
      }, RBX_METRICS_FADE);
    }

    render();
    let timer = setInterval(render, RBX_METRICS_INTERVAL);
    row.addEventListener("mouseenter", ()=> clearInterval(timer));
    row.addEventListener("mouseleave", ()=> timer = setInterval(render, RBX_METRICS_INTERVAL));
  }

  /* =========================
     Overlay control
  ========================== */
  function rbxOverlayControl(){
    const overlay = document.querySelector(".rbx-intro-overlay");
    const continueBtn = document.getElementById("rbx-continue");
    if(!overlay || !continueBtn) return;

    document.body.style.overflow = "hidden"; // lock scroll while visible

    continueBtn.addEventListener("click", function(e){
      e.preventDefault();
      overlay.classList.add("rbx-fade-out");
      setTimeout(()=>{
        overlay.classList.add("rbx-hidden");
        document.body.style.overflow = "auto";
      }, 280);
    }, { once: true });
  }

  /* =========================
     Init
  ========================== */
  document.addEventListener("DOMContentLoaded", function(){
    rbxFillBars();
    rbxRotateCaption();
    rbxRotateMetrics();
    rbxOverlayControl();
  });
})();