---
layout: default
title: "Rahul's Experience"
date: 2025-08-22
permalink: /experience/
thumbnail: /assets/images/impacts.webp
hide_thumbnail_in_body: true
comments: false
share: false
hide_date: true
featured: true
hide_top_title: true
custom_snippet: true
custom_snippet_text: "Role-based case studies with measurable outcomes from healthcare, retail, and finance."
---

{::nomarkdown}
<!-- ===== HERO ===== -->
<section class="impacts-hero">
  <div class="impacts-hero__inner">
    <h1>My Real-World Enterprise Impacts</h1>
    <p>Role-based case studies from retail, healthcare, and finance with measurable outcomes. All data anonymized.</p>
  </div>
</section>

<!-- ===== ROLE SUMMARY + MENU + PANELS ===== -->
<span id="role-summary" class="role-summary" aria-live="polite">Explore impacts by role.</span>
<br/>Choose a role:<br/><br/>

<div class="role-gallery">
  <nav class="role-menu" aria-label="Choose a role">
    <button class="role-btn active" data-role="analyst" aria-current="page">Data Analyst</button>
    <button class="role-btn" data-role="scientist">Data Scientist</button>
    <button class="role-btn" data-role="engineer">Data Engineer</button>
    <button class="role-btn" data-role="aiml">AI / ML Engineer</button>
    <button class="role-btn" data-role="genai">NLP / Gen AI Specialist</button>
    <button class="role-btn" data-role="businessanalyst">Business Analyst</button>
  </nav>

  <section class="role-slideshows">
    <!-- ========= Analyst ========= -->
    <div class="role-accordion" id="panel-analyst">
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-analyst-1" id="acc-analyst-1-label">
            UHG — BI Modernization (Tableau → Power BI)
            <span class="acc-meta">SQL Server · Power BI (RLS/DAX) · SSIS/SQL Agent</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-analyst-1" role="region" aria-labelledby="acc-analyst-1-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Migrated static Tableau reports to interactive Power BI with a governed KPI layer.</p>
            <details><summary>S · Situation</summary><p>Leadership needed fast, trusted clinical/pharmacy KPIs; legacy reports were static and slow.</p></details>
            <details><summary>T · Task</summary><p>Deliver a governed model with RLS, standardized metrics, and quick drilldowns.</p></details>
            <details><summary>A · Action</summary><p>Star schema; calc groups; incremental refresh; query tuning; SSIS ingestion; SQL Agent scheduling.</p></details>
            <details open><summary>R · Result</summary>
              <ul>
                <li><span class="kpi" data-target="50">0</span>%+ Tableau views migrated</li>
                <li>Significantly faster dashboard loads; higher exec adoption</li>
              </ul>
            </details>
            <p><strong>Stack:</strong> Power BI, SQL Server, DAX, SSIS, Tableau</p>
          </div>
        </div>
      </article>

      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-analyst-2" id="acc-analyst-2-label">
            Retail KPI & Cohort Analytics
            <span class="acc-meta">SQL · Power BI · Experimentation</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-analyst-2" role="region" aria-labelledby="acc-analyst-2-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Unified revenue, conversion, and retention with SSoT metrics and drilldowns.</p>
            <details><summary>S</summary><p>Conflicting metrics across teams led to slow decisions.</p></details>
            <details><summary>T</summary><p>Single source of truth with cohorts and A/B readouts.</p></details>
            <details><summary>A</summary><p>Window functions; cohorts; documented KPI contracts; semantic model.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li>Report assembly time cut from 6h → <span class="mono">30m</span></li>
                <li>Consistent KPIs improved trust and adoption</li>
              </ul>
            </details>
          </div>
        </div>
      </article>
    </div>

    <!-- ========= Scientist ========= -->
    <div class="role-accordion" id="panel-scientist" hidden>
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-scientist-1" id="acc-scientist-1-label">
            Walgreens — Workforce Forecasting at Scale
            <span class="acc-meta">Databricks · PySpark · ADF · Time-series</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-scientist-1" role="region" aria-labelledby="acc-scientist-1-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Modular time-series pipelines with automated refresh and seasonality.</p>
            <details><summary>S</summary><p>Manual, reactive scheduling across thousands of stores.</p></details>
            <details><summary>T</summary><p>Stabilize forecasts and reduce manual corrections.</p></details>
            <details><summary>A</summary><p>Feature engineering; hierarchical models; ADF triggers; staffing constraints.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li><span class="kpi" data-target="30">0</span>% reduction in manual intervention</li>
                <li>Improved stability; fewer late corrections</li>
              </ul>
            </details>
            <p><strong>Stack:</strong> Python, PySpark, Databricks, ADF, ADLS, SQL</p>
          </div>
        </div>
      </article>
    </div>

    <!-- ========= Engineer ========= -->
    <div class="role-accordion" id="panel-engineer" hidden>
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-engineer-1" id="acc-engineer-1-label">
            Delta Lake ELT Orchestration
            <span class="acc-meta">Databricks · Delta · ADF · PySpark</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-engineer-1" role="region" aria-labelledby="acc-engineer-1-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Bronze→Silver→Gold with CDC, data quality tests, and SLAs.</p>
            <details><summary>S</summary><p>Nightly delays; missed SLAs.</p></details>
            <details><summary>T</summary><p>Incremental ELT with reliability and lineage.</p></details>
            <details><summary>A</summary><p>Auto Loader; MergeInto; watermarking; expectations; monitoring.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li>Latency 24h → <span class="mono">30m</span></li>
                <li>On-time delivery at peaks</li>
              </ul>
            </details>
          </div>
        </div>
      </article>
    </div>

    <!-- ========= AI/ML, GenAI, BA panels (trimmed for brevity; keep your items) ========= -->
    <div class="role-accordion" id="panel-aiml" hidden></div>
    <div class="role-accordion" id="panel-genai" hidden></div>
    <div class="role-accordion" id="panel-businessanalyst" hidden></div>
  </section>
</div>

<!-- ===== EXPERIENCE FLOW (ARROW UNDERLAY) ===== -->
<section class="flow experience-flow" id="experience-flow">
  <svg class="flow-svg" id="flow-svg" aria-hidden="true"></svg>
  <div class="flow-grid" id="flow-grid">
    <!-- Use your resume roles here -->
    <article class="flow-card">
      <h3>Dec 2024 – Present · Deerfield, IL</h3>
      <h4>Senior Data Scientist — Walgreens (TCS)</h4>
      <p>Databricks forecasting; ADF automation; modular Python components.</p>
      <div class="meta"><span class="pill">Python</span><span class="pill">Databricks</span><span class="pill">ADF</span></div>
    </article>
    <article class="flow-card">
      <h3>Jul 2023 – Nov 2024 · Chicago, IL</h3>
      <h4>Sr. Data Analyst & Technical Architect — UnitedHealthcare (TCS)</h4>
      <p>Governed KPIs; Tableau → Power BI; SSIS & SQL Agent orchestration.</p>
      <div class="meta"><span class="pill">Power BI</span><span class="pill">Tableau</span><span class="pill">SSIS</span></div>
    </article>
    <article class="flow-card">
      <h3>Jul 2022 – May 2023 · Chicago, IL</h3>
      <h4>Data Analytics Lead — Bombardier (TCS)</h4>
      <p>Metrics for aero ops in Databricks; Python ETL; leadership insights.</p>
      <div class="meta"><span class="pill">Pandas</span><span class="pill">Databricks</span></div>
    </article>
    <article class="flow-card">
      <h3>Mar 2022 – Jun 2022 · Chicago, IL</h3>
      <h4>Data Analyst & Data Governance Lead — Galderma (TCS)</h4>
      <p>Automated KPI reporting; Databricks packages; 25%+ manual reduction.</p>
      <div class="meta"><span class="pill">Governance</span><span class="pill">Power BI</span></div>
    </article>
    <article class="flow-card">
      <h3>Oct 2017 – Feb 2022 · Deerfield, IL</h3>
      <h4>Data Analytics & EPM Lead — Walgreens (TCS)</h4>
      <p>Financial models for OKRs; forecasting; resolved ~75% process issues.</p>
      <div class="meta"><span class="pill">EPM</span><span class="pill">Forecasting</span></div>
    </article>
    <!-- Add older roles as needed -->
  </div>
</section>
{:/nomarkdown}

<style>
/* ---------- HERO ---------- */
.impacts-hero{ background:linear-gradient(135deg,#0b4dbf,#1182d6); color:#fff; border-radius:12px; margin:0 0 1rem }
.impacts-hero__inner{ padding:2rem 1.25rem; text-align:center; max-width:980px; margin:0 auto }
.impacts-hero h1{ margin:.2rem 0 .6rem; font-size:2rem; line-height:1.2 }
.impacts-hero p{ margin:0; opacity:.95 }

/* ---------- ROLES (left menu + panels) ---------- */
.role-gallery.impacts{ align-items:start }
.role-btn{ padding:.6rem .8rem; border:1px solid #dbe3f0; border-radius:.6rem; background:#fff; cursor:pointer; transition:background .15s, transform .05s }
.role-btn.active{ background:#eef5ff; border-color:#b7d2ff }
.role-btn:active{ transform: translateY(1px) }
.role-summary{ display:block; margin:.25rem 0 .5rem; color:#4b5563 }

/* Accordion */
.role-accordion{ display:grid; gap:.6rem }
.acc-item{ border:1px solid #e5e7eb; border-radius:12px; background:#fff; overflow:hidden; box-shadow:0 1px 6px rgba(0,0,0,.03) }
.acc-head{ display:block }
.acc-btn{ display:flex; flex-direction:column; align-items:flex-start; gap:.25rem; width:100%; padding:1rem; font-weight:650; color:#343a40; text-align:left; border:0; background:linear-gradient(#fafbfc,#f6f8fb) }
.acc-btn:focus{ outline:2px solid #e3ecff; outline-offset:2px }
.acc-meta{ font-size:.82rem; font-weight:400; color:#6b7280 }
.acc-panel[hidden]{ display:none }
.acc-body{ padding:.75rem 1rem 1rem }
.acc-body details{ margin:.35rem 0; border-left:3px solid #e5e7eb; padding:.25rem .75rem; background:#fafafa; border-radius:0 6px 6px 0 }
.acc-body summary{ cursor:pointer; user-select:none }
.acc-body ul{ margin:.35rem 0 .25rem 1.1rem }
.mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace }

/* ---------- EXPERIENCE FLOW ---------- */
/* Hide the visual on small screens to avoid cramping; remove if you want it visible */
@media (max-width: 900px){ .experience-flow{ display:none } }

.experience-flow{ position:relative; margin:2rem auto 2.5rem; max-width:1100px }
.flow-svg{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:0 }
.flow-grid{ position:relative; display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:22px; align-items:stretch; z-index:1 }
.flow-card{ background:#fff; border:1px solid #e8eaf0; border-radius:14px; padding:16px; box-shadow:0 8px 24px rgba(20,24,62,.06); transition:transform .18s, box-shadow .18s, border-color .18s }
.flow-card:hover,.flow-card:focus-within{ transform:translateY(-2px); border-color:#d7dbe8; box-shadow:0 12px 28px rgba(20,24,62,.09) }
.flow-card h3{ margin:0 0 6px; font-size:.95rem; font-weight:700; color:#0b4dbf }
.flow-card h4{ margin:0 0 8px; font-size:1.02rem; font-weight:700; color:#1e293b }
.flow-card p{ margin:0; color:#475569; font-size:.92rem; line-height:1.4 }
.meta{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px }
.pill{ font-size:.74rem; font-weight:600; padding:4px 8px; border-radius:999px; color:#0b4dbf; background:linear-gradient(180deg,#eff6ff,#e7f3ff); border:1px solid #cfe3ff }

/* Flow stroke + nodes */
.path-stroke{ stroke:url(#flow-grad); stroke-width:3.2; stroke-linecap:round; stroke-linejoin:round; fill:none; marker-end:url(#arrow-head); animation:pathDraw 1.2s ease-out both }
@keyframes pathDraw{ from{ stroke-dasharray:1 1000; opacity:0 } to{ stroke-dasharray:1000 0; opacity:1 } }
.node{ fill:#fff; stroke:#1e88e5; stroke-width:2.2; r:4.2 }
</style>

<script>
/* ===== Role UI ===== */
const ROLE_SUMMARIES = {
  analyst: "Analyst impacts: governed KPIs, SQL modeling, high-adoption dashboards.",
  scientist: "Scientist impacts: forecasting pipelines, feature engineering, measurable lifts.",
  engineer: "Engineer impacts: medallion ELT, reliability, cost control.",
  aiml: "AI/ML impacts: streaming detection, latency & precision improvements.",
  genai: "Gen AI impacts: RAG, evaluation, and guardrail design.",
  businessanalyst: "BA impacts: discovery → metrics → enablement with stakeholder alignment."
};
const summaryEl = document.getElementById("role-summary");

function setSummary(role){
  if(!summaryEl) return;
  summaryEl.textContent = ROLE_SUMMARIES[role] || "Explore impacts by role.";
  if (window.runWordIconizer) window.runWordIconizer(summaryEl);
}

function activateRole(role){
  document.querySelectorAll(".role-btn").forEach(btn => {
    const on = btn.dataset.role === role;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-current", on ? "page" : "false");
  });
  document.querySelectorAll(".role-slideshows .role-accordion").forEach(p => p.hidden = true);
  const panel = document.getElementById(`panel-${role}`);
  if(panel){ panel.hidden = false; }
  setSummary(role);
  if (window.runWordIconizer && panel) window.runWordIconizer(panel);
}
document.querySelectorAll(".role-btn").forEach(btn => btn.addEventListener("click", ()=>activateRole(btn.dataset.role)));

/* Accordions (one open per panel) */
function wireAccordion(container){
  container.addEventListener("click", (e) => {
    const btn = e.target.closest(".acc-btn");
    if(!btn) return;
    const panelId = btn.getAttribute("aria-controls");
    const panel = document.getElementById(panelId);
    const expanded = btn.getAttribute("aria-expanded") === "true";

    container.querySelectorAll(".acc-btn[aria-expanded='true']").forEach(b => b.setAttribute("aria-expanded","false"));
    container.querySelectorAll(".acc-panel").forEach(p => p.hidden = true);

    if(!expanded){
      btn.setAttribute("aria-expanded","true");
      panel.hidden = false;
      animateKPIs(panel);
      if (window.runWordIconizer) window.runWordIconizer(panel);
    }
  });
}
document.querySelectorAll(".role-accordion").forEach(wireAccordion);

/* KPI animation */
function animateKPIs(scope){
  const counters = (scope || document).querySelectorAll('.kpi');
  counters.forEach(counter => {
    const target = +counter.getAttribute('data-target');
    let val = 0;
    const step = Math.max(1, Math.round(target / 50));
    const tick = () => {
      val += step;
      if (val < target) { counter.textContent = val; requestAnimationFrame(tick); }
      else { counter.textContent = target; }
    };
    if (!counter.dataset.animated){ counter.dataset.animated = "1"; requestAnimationFrame(tick); }
  });
}

/* Default open role */
activateRole("analyst");

/* ===== Experience Flow (arrow underlay) ===== */
(function(){
  const grid = document.getElementById('flow-grid');
  const svg  = document.getElementById('flow-svg');
  if(!grid || !svg) return;

  const debounce = (fn, ms=120)=>{ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a), ms); }; };

  function drawFlow(){
    const cards = Array.from(grid.querySelectorAll('.flow-card'));
    if (!cards.length) return;

    const r = grid.getBoundingClientRect();
    const w = Math.ceil(r.width);
    const h = Math.ceil(r.height);
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.setAttribute('width', w);
    svg.setAttribute('height', h);

    svg.innerHTML = `
      <defs>
        <linearGradient id="flow-grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#1976d2"/><stop offset="100%" stop-color="#00bcd4"/>
        </linearGradient>
        <marker id="arrow-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#00bcd4"></path>
        </marker>
      </defs>
    `;

    const pts = cards.map(el=>{
      const cr = el.getBoundingClientRect();
      return { x: (cr.left - r.left) + cr.width/2, y: (cr.top - r.top) + cr.height/2 };
    });

    const rows = [];
    const rowThresh = 44;
    pts.forEach(p=>{
      let row = rows.find(rr => Math.abs(rr.y - p.y) < rowThresh);
      if (!row){ row = { y:p.y, pts:[] }; rows.push(row); }
      row.pts.push(p);
    });
    rows.sort((a,b)=>a.y-b.y);
    rows.forEach(row => row.pts.sort((a,b)=>a.x-b.x));

    let d = '';
    const nodes = [];
    rows.forEach((row, i)=>{
      const ordered = (i % 2 === 0) ? row.pts : row.pts.slice().reverse();
      ordered.forEach((p, j)=>{
        const x = Math.round(p.x), y = Math.round(p.y);
        nodes.push({x,y});
        d += (i===0 && j===0) ? `M ${x} ${y} ` : `L ${x} ${y} `;
      });
      const next = rows[i+1];
      if (next){
        const last = ordered[ordered.length-1];
        const nextFirst = ((i+1) % 2 === 0) ? next.pts[0] : next.pts.slice().reverse()[0];
        const midY = (last.y + nextFirst.y) / 2;
        d += `Q ${Math.round(last.x)} ${Math.round(midY)} ${Math.round(nextFirst.x)} ${Math.round(nextFirst.y)} `;
      }
    });

    const path = document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('class','path-stroke');
    path.setAttribute('d', d.trim());
    svg.appendChild(path);

    nodes.forEach(n=>{
      const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
      c.setAttribute('class','node');
      c.setAttribute('cx', Math.round(n.x));
      c.setAttribute('cy', Math.round(n.y));
      c.setAttribute('r', 4.2);
      svg.appendChild(c);
    });
  }

  const run = debounce(drawFlow, 60);
  window.addEventListener('load', run);
  window.addEventListener('resize', run);
  new ResizeObserver(run).observe(grid);
})();
</script>
