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
/* ─────────────────────────────────────────────
   Experience page (scoped, no global duplicates)
   ───────────────────────────────────────────── */

/* ===== Hero ===== */
.impacts-hero{
  margin: 0 0 1.5rem 0;
  background: linear-gradient(135deg,#eef4ff 0%,#f7fbff 40%,#ffffff 100%);
  border: 1px solid #e6eefb;
  border-radius: 14px;
}
.impacts-hero__inner{
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem 1.25rem;
}
.impacts-hero__inner h1{
  margin: 0 0 .5rem 0;
  font-size: clamp(1.4rem, 3.2vw, 2rem);
  line-height: 1.2;
}
.impacts-hero__inner p{
  margin: 0;
  color: #4a5668;
}

/* ===== Accordion (Experience-only content) ===== */
.role-accordion{ display: grid; gap: .65rem; }
.acc-item{
  border: 1px solid #e7edf6;
  border-radius: 12px;
  background: #fff;
  overflow: hidden;
}
.acc-head{ display:block; margin:0; padding:0; background:transparent; border:0; }
.acc-btn{
  /* Full-width header button for accordion; do NOT style .role-btn here */
  display: flex; align-items: flex-start; gap: .4rem;
  width: 100%;
  padding: .9rem 1rem;
  background: transparent; border:0; cursor:pointer;
  font-weight: 600; color:#39475a; text-align:left;
}
.acc-btn:focus-visible{ outline: 2px solid #7aa2ff; outline-offset: 2px; border-radius: 8px; }
.acc-meta{ color:#64748b; font-weight: 500; font-size: .9rem; }
.acc-panel[hidden]{ display:none; }
.acc-body{ padding: .75rem 1rem 1rem; color:#374151; }

/* ===== Arrow-flow layout (desktop/tablet only) ===== */
.experience-flow{
  position: relative;
  margin: 1rem 0 2rem;
  border-radius: 14px;
  background: #ffffff;
  border: 1px solid #eef2f7;
  padding: 1rem;
}
.flow-grid{
  display: grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap: 1rem;
  position: relative;
  z-index: 1; /* above arrows */
}
.flow-card{
  border: 1px solid #e8eef6;
  border-radius: 12px;
  background: #fff;
  padding: .9rem 1rem;
  box-shadow: 0 1px 8px rgba(20,35,70,.06);
}
.flow-card h4{ margin:.1rem 0 .35rem; font-size: 1rem; }
.flow-card .meta{ color:#6b7280; font-size: .9rem; }
.pill{
  display:inline-block; margin:.25rem .35rem 0 0; padding:.15rem .5rem;
  border:1px solid #e3e9f3; border-radius:999px; font-size:.8rem; color:#475569;
}

/* Arrow layer (SVG or canvas you place absolutely inside .experience-flow) */
.flow-svg{
  position:absolute; inset:0; z-index:0; pointer-events:none;
}
.flow-svg .arrow{ stroke:#c9d6ee; stroke-width:2; fill:none; }
.flow-svg .node { fill:#e8f1ff; stroke:#bcd0f5; }

/* ===== Desktop-only visuals (hide heavy graphic on phones) ===== */
@media (max-width: 768px){
  .experience-flow{ display:none; }
}

/* ===== Utilities specific to this page ===== */
.role-summary{ display:block; margin:.25rem 0 1rem; color:#475569; }

/* Note:
   - We intentionally do NOT style .role-menu or .role-btn here.
   - Those are defined in your site/global CSS and already work on Skills.
   - Make sure the wrapper is: <div class="role-gallery"> (no extra "impacts" class).
*/
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
