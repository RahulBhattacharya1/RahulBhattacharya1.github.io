---
layout: default
title: "Enterprise Impacts by Role"
date: 2025-09-06 13:45
permalink: /enterprise-impacts-role/
thumbnail: /assets/images/impacts.webp
hide_thumbnail_in_body: true
comments: false
share: false
hide_date: true
---

{::nomarkdown}
<section class="impacts-hero">
  <div class="impacts-hero__inner">
    <h1>My Real-World Enterprise Impacts</h1>
    <p>Role-based case studies from retail, healthcare, and finance with measurable outcomes. All data anonymized.</p>
  </div>
</section>

<span id="role-summary" class="role-summary" aria-live="polite">
  Explore impacts by role.
</span>
<br/>
Choose a role:
<br/><br/>

<div class="role-gallery impacts">
  <!-- Left: role menu (kept identical to your structure) -->
  <nav class="role-menu" aria-label="Choose a role">
    <button class="role-btn active" data-role="analyst" aria-current="page">Data Analyst</button>
    <button class="role-btn" data-role="scientist">Data Scientist</button>
    <button class="role-btn" data-role="engineer">Data Engineer</button>
    <button class="role-btn" data-role="aiml">AI / ML Engineer</button>
    <button class="role-btn" data-role="genai">NLP / Gen AI Specialist</button>
    <button class="role-btn" data-role="businessanalyst">Business Analyst</button>
  </nav>

  <!-- Right: role panels (accordion per role) -->
  <section class="role-slideshows">

    <!-- ===================== Data Analyst ===================== -->
    <div class="role-accordion" id="panel-analyst">
      <!-- UHG BI Modernization -->
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

      <!-- Retail KPI Dashboard -->
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
            <details><summary>S · Situation</summary><p>Conflicting metrics across teams led to slow decision cycles.</p></details>
            <details><summary>T · Task</summary><p>Single source of truth with cohorts and A/B readouts.</p></details>
            <details><summary>A · Action</summary><p>Window functions; cohorts; documented KPI contracts; semantic model.</p></details>
            <details open><summary>R · Result</summary>
              <ul>
                <li>Report assembly time cut from 6h → <span class="mono">30m</span></li>
                <li>Consistent KPIs improved trust and adoption</li>
              </ul>
            </details>
          </div>
        </div>
      </article>
    </div>

    <!-- ===================== Data Scientist ===================== -->
    <div class="role-accordion" id="panel-scientist" hidden>
      <!-- Walgreens Forecasting -->
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-scientist-1" id="acc-scientist-1-label">
            Walgreens — Workforce Forecasting at Scale
            <span class="acc-meta">Databricks · PySpark · Azure Data Factory · Time-series</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-scientist-1" role="region" aria-labelledby="acc-scientist-1-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Modular time-series pipelines with automated refresh, seasonality, promos, holidays.</p>
            <details><summary>S</summary><p>Manual, reactive scheduling across thousands of stores.</p></details>
            <details><summary>T</summary><p>Stabilize forecasts, reduce manual corrections, align to business cadence.</p></details>
            <details><summary>A</summary><p>Feature engineering; hierarchical models; ADF triggers; constraints for staffing caps and calendars.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li><span class="kpi" data-target="30">0</span>% reduction in manual intervention</li>
                <li>Improved stability across seasons; fewer late corrections</li>
              </ul>
            </details>
            <p><strong>Stack:</strong> Python, PySpark, Databricks, ADF, ADLS, SQL</p>
          </div>
        </div>
      </article>

      <!-- Demand ML -->
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-scientist-2" id="acc-scientist-2-label">
            Demand Forecasting with Gradient Boosting
            <span class="acc-meta">Python · XGBoost · MLflow · Optuna</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-scientist-2" role="region" aria-labelledby="acc-scientist-2-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Promo/price/holiday features lifted long-tail SKU accuracy.</p>
            <details><summary>S</summary><p>Baseline under-forecasted promotions.</p></details>
            <details><summary>T</summary><p>Improve MAPE on sparse series.</p></details>
            <details><summary>A</summary><p>Lags/rolls; categorical encoding; Optuna; tracked with MLflow.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li>MAPE +21% vs baseline</li>
                <li>Holding cost −8% in pilot</li>
              </ul>
            </details>
          </div>
        </div>
      </article>
    </div>

    <!-- ===================== Data Engineer ===================== -->
    <div class="role-accordion" id="panel-engineer" hidden>
      <!-- Delta Lake ELT -->
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-engineer-1" id="acc-engineer-1-label">
            Delta Lake ELT Orchestration
            <span class="acc-meta">Databricks · Delta · ADF · PySpark</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-engineer-1" role="region" aria-labelledby="acc-engineer-1-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Bronze→Silver→Gold medallion with CDC, data quality tests, and SLAs.</p>
            <details><summary>S</summary><p>Nightly ETL delays; missed downstream SLAs.</p></details>
            <details><summary>T</summary><p>Implement incremental ELT with reliability and lineage.</p></details>
            <details><summary>A</summary><p>Auto Loader; MergeInto; watermarking; expectations; job monitoring.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li>Latency 24h → <span class="mono">30m</span></li>
                <li>On-time delivery at month-end peaks</li>
              </ul>
            </details>
          </div>
        </div>
      </article>

      <!-- Cost Optimization -->
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-engineer-2" id="acc-engineer-2-label">
            Cost-Optimized Batch Processing
            <span class="acc-meta">Spark Tuning · Autoscaling · Monitoring</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-engineer-2" role="region" aria-labelledby="acc-engineer-2-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Adaptive query execution, Z-Order, autoscaling, and lineage-driven alerts.</p>
            <details><summary>S</summary><p>Month-end cost spikes risking budgets.</p></details>
            <details><summary>T</summary><p>Reduce cost without sacrificing SLAs.</p></details>
            <details><summary>A</summary><p>AQE; caching; partition pruning; autoscaling policies; alerting.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li>Compute cost −<span class="kpi" data-target="30">0</span>%</li>
                <li>Reliability 99.9% during peaks</li>
              </ul>
            </details>
          </div>
        </div>
      </article>
    </div>

    <!-- ===================== AI / ML Engineer ===================== -->
    <div class="role-accordion" id="panel-aiml" hidden>
      <!-- Streaming Anomaly Detection -->
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-aiml-1" id="acc-aiml-1-label">
            Real-time Payments Anomaly Detection
            <span class="acc-meta">Structured Streaming · Isolation Forest · Rules</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-aiml-1" role="region" aria-labelledby="acc-aiml-1-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Streaming features + IF + guardrail rules; explainable alerts.</p>
            <details><summary>S</summary><p>Fraud review lag impacted recovery.</p></details>
            <details><summary>T</summary><p>Detect under 2 minutes with low false positives.</p></details>
            <details><summary>A</summary><p>Sliding windows; IF; SHAP-style explanations in alerts.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li>False positives −35%</li>
                <li>p95 detection <span class="mono">&lt;60s</span></li>
              </ul>
            </details>
          </div>
        </div>
      </article>
    </div>

    <!-- ===================== Gen AI Specialist ===================== -->
    <div class="role-accordion" id="panel-genai" hidden>
      <!-- RAG Assistant -->
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-genai-1" id="acc-genai-1-label">
            RAG Knowledge Assistant with Guardrails
            <span class="acc-meta">Chunking · Hybrid Search · Rerank · Evals</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-genai-1" role="region" aria-labelledby="acc-genai-1-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Retrieval-augmented answers with citations, tone and safety guardrails.</p>
            <details><summary>S</summary><p>Docs scattered; search slow; inconsistent answers.</p></details>
            <details><summary>T</summary><p>Grounded, cited answers; fast retrieval; measurable quality.</p></details>
            <details><summary>A</summary><p>Semantic + keyword; reranking; eval harness; red-team prompts.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li>Knowledge search time −80%</li>
                <li>Answer quality scores ↑ on eval set</li>
              </ul>
            </details>
          </div>
        </div>
      </article>
    </div>

    <!-- ===================== Business Analyst ===================== -->
    <div class="role-accordion" id="panel-businessanalyst" hidden>
      <!-- Pricing Playbook -->
      <article class="acc-item">
        <span class="acc-head">
          <button class="acc-btn" aria-expanded="false" aria-controls="acc-ba-1" id="acc-ba-1-label">
            Pricing Strategy Playbook
            <span class="acc-meta">Stakeholder Workshops · Elasticities · Dashboards</span>
          </button>
        </span>
        <div class="acc-panel" id="acc-ba-1" role="region" aria-labelledby="acc-ba-1-label" hidden>
          <div class="acc-body">
            <p><strong>Summary:</strong> Cross-functional discovery → elasticities → KPI dashboards.</p>
            <details><summary>S</summary><p>Margin compression in key categories.</p></details>
            <details><summary>T</summary><p>Identify sweet-spot price bands; align offers.</p></details>
            <details><summary>A</summary><p>Workshops; test plan; SQL extracts; dashboard enablement.</p></details>
            <details open><summary>R</summary>
              <ul>
                <li>+3% margin in pilot categories</li>
                <li>Faster promo planning cycles</li>
              </ul>
            </details>
          </div>
        </div>
      </article>
    </div>

  </section>
</div>
{:/nomarkdown}

<script>
/* ===== Role UX ===== */
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
  if (window.runWordIconizer) window.runWordIconizer(summaryEl); // your existing helper, if present
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
  // optional iconizer pass on newly revealed panel
  if (window.runWordIconizer && panel) window.runWordIconizer(panel);
}

document.querySelectorAll(".role-btn").forEach(btn =>
  btn.addEventListener("click", () => activateRole(btn.dataset.role))
);

/* ===== Accordions (one open at a time within the visible panel) ===== */
function wireAccordion(container){
  container.addEventListener("click", (e) => {
    const btn = e.target.closest(".acc-btn");
    if(!btn) return;
    const panelId = btn.getAttribute("aria-controls");
    const panel = document.getElementById(panelId);
    const expanded = btn.getAttribute("aria-expanded") === "true";

    // close all
    container.querySelectorAll(".acc-btn[aria-expanded='true']").forEach(b => b.setAttribute("aria-expanded","false"));
    container.querySelectorAll(".acc-panel").forEach(p => p.hidden = true);

    // open selected
    if(!expanded){
      btn.setAttribute("aria-expanded","true");
      panel.hidden = false;
      animateKPIs(panel); // animate numbers when opening
      if (window.runWordIconizer) window.runWordIconizer(panel);
    }
  });
}
document.querySelectorAll(".role-accordion").forEach(wireAccordion);

/* ===== KPI Counter Animation ===== */
function animateKPIs(scope){
  const counters = (scope || document).querySelectorAll('.kpi');
  counters.forEach(counter => {
    const target = +counter.getAttribute('data-target');
    let val = 0;
    const step = Math.max(1, Math.round(target / 50));
    const tick = () => {
      val += step;
      if (val < target) {
        counter.textContent = val;
        requestAnimationFrame(tick);
      } else {
        counter.textContent = target;
      }
    };
    // run once per open
    if (!counter.dataset.animated){
      counter.dataset.animated = "1";
      requestAnimationFrame(tick);
    }
  });
}

/* default role */
activateRole("analyst");
</script>

<style>
/* ===== Page Hero ===== */
.impacts-hero{
  background: linear-gradient(135deg,#0b4dbf,#1182d6);
  color:#fff; border-radius:12px; margin:0 0 1rem 0;
}
.impacts-hero__inner{
  padding:2rem 1.25rem; text-align:center; max-width:980px; margin:0 auto;
}
.impacts-hero h1{ margin:.2rem 0 .6rem; font-size:2rem; line-height:1.2 }
.impacts-hero p{ margin:0; opacity:.95 }

/* ===== Reuse your layout; add minimal namespaced polish ===== */
.role-gallery.impacts{ align-items:start }
.role-menu{ position:sticky; top:1rem; align-self:start }

/* Buttons consistency */
.role-btn{ transition: background .15s, transform .05s }
.role-btn.active{ background:#eef5ff; border-color:#b7d2ff }
.role-btn:active{ transform: translateY(1px) }

/* Accordion */
.role-accordion{ display:grid; gap:.6rem }
.acc-item{ border:1px solid #e5e7eb; border-radius:12px; background:#fff; overflow:hidden; box-shadow:0 1px 6px rgba(0,0,0,.03) }
.acc-head{ display:block }
.acc-btn{
  display:flex; flex-direction:column; align-items:flex-start; gap:.25rem;
  width:100%; padding:1rem 1rem; font-size:1rem; font-weight:650;
  color:#343a40; text-align:left; border:0; background:linear-gradient(#fafbfc,#f5f7fa);
  cursor:pointer;
}
.acc-btn:focus{ outline:2px solid #e3ecff; outline-offset:2px }
.acc-meta{ font-size:.82rem; font-weight:400; color:#6b7280 }

.acc-panel[hidden]{ display:none }
.acc-body{ padding:.75rem 1rem 1rem }
.acc-body details{ margin:.35rem 0; border-left:3px solid #e5e7eb; padding:.25rem .75rem; background:#fafafa; border-radius:0 6px 6px 0 }
.acc-body summary{ cursor:pointer; user-select:none }
.acc-body ul{ margin:.35rem 0 .25rem 1.1rem }

/* Monospace snippets for numbers/times */
.mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace }

/* Responsive: keep your grid comfortable */
@media (max-width: 900px){
  .role-menu{ position:static }
  .impacts-hero h1{ font-size:1.6rem }
}
</style>
