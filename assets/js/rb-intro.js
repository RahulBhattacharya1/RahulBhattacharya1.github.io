// assets/js/rb-intro.js
(function () {
  "use strict";

  var intro   = document.getElementById("rbIntro");
  if (!intro) return;

  // Always show on load
  document.body.classList.add("rb-intro-lock");
  intro.classList.add("show");

  var rows    = Array.prototype.slice.call(intro.querySelectorAll(".rb-row"));
  var actions = document.getElementById("rbActions");
  var btnGo   = document.getElementById("rbContinueBtn");
  var tagEl   = document.getElementById("rbTag");

  /* ================= TAGLINE (no overlay effects) =================
     - Works with 0 or 1+ lines
     - Simple JS fade; no CSS pseudo-elements are used
  ================================================================== */
var TAGLINES = [
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
  "Trained 110+ machine learning models improving fraud detection accuracy by 18%",
  "Cut operational costs by 35% via Databricks Delta optimizations",
  "Delivered 45% faster financial reporting using Python and Power BI integration",
  "Achieved 99.9% uptime on retail analytics pipelines with Azure resilience",
  "Optimized 150+ complex PL/SQL packages reducing query times by 80%",
  "Created 220+ AI-driven dashboards improving stakeholder insights by 40%",
  "Reduced healthcare claims processing from 5 days to 8 hours with ADF",
  "Enabled 30% revenue growth via AI-driven retail demand forecasting",
  "Automated reconciliation of 1.2M finance transactions monthly in Informatica",
  "Trained deep learning models achieving 94% accuracy in X-ray classification",
  "Cut reporting backlog from 10 days to 1 day with Power BI",
  "Delivered 140+ AI solutions with direct impact on strategic KPIs",
  "Built 90+ Databricks notebooks driving $3M savings in operational costs",
  "Migrated 200+ Tableau dashboards reducing licensing costs by 28%",
  "Improved credit underwriting speed by 70% using Python-driven AI",
  "Trained predictive models that reduced patient readmission by 15%",
  "Enabled 24/7 retail insights with 500+ streaming ADF pipelines",
  "Migrated 300+ reports to Power BI accelerating CFO reporting by 60%",
  "Automated 75% of healthcare data validations saving 2,000+ staff hours",
  "Reduced ETL failure rates by 40% through Databricks workflow redesign",
  "Improved query execution by 65% across 50+ finance PL/SQL workloads",
  "Designed 250+ Informatica mappings cutting data latency by 35%",
  "Built 180+ AI dashboards giving stakeholders 30% faster insights",
  "Migrated 2,500+ credit risk reports into modern Tableau environment",
  "Reduced analytics compute spend by 20% via Databricks job clustering",
  "Improved patient outcomes by 18% with predictive healthcare AI models",
  "Saved $2.5M annually optimizing 350+ finance pipelines in ADF",
  "Delivered 200+ AI experiments boosting retail forecast accuracy by 28%",
  "Migrated 400+ PL/SQL reports to cloud-native solutions in 9 months",
  "Cut risk assessment time from 48 hours to 6 hours via Python",
  "Enabled 85% faster fraud detection with AI in credit finance",
  "Built 100+ data flows in Informatica reducing SLA breaches by 70%",
  "Developed 300+ AI training pipelines boosting model speed by 40%",
  "Migrated 120+ complex retail reports from legacy to Power BI",
  "Saved 1,800+ analyst hours annually with automated Tableau dashboards",
  "Delivered 50+ healthcare AI dashboards improving patient insights by 45%",
  "Enabled near real-time credit monitoring for 10M+ customers",
  "Cut ETL errors by 35% via Informatica data quality initiatives",
  "Boosted CFO reporting accuracy by 99% with automated reconciliation",
  "Built 200+ AI models reducing operational costs across 8,000+ stores",
  "Increased data scientist productivity by 60% with Databricks templates",
  "Migrated 600+ dashboards to Tableau driving 40% more adoption",
  "Built 350+ finance AI models improving audit compliance by 25%",
  "Optimized ADF pipelines reducing nightly batch time by 80%",
  "Delivered $5M+ value via performance-tuned PL/SQL workloads",
  "Built 90+ Python APIs enabling seamless AI integration in retail",
  "Automated data checks saving 3,000+ hours annually in healthcare",
  "Migrated 200+ critical reports to Power BI within 3 months",
  "Enabled predictive analytics for 20+ hospitals improving outcomes by 12%",
  "Reduced report generation times from 6 hours to 20 minutes",
  "Trained AI models boosting fraud detection true positives by 22%",
  "Developed 450+ ETL flows across Informatica improving stability by 30%",
  "Saved $1.8M by consolidating 400+ Databricks notebooks",
  "Cut dashboard refresh time from 45 min to under 5 min",
  "Delivered AI forecasting models reducing stockouts by 25%",
  "Migrated 2,000+ finance reports in record 9 months",
  "Engineered 80+ complex ADF pipelines ensuring SLA compliance",
  "Optimized credit scoring PL/SQL reducing batch cycle by 70%",
  "Built 600+ Python models scaling to 1B+ rows of data",
  "Enabled 90% faster accounting reconciliations via automation",
  "Developed 100+ AI dashboards for retail execs cutting meetings by 50%",
  "Migrated healthcare ETLs reducing claim processing costs by 15%",
  "Automated Tableau refreshes saving 1,200+ staff hours monthly",
  "Improved retail demand forecast accuracy from 60% to 89%",
  "Deployed Databricks MLflow pipelines across 200+ AI projects",
  "Reduced finance closing cycle from 10 days to 2 days",
  "Enabled predictive maintenance saving $4.2M annually in operations",
  "Improved data quality by 40% with Informatica profiling",
  "Migrated 350+ SQL jobs to ADF with 99.8% accuracy",
  "Reduced healthcare ETL costs by 22% with Databricks scaling",
  "Built 180+ AI-driven dashboards enhancing CFO decision speed by 45%",
  "Trained 300+ deep learning models with 94% precision in finance",
  "Automated invoice processing saving 5,000+ hours yearly",
  "Migrated 250+ complex dashboards to Tableau Cloud",
  "Delivered 120+ Python AI models with 92% deployment success",
  "Cut PL/SQL runtime from 2 hours to 10 minutes",
  "Scaled fraud detection to process 40M+ transactions daily",
  "Optimized 200+ ADF data flows boosting throughput by 35%",
  "Reduced ETL incidents by 70% with Databricks job monitoring",
  "Enabled healthcare AI models cutting readmissions by 18%",
  "Built 400+ retail dashboards improving time-to-insight by 50%",
  "Delivered $8M+ cost savings with AI transformation programs",
  "Migrated 2,500+ finance reports from SSRS to Power BI",
  "Created 300+ Informatica mappings with 40% better efficiency",
  "Optimized AI training workflows reducing runtime by 60%",
  "Improved forecasting accuracy by 30% for retail stakeholders",
  "Automated 350+ data validations improving compliance by 99%",
  "Enabled near real-time analytics for 5M+ credit accounts",
  "Cut processing time by 80% across 120+ ADF workflows",
  "Migrated 140+ dashboards cutting licensing costs by 20%",
  "Built 500+ AI notebooks enabling faster experiments by 45%",
  "Optimized PL/SQL codebase saving 3,000+ execution hours yearly",
  "Developed 90+ Python APIs driving 60% faster integrations",
  "Delivered 200+ AI reports driving measurable $2M ROI",
  "Improved healthcare patient triage by 25% using AI",
  "Migrated 300+ Tableau dashboards for CFO analytics in 4 months",
  "Reduced ETL errors by 40% through Informatica automation",
  "Enabled 30% faster reporting for retail C-suite",
  "Cut finance pipeline costs by $1.5M annually via Databricks tuning",
  "Built 250+ AI dashboards reducing manual reporting effort by 70%",
  "Trained 100+ AI models in Databricks with 95% accuracy",
  "Migrated 400+ Power BI reports to Azure Cloud",
  "Reduced data latency by 60% in finance batch workloads",
  "Automated fraud checks saving 1,000+ compliance hours yearly",
  "Scaled predictive models to 500M+ retail records",
  "Improved PL/SQL performance cutting credit risk assessment from 4h to 20m",
  "Enabled predictive healthcare models reducing ER visits by 10%",
  "Built 150+ Informatica jobs improving SLA adherence by 35%",
  "Deployed 200+ AI models reducing operating costs across 8,000 stores",
  "Reduced nightly ETL time from 6h to 1h with ADF",
  "Migrated 1,000+ dashboards into Tableau Online",
  "Delivered 250+ AI pipelines boosting productivity by 60%",
  "Cut healthcare data processing costs by 25%",
  "Improved retail AI forecast accuracy from 72% to 92%",
  "Enabled $10M+ savings with performance-tuned Python pipelines",
  "Built 300+ dashboards for finance reducing month-end close by 40%",
  "Reduced ETL costs by 22% via Databricks auto-scaling",
  "Migrated 500+ reports into Power BI with zero data loss",
  "Built 200+ predictive dashboards improving stakeholder trust by 30%",
  "Cut PL/SQL execution from 3h to 15m with tuning",
  "Enabled real-time credit scoring for 12M+ accounts",
  "Automated 80% of finance reconciliations saving 6,000+ hours",
  "Delivered 140+ AI experiments cutting retail stockouts by 20%",
  "Migrated 240+ dashboards boosting executive reporting by 45%",
  "Improved healthcare claims AI accuracy by 18%",
  "Built 400+ complex Databricks notebooks reducing pipeline effort by 60%",
  "Reduced Tableau refresh times from 30m to 3m",
  "Enabled predictive finance reporting cutting costs by $4M",
  "Automated healthcare ETLs saving 2,200+ staff hours yearly",
  "Migrated 180+ PL/SQL procedures reducing runtime by 75%",
  "Trained 250+ models improving retail insights by 35%",
  "Delivered 1,000+ complex data flows across ADF",
  "Cut reporting time from 12h to 1h for CFO teams",
  "Built 100+ AI models increasing fraud detection by 22%",
  "Migrated 320+ reports into Power BI in 100 days",
  "Reduced healthcare ETL incidents by 45%",
  "Improved retail forecast accuracy by 20% using Databricks",
  "Built 280+ Informatica flows improving compliance by 30%",
  "Delivered $6M+ value in finance AI transformation",
  "Migrated 250+ Tableau dashboards for 5,000+ stakeholders",
  "Optimized PL/SQL queries cutting ETL cycle by 70%",
  "Built 150+ AI dashboards for healthcare decision-makers",
  "Reduced retail reporting lag by 80% with ADF",
  "Enabled 90% faster insights for finance stakeholders",
  "Migrated 600+ Power BI reports with zero SLA breaches",
  "Built 350+ AI projects reducing costs across industries",
  "Trained 100+ deep learning models with 96% accuracy",
  "Cut credit assessment times by 75% with Python AI",
  "Delivered predictive dashboards boosting CFO trust by 40%",
  "Migrated 2,800+ reports into cloud analytics platforms",
  "Reduced Informatica errors by 60% boosting SLA reliability",
  "Built 200+ AI-powered dashboards improving retail ROI by 28%",
  "Enabled near real-time analytics cutting decision time by 70%",
  "Automated healthcare processes saving 5,000+ staff hours",
  "Migrated 120+ critical reports reducing downtime by 85%",
  "Trained 180+ predictive models scaling to 1B+ rows",
  "Delivered 400+ data pipelines in Databricks cutting costs by $3M",
  "Built 600+ dashboards across Power BI and Tableau",
  "Reduced finance closing cycle by 6 days with automation",
  "Enabled predictive AI models improving patient outcomes by 20%",
  "Migrated 350+ PL/SQL scripts into optimized cloud pipelines",
  "Cut ETL cycle time from 8h to 2h saving $2M",
  "Built 500+ AI dashboards enhancing executive insights",
  "Delivered 700+ Python models across healthcare, retail, finance",
  "Automated 60% of accounting workflows saving 4,000+ hours",
  "Reduced dashboard build times by 45% using Databricks",
  "Migrated 450+ reports into Tableau Cloud in 6 months",
  "Enabled $12M+ annual savings with AI cost optimization",
  "Built 200+ complex Informatica ETLs reducing risk by 35%",
  "Trained 250+ AI models driving measurable KPIs in retail",
  "Reduced finance ETL failures by 55% via ADF optimizations",
  "Delivered predictive insights cutting stockouts by 18%",
  "Built 300+ dashboards for healthcare stakeholders improving care quality",
  "Migrated 500+ Power BI reports reducing downtime by 70%",
  "Cut PL/SQL runtime from 90m to 5m in core systems",
  "Enabled real-time analytics for 8,000+ stores via Databricks",
  "Automated 85% of ETL testing saving 2,500+ hours",
  "Built 350+ AI notebooks achieving 95% model reproducibility",
  "Migrated 200+ Tableau dashboards improving adoption by 30%",
  "Reduced ETL latency by 60% cutting costs by $5M",
  "Built 400+ AI pipelines delivering measurable ROI across domains",
  "Trained 150+ healthcare models reducing patient risk by 12%",
  "Enabled predictive analytics cutting finance fraud by 20%",
  "Delivered 600+ dashboards across retail & healthcare",
  "Migrated 1,200+ reports into Power BI in under a year",
  "Reduced runtime by 80% on 220+ PL/SQL packages",
  "Built 250+ AI projects saving $10M in costs",
  "Automated 300+ accounting workflows reducing staff load by 70%",
  "Improved healthcare AI outcomes by 22% with ML models",
  "Migrated 320+ Tableau dashboards cutting licensing costs by 20%",
  "Enabled predictive retail forecasts improving margins by 25%",
  "Reduced Databricks costs by $2.8M annually",
  "Built 500+ Python APIs enabling faster AI deployment",
  "Trained 280+ AI models achieving 96% accuracy",
  "Delivered predictive dashboards increasing stakeholder trust by 35%"
];

  // prepare a simple fade transition (no stars/sheen)
  if (tagEl) {
    tagEl.style.transition = "opacity 380ms ease";
    tagEl.style.opacity = "0";
  }

  function setTagline(text) {
    if (!tagEl) return;
    // fade out
    tagEl.style.opacity = "0";
    // swap text after a short delay, then fade in
    setTimeout(function () {
      tagEl.textContent = text || "";
      tagEl.style.opacity = "1";
    }, 120);
  }

  async function playTaglines() {
    if (!tagEl) return;

    if (!TAGLINES || TAGLINES.length === 0) {
      // nothing to show
      tagEl.textContent = "";
      tagEl.style.opacity = "1";
      return;
    }
    if (TAGLINES.length === 1) {
      setTagline(TAGLINES[0]);
      return;
    }
    // 2+ lines: show each once; adjust timing if you want
    var per = 1100; // ~1.1s per line
    for (var i = 0; i < TAGLINES.length; i++) {
      setTagline(TAGLINES[i]);
      // wait for the line to display before moving to next
      // (per includes fade timings above)
      // eslint-disable-next-line no-loop-func
      await new Promise(function (r) { setTimeout(r, per); });
    }
    // leave the first line visible at the end
    setTagline(TAGLINES[0]);
  }

  /* ================= PROGRESS BARS =================
     - JS animates width from 0 -> target%
     - Ensure CSS positions the fill with left/top/bottom + width
       (NOT with inset:0) so the width animation renders correctly.
  =================================================== */
  function animateBar(row) {
    return new Promise(function (resolve) {
      var pctEl  = row.querySelector(".rb-pct");
      var fill   = row.querySelector(".rb-bar i");
      var target = Number(row.getAttribute("data-target") || 100);
      var dur    = 900;
      var start  = performance.now();

      // small visual kick so "0%" bars don't look empty if glare is disabled
      if (fill && !fill.style.width) fill.style.width = "0%";

      function ease(t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; }

      function frame(now) {
        var p   = Math.min(1, (now - start) / dur);
        var val = Math.round(target * ease(p));
        if (fill) fill.style.width = val + "%";
        if (pctEl) pctEl.textContent = val + "%";
        if (p < 1) requestAnimationFrame(frame);
        else {
          if (fill) fill.style.width = target + "%";
          if (pctEl) pctEl.textContent = target + "%";
          resolve();
        }
      }
      requestAnimationFrame(frame);
    });
  }



  if (btnGo) btnGo.addEventListener("click", dismiss);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") dismiss();
  }, { once: true });
})();
