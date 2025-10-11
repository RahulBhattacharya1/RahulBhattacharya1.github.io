(function(){
  /* =========================
     Config + Data
  ========================== */
  const RBX_ROTATE_MS = 4000;
  const RBX_FADE_MS   = 200;

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

  const RBX_METRICS = [
    { v: "100+", c: "AI Models trained" },
    { v: "300+", c: "Dashboards created" },
    { v: "200+", c: "Data flows & pipelines" },
    { v: "80+",  c: "Databricks" },
    { v: "80+",  c: "Warehouses designed" },
    { v: "8,000+", c: "Stores supported" },
    { v: "40%",    c: "Faster pipelines" },
    { v: "1,200+", c: "Compute hours saved" },
    { v: "95%",    c: "Query speed gains" },
    { v: "5,000+", c: "Staff hours saved" },
    { v: "500M+",  c: "Records processed" },
    { v: "92%",    c: "Forecast accuracy" },
    { v: "$3M",    c: "Annualized savings" },
    { v: "150k",   c: "Employees scheduled" },
    { v: "99.9%",  c: "Pipeline uptime" }    
  ];
  const RBX_METRICS_INTERVAL = 4000;
  const RBX_METRICS_FADE     = 200;

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

// === Bold numbers (with optional %, +, or unit) ===
const RBX_CAP_UNIT_WORDS =
  '(?:hours?|hrs?|hr|minutes?|mins?|min|seconds?|secs?|sec|ms|days?|day|yrs?|years?|transactions?|txns?|records?|rows?|reports?|models?|pipelines?|dashboards?|stores?|users?|hospitals?|accounts?|GB|TB|MB|KB|B|h|m|s)?';

// match any number, with optional decimal/comma, then optional % or + or unit
const RBX_CAP_REGEX = new RegExp(
  String.raw`\b(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:%|\+)?\s*` + RBX_CAP_UNIT_WORDS + String.raw`\b`,
  'g'
);

function rbxFormatCaption(text){
  return text.replace(RBX_CAP_REGEX, m => `<strong>${m}</strong>`);
}

  /* =========================
     Caption rotation (single span)
  ========================== */
  // Banner-only iconizer (scoped to the caption element)
function rbxIconizeCaption(el){
  if (!el) return;

  // Terms → icons (lowercased keys)
  const ICONS = {
    'python':        '/assets/icons/python.svg',
    'databricks':    '/assets/icons/databricks.svg',
    'azure':         '/assets/icons/azure.svg',
    'power bi':      '/assets/icons/powerbi.svg',
    'tableau':       '/assets/icons/tableau.svg',
    'adf':           '/assets/icons/azure.svg',
    'numpy':         '/assets/icons/numpy.svg',
    'hugging face':  '/assets/icons/huggingface.svg'
  };

  // Match common variants (case-insensitive); allow “PowerBI”, “HuggingFace”, “pl/sql”, etc.
  const RE = /\b(databricks|python|azure|power\s*bi|tableau|adf|numpy|hugging\s*face)\b/gi;

  // Only touch this one caption node; don’t walk the page
  el.innerHTML = el.innerHTML.replace(RE, (m) => {
    const k = m.toLowerCase().replace(/\s+/g,' ');
    const src = ICONS[k];
    if (!src) return m;
    return `${m}<img src="${src}" alt="${m} icon" class="inl-icon" style="height:1em;vertical-align:-0.15em;margin-left:.15em;">`;
  });
}

  function rbxRotateCaption(){
    const el = document.getElementById("rbx-rotating-caption");
    if(!el || !Array.isArray(RBX_CAPTIONS) || RBX_CAPTIONS.length === 0) return;

    let i = 0;
    el.innerHTML = rbxFormatCaption(RBX_CAPTIONS[i]); // initial paint
    rbxIconizeCaption(el);

    function next(){
      el.classList.add("rbx-cap-out");
      setTimeout(()=>{
        i = (i + 1) % RBX_CAPTIONS.length;
        el.innerHTML = rbxFormatCaption(RBX_CAPTIONS[i]); // IMPORTANT: innerHTML
        rbxIconizeCaption(el);
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
  if(!row || !m1v || !m1c || !m2v || !m2c) return;  // ← removed indicator requirement

  let i = 0; // left index of the pair
  let timer = null;

  function render(){
    const j = (i + 1) % RBX_METRICS.length;
    row.classList.add("rbx-fade-out");
    setTimeout(()=>{
      m1v.textContent = RBX_METRICS[i].v;
      m1c.textContent = RBX_METRICS[i].c;
      m2v.textContent = RBX_METRICS[j].v;
      m2c.textContent = RBX_METRICS[j].c;
      row.classList.remove("rbx-fade-out");
      i = (i + 2) % RBX_METRICS.length;;
    }, RBX_METRICS_FADE);
  }

  function start(){ if(!timer) timer = setInterval(render, RBX_METRICS_INTERVAL); }
  function stop(){ if(timer){ clearInterval(timer); timer = null; } }

  render();
  start();

  // pause on hover/focus; resume on leave/blur
  row.addEventListener("mouseenter", stop);
  row.addEventListener("mouseleave", start);
  row.addEventListener("focusin", stop);
  row.addEventListener("focusout", start);
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
