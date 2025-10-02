---
layout: default
title: "Rahul's Documents"
date: 2025-08-22
thumbnail: /assets/images/doc.webp
permalink: /bio/
hide_thumbnail_in_body: true
comments: false
share: false
hide_date: true
hide_top_title: true
---

{::nomarkdown}
<!-- ===== HERO ===== -->
<section class="impactful-hero">
  <div class="impactful-hero__inner">
    <h1>A Bit About Rahul</h1>
    <p>Data-driven professional with a passion for analytics, AI, and storytelling. Exploring how technology creates value across industries.</p>
  </div>
</section>

<!-- About | Bold-Preserving Animation -->
<section class="about-rich">
  <div class="about-wrap">
    
    <!-- Put your EXACT formatted text below inside <template>. 
         You may include <strong>, <em>, links, and emojis. -->
    <template id="about-source">
      <p>Hi there ✨, I’m <strong>Rahul Bhattacharya</strong> — a builder, dreamer, and relentless explorer at the intersection of <strong>data, AI, and business transformation</strong>.</p>
      
      <p>For me, data has never just been rows and columns; it’s a living story - one that, when told well, can shift strategies, unlock growth, and change how entire organizations move forward.</p>

      <p>Over the last decade, I’ve had the privilege of shaping that story at companies like <strong>Walgreens, Experian, Toyota Financial Services, Galderma, UnitedHealth Group, and Bombardier</strong>. In each role, I’ve found not just problems to fix, but opportunities to reimagine what’s possible:</p>

      <p>⚡ <strong>Cutting reporting downtime by more than 70% through automation</strong></p>
      <p>💡 Leading a migration from <strong>Tableau to Power BI</strong>, saving millions while giving teams faster, sharper insights</p>
      <p>🔮 Building predictive models that turned hindsight into foresight - so leaders could see tomorrow’s challenges today</p>
      <p>🌍 Designing KPIs and frameworks that united global teams from <strong>EMEA to APAC to North America</strong></p>

      <p>But numbers, though powerful, are only part of the story. What excites me most is the <strong>thrill of the future</strong>: a future where AI doesn’t just answer questions, it anticipates them; where dashboards don’t just report, they advise; where decision-makers everywhere can move with confidence because their systems whisper the truth before it’s visible.</p>

      <p>My philosophy is simple: <strong>problems are signals</strong>. Some tell you to wait and observe, others demand immediate action. The art is knowing the difference, and the joy is in solving them in ways that inspire trust.</p>

      <p>Outside of my enterprise work, I’ve built a portfolio of <strong>AI/ML projects</strong> - from resume analyzers to trip planners, from defect classifiers to sentiment tools. Each one is both an experiment and a gift: a way to test ideas, share knowledge, and invite others into the excitement of this journey.</p>

      <p>Looking ahead, I see the next 20 years not just as a career, but as an <strong>adventure</strong>: one where AI becomes the nervous system of every business, where insights move at the speed of thought, and where I can help shape the tools that will power decisions for a generation. 🚀</p>

      <p>Thank you for visiting my blog and reading thus far, every reader encourages me to do more. You’re not just seeing what I’ve built; you’re stepping into the vision I’m building toward. The best part? The most thrilling chapters are still to come. 🌌</p>
    </template>

    <div id="about-animated" class="about-animated"></div>
  </div>
</section>

<style>
.about-wrap{max-width:880px;margin:0 auto;padding:.5rem 1.25rem;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
.headline{font-size:clamp(2rem,5vw,3rem);font-weight:800;margin:0 0 1.25rem;background:linear-gradient(90deg,#6aa7ff,#b388ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.about-animated p{font-size:1.08rem;line-height:1.8;margin:0 0 1.05rem 0;opacity:0;transform:translateY(16px)}
.about-animated p.reveal{animation:rise .85s cubic-bezier(.21,.98,.6,.99) forwards}
@keyframes rise{to{opacity:1;transform:translateY(0)}}

@media (max-width: 768px){
  #bio-mobile.m-tabs { padding: 0 !important; }
  #bio-mobile .m-panels,
  #bio-mobile .m-panel { margin: 0 !important; padding: 0 !important; }
  #bio-mobile .m-panel iframe {
    display: block;
    width: 100% !important;
    max-width: 100% !important;
    border: 0;
  }
}

</style>

<script>
document.addEventListener('DOMContentLoaded', () => {
  const tpl = document.getElementById('about-source');
  const target = document.getElementById('about-animated');

  // Clone all template nodes so original HTML (including <strong>, links, emojis) is preserved
  const frag = tpl.content.cloneNode(true);
  const nodes = Array.from(frag.querySelectorAll('p'));

  // Append and animate, preserving innerHTML (keeps bold)
  nodes.forEach((p, i) => {
    const clone = p.cloneNode(true);
    target.appendChild(clone);
    setTimeout(() => clone.classList.add('reveal'), i * 450);
  });
});
</script>

<!-- Desktop: keep your existing bio layout here -->
<div id="bio-desktop">
  <!-- Your current desktop HTML stays here unchanged -->
  <!-- e.g., the blue hero bar, paragraphs, bullets, etc. -->
</div>

<!-- Mobile Tabs (shown only on small screens) -->
<section id="bio-mobile" class="m-tabs" hidden>
  <nav class="m-tablist" role="tablist" aria-label="Profile sections">
    <button class="m-tab is-active" role="tab" aria-selected="true" aria-controls="panel-about" id="tab-about" data-tab="about">About Me</button>
    <button class="m-tab" role="tab" aria-selected="false" aria-controls="panel-skills" id="tab-skills" data-tab="skills">Skills</button>
    <button class="m-tab" role="tab" aria-selected="false" aria-controls="panel-exp" id="tab-exp" data-tab="experience">Experience</button>
  </nav>

  <div class="m-panels">
    <article id="panel-about" class="m-panel" role="tabpanel" aria-labelledby="tab-about"></article>
    <article id="panel-skills" class="m-panel" role="tabpanel" aria-labelledby="tab-skills" hidden></article>
    <article id="panel-exp" class="m-panel" role="tabpanel" aria-labelledby="tab-exp" hidden></article>
  </div>
</section>

<style>
/* --- Layout visibility --- */
#bio-desktop { display:block; }
#bio-mobile { display:none; }
@media (max-width: 768px){
  #bio-desktop { display:none; }
  #bio-mobile { display:block; }
}

/* --- Mobile tabs styling --- */
.m-tabs { padding: 0 1rem 2rem; }
.m-tablist {
  position: sticky; top: 0; z-index: 5;
  display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem;
  padding: .75rem 0; background: transparent;
}
.m-tab {
  appearance: none; border: 1px solid rgba(0,0,0,.12);
  background: rgba(0,0,0,.03);
  padding: .6rem .8rem; border-radius: 10px;
  font: 600 0.95rem/1.1 Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}
.m-tab.is-active {
  border-color: rgba(0,0,0,.18);
  background: linear-gradient(90deg, #6aa7ff22, #b388ff22);
}

.m-panels { margin-top: .5rem; }
.m-panel { padding: .5rem 0 0; }
.m-panel p { margin: 0 0 1rem 0; line-height: 1.7; }
.m-panel ul { padding-left: 1.25rem; }

/* Light/dark friendly borders */
@media (prefers-color-scheme: dark){
  .m-tab { border-color: rgba(255,255,255,.16); background: rgba(255,255,255,.04); }
  .m-tab.is-active { border-color: rgba(255,255,255,.26); }
}
</style>

<script>
(function(){
  // URLs to load
  const routes = {
    about: "/about-me-embed/",                             // your About lives on this page or root
    skills: "/skills-embed/",
    experience: "/experience-embed/"
  };

  // Content selectors to extract from those pages
  const candidateSelectors = [
    "main .post-content",
    "article.post .post-content",
    "main article",
    "main",
    ".page-content",
    "#main"
  ];

  // Cache loaded HTML fragments
  const cache = new Map();

  // Elements
  const mobile = document.getElementById("bio-mobile");
  const aboutPanel = document.getElementById("panel-about");
  const skillsPanel = document.getElementById("panel-skills");
  const expPanel = document.getElementById("panel-exp");
  const tabs = Array.from(document.querySelectorAll(".m-tab"));
  const panels = {
    about: aboutPanel,
    skills: skillsPanel,
    experience: expPanel
  };

  // Show mobile section on small screens (CSS handles, but unhide attribute for a11y)
  const mq = window.matchMedia("(max-width: 768px)");
  function toggleHidden(){
    mobile.hidden = !mq.matches;
  }
  toggleHidden();
  mq.addEventListener("change", toggleHidden);

  // Load extractor
  async function loadSection(key){
    if (cache.has(key)) return cache.get(key);

    // If "about" is this page, clone visible desktop content (first render only)
    if (key === "about" && location.pathname.includes("/bio")) {
      const clone = document.getElementById("bio-desktop")?.cloneNode(true) || document.body.cloneNode(false);
      // Try to grab the main rich area from your desktop bio
      const mainCopy = clone.querySelector(".post-content, main, article, .content") || clone;
      cache.set("about", mainCopy.innerHTML);
      return mainCopy.innerHTML;
    }

    const res = await fetch(routes[key], { credentials: "same-origin" });
    const html = await res.text();
    const doc = new DOMParser().parseFromString(html, "text/html");

    let found = "";
    for (const sel of candidateSelectors){
      const el = doc.querySelector(sel);
      if (el && el.innerHTML.trim()) { found = el.innerHTML; break; }
    }
    // Fallback to body if no selector matched
    if (!found) found = (doc.body && doc.body.innerHTML) ? doc.body.innerHTML : "<p>Content not found.</p>";

    cache.set(key, found);
    return found;
  }

  // Activate a tab
async function activate(key, push = true) {
  tabs.forEach(btn => {
    const active = btn.dataset.tab === key;
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-selected", String(active));
    panels[btn.dataset.tab].hidden = !active;
  });

  const panel = panels[key];

  if (key === "about") {
    // keep existing About Me behavior
    if (!panel.dataset.loaded) {
      const clone = document.getElementById("bio-desktop")?.cloneNode(true);
      const mainCopy = clone?.querySelector(".post-content, main, article, .content") || clone || document.body;
      panel.innerHTML = mainCopy.innerHTML;
      panel.dataset.loaded = "true";
    }
  } else {
    // Skills and Experience: use iframe embed
    if (!panel.dataset.loaded) {
      panel.innerHTML = "";
      const frame = document.createElement("iframe");
      frame.src = routes[key];
      frame.title = key + " section";
      frame.loading = "lazy";
      frame.style.width = "100%";
      frame.style.border = "0";
      frame.style.minHeight = "70vh";

      frame.addEventListener("load", () => {
        try {
          const doc = frame.contentDocument || frame.contentWindow.document;
          const resize = () => {
            frame.style.height = Math.max(600, doc.body.scrollHeight) + "px";
          };
          resize();
          new MutationObserver(resize).observe(doc.body, { childList: true, subtree: true });
          window.addEventListener("resize", resize);
        } catch(e){}
      });

      panel.appendChild(frame);
      panel.dataset.loaded = "true";
    }
  }

  if (push) {
    const url = new URL(location.href);
    url.hash = "tab=" + key;
    history.replaceState(null, "", url);
  }
}
  
  // Tab clicks + keyboard
  tabs.forEach(btn => {
    btn.addEventListener("click", () => activate(btn.dataset.tab));
    btn.addEventListener("keydown", e => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      const i = tabs.indexOf(btn);
      const next = e.key === "ArrowRight" ? (i+1) % tabs.length : (i-1+tabs.length) % tabs.length;
      tabs[next].focus();
      activate(tabs[next].dataset.tab);
    });
  });

  // Initial tab from hash/query
  function getInitialTab(){
    const h = (location.hash || "").replace(/^#/, "");
    const params = new URLSearchParams(h.includes("=") ? h : "");
    const fromHash = params.get("tab");
    const fromQuery = new URLSearchParams(location.search).get("tab");
    return (fromHash || fromQuery || "about").toLowerCase();
  }

  // Boot
  activate(getInitialTab(), false);
})();
</script>

<noscript>
  <!-- Fallback for no-JS: show a simple list of links -->
  <p><a href="/">About Me</a> · <a href="/skills/">Skills</a> · <a href="/experience/">Experience</a></p>
</noscript>


  <div class="emf-frame">
    <iframe
      src="https://www.emailmeform.com/builder/form/51b1w1dDfW07xN1a6"
      title="Contact Form"
      loading="lazy"
      frameborder="0"
      scrolling="no"
      width="100%"
      height="420">
    </iframe>
  </div>

{:/nomarkdown}
