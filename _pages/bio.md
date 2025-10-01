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
