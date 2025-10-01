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

<style>
  details { border: 1px solid #e5e7eb; border-radius: 8px; padding: .75rem 1rem; margin: .75rem 0; background:#fff; }
  details[open] { box-shadow: 0 2px 10px rgba(0,0,0,.05); }
  summary { cursor: pointer; font-weight: 600; outline: none; }
  summary::-webkit-details-marker { display: none; }
  summary::after { content:"▾"; float:right; transition: transform .2s ease; }
  details[open] summary::after { transform: rotate(180deg); }
  details > div { margin-top: .75rem; }

  /* Locker form – left aligned, premium look, B/W button */
  #login-form { margin-top: 1.25rem; max-width: 520px; }
  #login-form p { margin: 0 0 .5rem 0; }
  #passwordInput {
    width: 100%;
    padding: .75rem 1rem;
    border: 1px solid #d1d5db;
    border-radius: .5rem;
    font-size: 1rem;
    outline: none;
    transition: box-shadow .2s, border-color .2s;
  }
  #passwordInput:focus {
    border-color: #717172;
    box-shadow: 0 0 0 3px rgba(0,0,0,.08);
  }
  #login-form button {
    margin-top: .75rem;
    padding: .6rem 1.1rem;
    font-weight: 600;
    border-radius: .5rem;
    border: 1px solid #A8A8A8;
    background: #717172;
    color: #fff;
    cursor: pointer;
    transition: filter .15s ease, background .15s ease, color .15s ease;
  }
  #login-form button:hover { filter: brightness(0.9); }
  #login-form button:active { filter: brightness(0.8); }
  #error-msg { margin-top: .5rem; color: #dc2626; }
</style>



<script>
  const PASSWORD = "713";

  function checkPassword() {
    const input = document.getElementById("passwordInput").value.trim();
    if (input === PASSWORD) {
      document.getElementById("locker-container").style.display = "block";
      document.getElementById("login-form").style.display = "none";
    } else {
      document.getElementById("error-msg").textContent = "Incorrect password!";
    }
  }
</script>

<section class="about-story">
  <div class="about-wrap">
    <h1 class="headline">About Me</h1>
    <div id="story" data-paragraphs='[
      "Hi there ✨, I’m Rahul Bhattacharya — a builder, dreamer, and relentless explorer at the intersection of data, AI, and business transformation. For me, data has never just been rows and columns; it’s a living story - one that, when told well, can shift strategies, unlock growth, and change how entire organizations move forward.",
      "Over the last decade, I’ve had the privilege of shaping that story at companies like Walgreens, Experian, Toyota Financial Services, Galderma, UnitedHealth Group, and Bombardier. In each role, I’ve found not just problems to fix, but opportunities to reimagine what’s possible:",
      "⚡ Cutting reporting downtime by more than 70% through automation",
      "💡 Leading a migration from Tableau to Power BI, saving millions while giving teams faster, sharper insights",
      "🔮 Building predictive models that turned hindsight into foresight - so leaders could see tomorrow’s challenges today",
      "🌍 Designing KPIs and frameworks that united global teams from EMEA to APAC to North America",
      "But numbers, though powerful, are only part of the story. What excites me most is the thrill of the future: a future where AI doesn’t just answer questions, it anticipates them; where dashboards don’t just report, they advise; where decision-makers everywhere can move with confidence because their systems whisper the truth before it’s visible.",
      "My philosophy is simple: problems are signals. Some tell you to wait and observe, others demand immediate action. The art is knowing the difference, and the joy is in solving them in ways that inspire trust.",
      "Outside of my enterprise work, I’ve built a portfolio of AI/ML projects - from resume analyzers to trip planners, from defect classifiers to sentiment tools. Each one is both an experiment and a gift: a way to test ideas, share knowledge, and invite others into the excitement of this journey.",
      "Looking ahead, I see the next 20 years not just as a career, but as an adventure: one where AI becomes the nervous system of every business, where insights move at the speed of thought, and where I can help shape the tools that will power decisions for a generation. 🚀",
      "Thank you for visiting my blog and reading thus far, every reader encourages me to do more. You’re not just seeing what I’ve built; you’re stepping into the vision I’m building toward. The best part? The most thrilling chapters are still to come. 🌌"
    ]'></div>
  </div>
</section>

<style>
.about-story { 
  min-height: 100vh; padding: 4rem 1.5rem; 
  display:flex; justify-content:center; align-items:flex-start; 
}
.about-wrap { max-width: 880px; font-family: "Inter", sans-serif; }
.headline {
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 700; margin-bottom: 2rem;
  background: linear-gradient(90deg,#6aa7ff,#b388ff);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.story-line {
  font-size: 1.12rem; line-height: 1.75;
  margin: 0 0 1.25rem 0;
  opacity:0; transform: translateY(20px);
}
.reveal { animation: rise .9s cubic-bezier(.21,.98,.6,.99) forwards; }
@keyframes rise { to{ opacity:1; transform:translateY(0); } }
/* Slight emphasis for emoji-bullets */
.story-line:has(⚡), .story-line:has(💡), .story-line:has(🔮), .story-line:has(🌍) {
  font-weight:600; color:#444;
}
</style>

<script>
document.addEventListener("DOMContentLoaded", () => {
  const box = document.getElementById("story");
  const paras = JSON.parse(box.getAttribute("data-paragraphs"));
  paras.forEach((txt,i) => {
    const p=document.createElement("p");
    p.className="story-line";
    p.textContent=txt;
    box.appendChild(p);
    setTimeout(()=>{ p.classList.add("reveal"); }, i*600);
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
