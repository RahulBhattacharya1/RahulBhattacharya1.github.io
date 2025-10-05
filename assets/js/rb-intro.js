
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
    // Keep empty, or add one line:
    // "Designing Precision AI for Real-World Impact"
  ];

  // prepare a simple fade transition (no stars/sheen)
  if (tagEl) {
    tagEl.style.transition = "opacity 380ms ease";
    tagEl.style.opacity = "0";
  }


})();
