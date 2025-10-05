// assets/js/rb-intro.js
(function(){
  var intro = document.getElementById('rbIntro');
  if (!intro) return;

  // Always show on page load
  document.body.classList.add('rb-intro-lock');
  intro.classList.add('show');

  var rows    = Array.prototype.slice.call(intro.querySelectorAll('.rb-row'));
  var actions = document.getElementById('rbActions');
  var btnGo   = document.getElementById('rbContinueBtn');

/* ==== Tagline variations: rotate all 5 within ~6s with cinematic effect ==== */
var TAGLINES = [
  "Designing Precision AI for Real-World Impact",
  "Turning Complex Data into Clear Decisions",
  "Scaling Models from Notebook to Production",
  "Building Reliable ML Systems End-to-End",
  "Engineering Intelligence that Ships Value"
];
var tagEl = document.getElementById('rbTag');

/* Show one line with the effect, then resolve when its animation ends */
function showOneLine(text, runtimeMs){
  return new Promise(function(resolve){
    if (!tagEl) return resolve();
    tagEl.textContent = text;

    // retrigger CSS animation
    tagEl.classList.remove('showline');
    void tagEl.offsetWidth;  // reflow
    tagEl.classList.add('showline');

    // resolve when the per-line animation completes
    setTimeout(resolve, runtimeMs);
  });
}

/* Play all lines sequentially once */
async function playAllLines(){
  var per = 1100; // must match --dur in CSS
  for (var i = 0; i < TAGLINES.length; i++){
    await showOneLine(TAGLINES[i], per);
  }
  // After last fade-out, leave the strongest line visible statically:
  tagEl.classList.remove('showline');
  tagEl.textContent = TAGLINES[0]; // choose your favorite to persist
}
playAllLines();

  /* ============================================ */

  function animateBar(row){
    return new Promise(function(resolve){
      var pctEl = row.querySelector('.rb-pct');
      var fill  = row.querySelector('.rb-bar i');
      var target= Number(row.getAttribute('data-target')||100);
      var dur=900; var start=performance.now();
      function ease(t){ return t<.5?2*t*t:-1+(4-2*t)*t; }
      function frame(now){
        var p=Math.min(1,(now-start)/dur);
        var val=Math.round(target*ease(p));
        fill.style.width=val+'%'; pctEl.textContent=val+'%';
        if(p<1) requestAnimationFrame(frame);
        else{ fill.style.width=target+'%'; pctEl.textContent=target+'%'; resolve(); }
      }
      requestAnimationFrame(frame);
    });
  }

  (async function(){
    for (var i=0;i<rows.length;i++){
      await animateBar(rows[i]);
      await new Promise(function(r){ setTimeout(r,150); });
    }
    actions.classList.add('show');
  })();

  function dismiss(){
    intro.classList.add('reveal');
    setTimeout(function(){
      intro.classList.add('exit');
      setTimeout(function(){
        intro.parentNode && intro.parentNode.removeChild(intro);
        document.body.classList.remove('rb-intro-lock');
      }, 900);
    }, 900);
  }

  btnGo && btnGo.addEventListener('click', dismiss);
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape') dismiss(); }, {once:true});
})();
