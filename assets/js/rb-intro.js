// assets/js/rb-intro.js
(function(){
  var STORAGE_KEY = "rb_intro_v1_dismissed";
  var DISMISS_TTL_DAYS = 365; // set to null to never show again after first dismiss

  function ttlValid(saved){
    if (!DISMISS_TTL_DAYS) return true;
    try{
      var obj = JSON.parse(saved); // {t: epochMs}
      if (!obj || !obj.t) return false;
      var ms = DISMISS_TTL_DAYS*24*60*60*1000;
      return (Date.now() - obj.t) < ms;
    }catch(e){ return false; }
  }

  var intro = document.getElementById('rbIntro');
  if (!intro) return;

  var flag = localStorage.getItem(STORAGE_KEY);
  var shouldShow = !flag || !ttlValid(flag);
  if (!shouldShow) return;

  document.body.classList.add('rb-intro-lock');
  intro.classList.add('show');

  var rows    = Array.prototype.slice.call(intro.querySelectorAll('.rb-row'));
  var actions = document.getElementById('rbActions');
  var btnGo   = document.getElementById('rbContinueBtn');

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
    localStorage.setItem(STORAGE_KEY, JSON.stringify({t: Date.now()}));
  }

  btnGo && btnGo.addEventListener('click', dismiss);
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape') dismiss(); }, {once:true});
})();
