(function(){
  // Config (placeholder caption list; replace later)
  const RBX_CAPTIONS = ["Placeholder caption text"];
  const RBX_ROTATE_MS = 2600;
  const RBX_FADE_MS = 450;

  function rbxFillBars(){
    document.querySelectorAll(".rbx-bar-fill").forEach(el=>{
      const v = el.getAttribute("data-rbx-value");
      if(!v) return;
      // kick in on next frame to ensure transition
      requestAnimationFrame(()=>{ el.style.width = v + "%"; });
    });
  }

  function rbxRotateCaption(){
    const el = document.getElementById("rbx-rotating-caption");
    if(!el || RBX_CAPTIONS.length === 0) return;
    let i = 0;
    function next(){
      el.classList.add("rbx-cap-out");
      setTimeout(()=>{
        el.textContent = RBX_CAPTIONS[i];
        el.classList.remove("rbx-cap-out");
        el.classList.add("rbx-cap-in");
        setTimeout(()=> el.classList.remove("rbx-cap-in"), RBX_FADE_MS+20);
        i = (i + 1) % RBX_CAPTIONS.length;
      }, RBX_FADE_MS);
    }
    let timer = setInterval(next, RBX_ROTATE_MS);
    el.addEventListener("mouseenter", ()=> clearInterval(timer));
    el.addEventListener("mouseleave", ()=> timer = setInterval(next, RBX_ROTATE_MS));
    el.addEventListener("focus", ()=> clearInterval(timer), true);
    el.addEventListener("blur", ()=> timer = setInterval(next, RBX_ROTATE_MS), true);
  }

  function rbxOverlayControl(){
    const overlay = document.querySelector(".rbx-intro-overlay");
    const continueBtn = document.getElementById("rbx-continue");
    if(!overlay || !continueBtn) return;

    // lock scroll while overlay visible
    document.body.style.overflow = "hidden";

    continueBtn.addEventListener("click", function(e){
      e.preventDefault();
      overlay.classList.add("rbx-fade-out");
      setTimeout(()=>{
        overlay.classList.add("rbx-hidden");
        document.body.style.overflow = "auto";
      }, 280); // match CSS transition
    }, { once: true });
  }

  document.addEventListener("DOMContentLoaded", function(){
    rbxFillBars();
    rbxRotateCaption();
    rbxOverlayControl();
  });
})();
