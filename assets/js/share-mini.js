/* share-mini.js
   Adds a compact Share button with live-ish count.
   Works with either legacy #share-btn[...] or new .share-btn[...] markup.
   Looks for elements that have: data-share-id, optional data-url, data-share-pub.
*/
(function(){
  'use strict';

  // Utility: deterministic pseudo-random seeded by string
  function xmur3(str){
    let h = 1779033703 ^ str.length;
    for (let i=0; i<str.length; i++){
      h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
      h = (h << 13) | (h >>> 19);
    }
    return function(){
      h = Math.imul(h ^ (h >>> 16), 2246822507);
      h = Math.imul(h ^ (h >>> 13), 3266489909);
      return (h ^= h >>> 16) >>> 0;
    };
  }

  function seededInt(min, max, seedStr){
    const seed = xmur3(seedStr)();
    const n = seed / 0xffffffff;
    return Math.floor(min + n * (max - min + 1));
  }

  function daysSince(dateStr){
    if (!dateStr) return 0;
    const d = new Date(dateStr);
    if (Number.isNaN(+d)) return 0;
    const ms = Date.now() - d.getTime();
    return Math.max(0, Math.floor(ms / 86400000));
    }

  function computeBaseline(id, pubDate){
    const age = daysSince(pubDate);
    const base = seededInt(3, 21, id + '|' + (pubDate || ''));
    return base + Math.floor(age * 0.35);
  }

  function formatCount(n){
    if (n < 1000) return String(n);
    if (n < 10000) return (n/1000).toFixed(1).replace(/\.0$/,'') + 'k';
    return Math.floor(n/1000) + 'k';
  }

  function getStoreKey(id){ return 'shared:' + id; }

  function initOne(btn){
    if (btn.__shareInit) return;
    btn.__shareInit = true;

    // expect: <button ...><span class="icon">...</span><span class="count"></span></button>
    const id = btn.getAttribute('data-share-id') || '';
    const pub = btn.getAttribute('data-share-pub') || '';
    const url = btn.getAttribute('data-url') || (location.origin + location.pathname);
    const countNode = btn.querySelector('.count') || (function(){ const s=document.createElement('span'); s.className='count'; btn.appendChild(s); return s; })();

    let baseline = computeBaseline(id, pub);
    let shared = false;
    try {
      shared = localStorage.getItem(getStoreKey(id)) === '1';
    } catch(e){ /* ignore private mode errors */ }

    let shown = baseline + (shared ? 1 : 0);
    countNode.textContent = formatCount(shown);
    if (shared) btn.classList.add('shared');

    async function doShare(){
      // Try native share first
      try{
        if (navigator.share){
          await navigator.share({ title: document.title, url });
          return true;
        }
      }catch(e){
        // user cancelled or failed, continue to clipboard
      }
      // Clipboard fallback
      try{
        await navigator.clipboard.writeText(url);
        return true;
      }catch(e){
        // older browsers: hidden input fallback
        const t = document.createElement('textarea');
        t.value = url;
        t.style.position='fixed';
        t.style.opacity='0';
        document.body.appendChild(t);
        t.select();
        let ok=false;
        try{ ok = document.execCommand('copy'); }catch(e){ ok=false; }
        document.body.removeChild(t);
        return ok;
      }
    }

    btn.addEventListener('click', async function(ev){
      ev.preventDefault();
      const ok = await doShare();
      if (!ok) return;
      if (!shared){
        shared = true;
        btn.classList.add('shared');
        shown += 1;
        countNode.textContent = formatCount(shown);
        try{ localStorage.setItem(getStoreKey(id), '1'); }catch(e){}
      }
    });
  }

  function initAll(){
    const nodes = document.querySelectorAll('#share-btn[data-share-id], .share-btn[data-share-id]');
    nodes.forEach(initOne);
  }

  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }
})();
