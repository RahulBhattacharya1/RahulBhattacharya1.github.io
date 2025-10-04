  const fab = document.getElementById('fab');
  const fabMenu = document.getElementById('fabMenu');
  const contactParent = document.getElementById('contactParent');
  const contactSub = document.getElementById('contactSub');

  setTimeout(() => {
    fab.classList.add('reveal');
    fab.addEventListener('animationend', () => fab.classList.add('settled'), { once:true });
  }, 1200);

  let hideTimeout = null;
  function openMenu(){ fabMenu.classList.add('show'); fab.setAttribute('aria-expanded','true'); restartAutoClose(); }
  function closeMenu(){ fabMenu.classList.remove('show'); fab.setAttribute('aria-expanded','false'); contactSub.classList.remove('show'); clearTimeout(hideTimeout); hideTimeout=null; }
  function restartAutoClose(){ clearTimeout(hideTimeout); hideTimeout=setTimeout(closeMenu, 5000); }

  fab.addEventListener('click', e=>{ e.stopPropagation(); fabMenu.classList.contains('show')? closeMenu():openMenu(); });
  contactParent.addEventListener('click', e=>{ e.stopPropagation(); contactSub.classList.toggle('show'); restartAutoClose(); });

  document.addEventListener('click', e=>{ if(!e.target.closest('.fab, .fab-menu')) closeMenu(); });
  document.addEventListener('keydown', e=>{ if(e.key==='Escape') closeMenu(); });
