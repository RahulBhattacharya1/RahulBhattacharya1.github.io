const fab = document.getElementById('fab');
const fabMenu = document.getElementById('fabMenu');
const contactParent = document.getElementById('contactParent');
const contactSub = document.getElementById('contactSub');

// Delayed dramatic reveal + lock
setTimeout(() => {
  if (!fab) return;
  fab.classList.add('reveal');
  fab.addEventListener('animationend', () => fab.classList.add('settled'), { once: true });
}, 1200);

let hideTimeout = null;

function openMenu() {
  fabMenu.classList.add('show');
  fab.setAttribute('aria-expanded', 'true');
  restartAutoClose();
}
function closeMenu() {
  fabMenu.classList.remove('show');
  fab.setAttribute('aria-expanded', 'false');
  contactSub.classList.remove('show');
  clearTimeout(hideTimeout);
  hideTimeout = null;
}
function restartAutoClose() {
  clearTimeout(hideTimeout);
  hideTimeout = setTimeout(closeMenu, 5000);
}

// Toggle main menu
fab.addEventListener('click', (e) => {
  e.stopPropagation();
  if (fabMenu.classList.contains('show')) closeMenu();
  else openMenu();
});

// Toggle contact submenu
contactParent.addEventListener('click', (e) => {
  e.stopPropagation();
  contactSub.classList.toggle('show');
  restartAutoClose();
});

// Outside click / ESC closes
document.addEventListener('click', (e) => {
  if (!e.target.closest('.fab, .fab-menu')) closeMenu();
}, { passive: true });

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeMenu();
});
