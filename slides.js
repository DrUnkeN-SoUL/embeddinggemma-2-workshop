// Deck navigation. Keyboard: ←/→, Space, PageUp/PageDown, Home/End, F (fullscreen), B (blank screen).
// When served by server.js, the deck also follows a remote control over WebSocket and reports its state back.
const slides = [...document.querySelectorAll('.slide')];
let current = 0;
let socket = null;
const params = new URLSearchParams(location.search);
const PREVIEW = params.has('preview'); // read-only mirror inside the remote page

function show(index, dir = 1) {
  current = Math.max(0, Math.min(slides.length - 1, index));
  const glow = [[12, 7], [5, 12], [-14, 4], [9, -8], [-5, 10]][current % 5];
  document.body.style.setProperty('--glow-x', `${glow[0]}%`);
  document.body.style.setProperty('--glow-y', `${glow[1]}%`);
  slides.forEach((s, i) => { s.hidden = i !== current; });
  document.getElementById('slide-label').textContent = slides[current].getAttribute('aria-label');
  document.getElementById('counter').textContent = `${String(current + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}`;
  history.replaceState(null, '', `#${current + 1}`);
  if (slides[current].onEnter) slides[current].onEnter(dir);
  report();
}
// Slides with internal steps (the embedding map) consume presses until their steps run out.
function move(dir) {
  const s = slides[current];
  if (s.step && s.step(dir)) report();
  else show(current + dir, dir);
}
function blank(on = !document.body.classList.contains('blank')) { document.body.classList.toggle('blank', on); report(); }
function fromHash() { const n = Number(location.hash.slice(1)); show(Number.isInteger(n) && n > 0 ? n - 1 : 0); }
async function fullscreen() { try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch { /* unavailable */ } }

addEventListener('keydown', e => {
  if (PREVIEW) return;
  if (e.ctrlKey || e.metaKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
  if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); move(1); }
  else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); move(-1); }
  else if (e.key === 'Home') show(0, -1);
  else if (e.key === 'End') show(slides.length - 1);
  else if (e.key.toLowerCase() === 'f') fullscreen();
  else if (e.key.toLowerCase() === 'b' || e.key === '.') blank();
});
addEventListener('hashchange', fromHash);

// ---- Remote control (only when served over http by server.js) ----
function state() {
  const s = slides[current], n = slides[current + 1], notes = s.querySelector('aside.notes');
  return {
    type: 'state', slide: current + 1, total: slides.length,
    label: s.getAttribute('aria-label'), next: n ? n.getAttribute('aria-label') : null,
    step: s.stepInfo || null, notes: notes ? notes.innerHTML : '',
    blank: document.body.classList.contains('blank'),
    outline: slides.map(x => x.getAttribute('aria-label')),
  };
}
function report() { if (!PREVIEW && socket && socket.readyState === 1) socket.send(JSON.stringify(state())); }
// Preview: jump to the display's slide, then walk the in-slide steps to match.
function mirror(m) {
  if (current !== m.slide - 1) show(m.slide - 1, m.slide - 1 < current ? -1 : 1);
  const s = slides[current];
  if (s.step && s.stepInfo && m.step) {
    let guard = 50;
    while (s.stepInfo.i < m.step.i && guard-- && s.step(1));
    while (s.stepInfo.i > m.step.i && guard-- && s.step(-1));
  }
  document.body.classList.toggle('blank', !!m.blank);
}
function connect() {
  if (!location.protocol.startsWith('http')) return;
  const role = PREVIEW ? `preview&key=${encodeURIComponent(params.get('key') || '')}` : 'display';
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws?role=${role}`);
  socket.onopen = report;
  socket.onmessage = e => {
    let m; try { m = JSON.parse(e.data); } catch { return; }
    if (PREVIEW) { if (m.type === 'state') mirror(m); return; }
    if (m.type === 'nav') move(m.dir > 0 ? 1 : -1);
    else if (m.type === 'goto') show(m.slide - 1, m.slide - 1 < current ? -1 : 1);
    else if (m.type === 'blank') blank(m.on);
    else if (m.type === 'hello') report();
  };
  socket.onclose = () => setTimeout(connect, 1500);
}
connect();
