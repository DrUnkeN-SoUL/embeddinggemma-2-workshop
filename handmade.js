// "Build an embedding by hand": Kerala dishes scored on questions we choose ourselves.
// Keyboard-stepped like map.js (slides.js calls slide.step / slide.onEnter).
(() => {
  const svg = document.getElementById('hand-map');
  if (!svg) return;
  const slide = svg.closest('.slide');
  const caption = document.getElementById('hand-caption');
  const panel = document.getElementById('hand-panel');
  const progress = document.getElementById('hand-progress');
  const NS = 'http://www.w3.org/2000/svg';
  const W = 1000, H = 620, L = 90, R = 40, T = 40, B = 80;

  // [spicy, sweet, breakfast] on a 0–10 scale: our own rough ratings, not model output.
  const DISHES = [
    ['Appam', 1, 3, 9], ['Puttu', 0.8, 1.2, 9], ['Idiyappam', 2, 2, 8], ['Banana chips', 1.6, 0.4, 1],
    ['Palada payasam', 0.5, 9.5, 1], ['Unniyappam', 1, 8, 3], ['Pazham pori', 0.6, 6.5, 4],
    ['Chicken stew', 5, 3, 6], ['Mango pickle', 8.5, 3.5, 3],
    ['Meen curry', 8, 1.8, 2], ['Kappa & meen curry', 7, 0.5, 2], ['Beef fry', 9, 0.5, 1],
  ].map(([name, spicy, sweet, bf]) => ({ name, v: [spicy, sweet, bf] }));
  const QUERIES = { sweet: { t: 'Something sweet after lunch', v: [1.6, 8.8, 1] }, spicy: { t: 'Something spicy for dinner', v: [9, 0.6, 1] } };
  const at = n => DISHES.findIndex(d => d.name === n);
  const PUTTU = at('Puttu');
  const dist = (a, b, dims) => Math.hypot(...a.slice(0, dims).map((x, i) => x - b[i]));
  const nearest = (v, dims, skip = -1) => DISHES.map((d, i) => [i, dist(v, d.v, dims)]).filter(([i]) => i !== skip).sort((a, b) => a[1] - b[1]).slice(0, 3);

  const STEPS = [
    { show: [at('Appam')], say: 'Let’s describe food with two questions. <b>How spicy is it?</b> <b>How sweet is it?</b> Each answer is a number from 0 to 10. Appam: spicy 1, sweet 3.' },
    { show: 'all', say: 'Score twelve dishes the same way and each becomes a point. Similar dishes land near each other without anyone sorting them: sweets top-left, curries on the right.' },
    { show: 'all', query: 'sweet', say: 'A question is just another point. “Something sweet after lunch” is roughly spicy 2, sweet 9. Search means taking the nearest dots.' },
    { show: 'all', query: 'spicy', say: '“Something spicy for dinner” lands among the curries. No keywords involved: just distance.' },
    { show: 'all', focus: PUTTU, dims: 2, say: 'But two questions aren’t enough. Puttu’s nearest neighbour is <b>banana chips</b>, because both are neither spicy nor sweet. They are very different foods.' },
    { show: 'all', focus: PUTTU, dims: 3, vectors: true, say: 'Add a third question, <b>“Is it eaten at breakfast?”</b>, and the mistake fixes itself. Each dish is now a list of three numbers. <b>That list is an embedding.</b>' },
    { show: 'all', final: true, say: 'An embedding model does exactly this with <b>768 questions</b>. Nobody chooses or names them: the model learns them from millions of examples, for sentences, photos and sounds.' },
  ];

  const el = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
  const X = v => L + v / 10 * (W - L - R), Y = v => H - B - v / 10 * (H - T - B);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  // Axes with plain-language titles
  el('line', { x1: L, y1: H - B, x2: W - R, y2: H - B, stroke: '#3a404b', 'stroke-width': 1.5 });
  el('line', { x1: L, y1: H - B, x2: L, y2: T, stroke: '#3a404b', 'stroke-width': 1.5 });
  [0, 5, 10].forEach(v => {
    el('text', { x: X(v), y: H - B + 26, 'text-anchor': 'middle', 'font-size': 15, fill: '#6d7480' }).textContent = v;
    el('text', { x: L - 14, y: Y(v) + 5, 'text-anchor': 'end', 'font-size': 15, fill: '#6d7480' }).textContent = v;
  });
  el('text', { x: (L + W - R) / 2, y: H - 22, 'text-anchor': 'middle', 'font-size': 18, 'font-weight': 600, fill: '#f28b82' }).textContent = 'How spicy? →';
  el('text', { x: 28, y: (T + H - B) / 2, 'text-anchor': 'middle', 'font-size': 18, 'font-weight': 600, fill: '#f6c344', transform: `rotate(-90 28 ${(T + H - B) / 2})` }).textContent = 'How sweet? →';

  const gRing = el('g', {}), gLinks = el('g', {}), gPts = el('g', {}), gQ = el('g', {});
  const ring = el('circle', { class: 'ring', fill: 'none', stroke: '#fff', 'stroke-opacity': .35, 'stroke-dasharray': '4 6', 'stroke-width': 1.5, opacity: 0 }, gRing);
  const links = [0, 1, 2].map(() => el('line', { class: 'link', stroke: '#fff', 'stroke-opacity': .5, 'stroke-width': 1.5, opacity: 0 }, gLinks));
  const items = DISHES.map(d => {
    const g = el('g', { class: 'pt-g', opacity: 0 }, gPts);
    const x = X(d.v[0]), y = Y(d.v[1]), right = x < W * .6;
    el('circle', { cx: x, cy: y, r: 8, fill: '#6ea0ff' }, g);
    const t = el('text', { x: right ? x + 14 : x - 14, y: y + 6, 'text-anchor': right ? 'start' : 'end', 'font-size': 17, fill: '#e8eaed', 'paint-order': 'stroke', stroke: '#080b10', 'stroke-width': 6 }, g);
    t.textContent = d.name;
    return g;
  });
  const qMark = el('g', { class: 'q', opacity: 0 }, gQ);
  el('rect', { width: 18, height: 18, x: -9, y: -9, transform: 'rotate(45)', fill: '#fff' }, qMark);
  const qText = el('text', { y: -22, 'text-anchor': 'middle', 'font-size': 18, 'font-weight': 600, fill: '#fff', 'paint-order': 'stroke', stroke: '#080b10', 'stroke-width': 6 }, qMark);

  const row = (i, k, extra = '') => `<div><b class="rank">${k + 1}</b>${DISHES[i].name}<span>${extra}</span></div>`;
  const vec = v => `[${v.map(n => String(n).padStart(3, ' ')).join(', ')}]`;
  let step = 0;
  function render() {
    const s = STEPS[step];
    const q = s.query ? QUERIES[s.query] : null;
    const hubV = q ? q.v : s.focus != null ? DISHES[s.focus].v : null;
    const near = hubV ? nearest(hubV, s.dims || 2, s.focus ?? -1) : [];
    const lit = new Set(near.map(([i]) => i)); if (s.focus != null) lit.add(s.focus);
    items.forEach((g, i) => {
      const vis = s.show === 'all' || s.show.includes(i);
      g.setAttribute('opacity', !vis ? 0 : hubV && !lit.has(i) ? .22 : s.final ? .5 : 1);
    });
    links.forEach((l, k) => {
      const j = near[k] && near[k][0];
      if (hubV && j != null) { l.setAttribute('x1', X(hubV[0])); l.setAttribute('y1', Y(hubV[1])); l.setAttribute('x2', X(DISHES[j].v[0])); l.setAttribute('y2', Y(DISHES[j].v[1])); l.setAttribute('opacity', 1); }
      else l.setAttribute('opacity', 0);
    });
    if (q) {
      const qx = X(q.v[0]), qy = Y(q.v[1]);
      const far = Math.max(...near.map(([j]) => Math.hypot(X(DISHES[j].v[0]) - qx, Y(DISHES[j].v[1]) - qy)));
      ring.setAttribute('cx', qx); ring.setAttribute('cy', qy); ring.setAttribute('r', far + 16); ring.setAttribute('opacity', 1);
      qMark.setAttribute('transform', `translate(${qx},${qy})`); qMark.setAttribute('opacity', 1);
      qText.textContent = `“${q.t}”`;
      // Left half: label beside the diamond (open space to the right). Right half: label above it.
      const left = qx < W * .5;
      qText.setAttribute('text-anchor', left ? 'start' : 'end'); qText.setAttribute('x', left ? 20 : 10); qText.setAttribute('y', left ? 6 : -22);
    } else { ring.setAttribute('opacity', 0); qMark.setAttribute('opacity', 0); }

    caption.innerHTML = s.say;
    if (s.vectors) {
      const two = nearest(DISHES[PUTTU].v, 2, PUTTU)[0][0], three = nearest(DISHES[PUTTU].v, 3, PUTTU);
      panel.innerHTML = `<div class="vec-table">${[PUTTU, at('Idiyappam'), at('Appam'), at('Banana chips')].map(i =>
        `<div><span>${DISHES[i].name}</span><code>${vec(DISHES[i].v)}</code></div>`).join('')}<p class="vec-head">spicy, sweet, breakfast</p></div>
        <p class="hand-compare">Puttu’s nearest with 2 questions: <b>${DISHES[two].name}</b>. With 3: <b>${DISHES[three[0][0]].name}</b>.</p>`;
    } else if (s.final) {
      panel.innerHTML = `<div class="vec-table"><div><span>Our dishes</span><code>3 numbers, questions we named</code></div><div><span>EmbeddingGemma 2</span><code>768 numbers, questions it learned</code></div></div>`;
    } else panel.innerHTML = near.map(([i], k) => row(i, k)).join('');
    progress.innerHTML = STEPS.map((_, k) => `<i class="${k <= step ? 'on' : ''}"></i>`).join('') + `<span>${step + 1} / ${STEPS.length}</span>`;
    slide.stepInfo = { i: step + 1, n: STEPS.length };
  }
  slide.step = dir => { const n = step + dir; if (n < 0 || n >= STEPS.length) return false; step = n; render(); return true; };
  slide.onEnter = dir => { step = dir < 0 ? STEPS.length - 1 : 0; render(); };
  render();
})();
