// Embedding map as a keyboard-driven story (SVG, 2-D). slides.js calls slide.step(dir) on ←/→;
// it returns true while there are steps left, so the deck only advances after the last one.
(() => {
  const svg = document.getElementById('embed-map');
  const D = window.MAP_DATA;
  if (!svg || !D) return;
  const slide = svg.closest('.slide');
  const caption = document.getElementById('map-caption');
  const results = document.getElementById('map-results');
  const progress = document.getElementById('map-progress');
  const real = D.source !== 'illustrative';
  document.getElementById('map-source').textContent = real
    ? 'Real EmbeddingGemma 2 vectors (768-D), projected to 2-D with PCA. Scores are cosine similarity in the full 768-D space.'
    : 'Illustrative layout. Run demo/build_map.py to see real model output and scores.';

  const COLORS = { food: '#f6c344', places: '#7fcf98', tech: '#6ea0ff', weather: '#b49cff', sports: '#f28b82' };
  const NAMES = { food: 'Food', places: 'Places', tech: 'Tech', weather: 'Weather', sports: 'Sports' };
  const W = 1000, H = 620, PAD = 70;
  const NS = 'http://www.w3.org/2000/svg';

  // ---- Positions: real PCA output, or a seeded illustrative layout ----
  const C = { food: [-.68, .5], places: [-.02, .62], tech: [.7, -.55], weather: [-.55, -.6], sports: [.62, .32] };
  const BRIDGE = { 'Snake boat race at Punnamada lake': 'places', 'Where can I get a good sadya on Onam?': 'places', 'Weekend trip from Kochi to the hills': 'places' };
  let seed = 11;
  const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - .5;
  const place = (g, t) => { const a = C[g], b = C[BRIDGE[t] || g], m = BRIDGE[t] ? .5 : 0; const j = m ? .06 : 1; return [a[0] * (1 - m) + b[0] * m + rand() * .42 * j, a[1] * (1 - m) + b[1] * m + rand() * .34 * j]; };
  const pts = D.points.map(p => ({ ...p, pos: p.p || place(p.g, p.t) }));
  const qs = (D.queries || []).map(q => ({ ...q, pos: q.p || C[q.g].map(v => v * .8 + rand() * .1) }));
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const nearest = (pos, skip) => pts.map((p, j) => [j, dist(pos, p.pos)]).filter(([j]) => j !== skip).sort((a, b) => a[1] - b[1]).slice(0, 3).map(([j]) => [j, null]);
  pts.forEach((p, i) => { if (!p.n) p.n = nearest(p.pos, i); });
  qs.forEach(q => { if (!q.n) q.n = nearest(q.pos, -1); });
  const X = v => PAD + (v + 1) / 2 * (W - 2 * PAD), Y = v => H - PAD - (v + 1) / 2 * (H - 2 * PAD);
  const idx = t => pts.findIndex(p => p.t === t);

  // ---- The story ----
  const first = idx('Appam with vegetable stew for breakfast');
  const boat = idx('Snake boat race at Punnamada lake');
  const STEPS = [
    { show: [first], focus: first, say: 'Every sentence becomes 768 numbers. We can’t draw 768 dimensions, so this map squashes them down to 2. Here is our first sentence.' },
    { reveal: ['food'], say: 'Now add the other food sentences. They land close together, and nobody told the model they were about food.' },
    { reveal: ['food', 'places'], say: 'Add six sentences about Kerala places: backwaters, Munnar, Fort Kochi. They form their own neighbourhood, away from the food.' },
    { reveal: ['food', 'places', 'tech'], say: 'Six tech sentences land far from both. Kubernetes, Python errors and GPUs have nothing in common with appam.' },
    { reveal: ['food', 'places', 'tech', 'weather'], say: 'Weather and monsoon sentences group together, including the Malayalam one: “മഴ ശക്തമായി പെയ്യുന്നു” means “it’s raining heavily”.' },
    { reveal: ['food', 'places', 'tech', 'weather', 'sports'], say: 'Finally, six sports sentences: cricket, football, a snake boat race. That’s all 30.' },
    { show: 'all', groups: true, say: 'All 30 sentences. Five topics form five neighbourhoods. That’s the whole idea: <b>similar meaning, nearby point</b>.' },
    { show: 'all', query: 0, say: 'Search drops the query onto the same map and takes the closest points. The results are all food, and most don’t contain the word “breakfast” at all.' },
    { show: 'all', query: 1, say: 'This query is in Malayalam: “Will it rain in Kochi today?” It still lands in the weather neighbourhood, next to the English sentences.' },
    { show: 'all', query: 2, say: '“My container keeps crashing” shares no words with “Restart the crashed Kubernetes pod”, but it means nearly the same thing.' },
    { show: 'all', focus: boat, say: 'A snake boat race is both a sport and a place, so it sits <b>between</b> the two neighbourhoods. Meaning is a position, not a single label.' },
    { show: 'all', groups: true, say: 'One caveat: this is a 2-D shadow of a 768-D space. The search itself uses all 768 numbers, so it’s more precise than the picture.' },
  ];

  // ---- Build the SVG once; each step only changes attributes (CSS transitions animate them) ----
  const el = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const gRing = el('g', {}), gLinks = el('g', {}), gPts = el('g', {}), gLbl = el('g', {}), gGrp = el('g', {}), gQ = el('g', {});
  const ring = el('circle', { class: 'ring', fill: 'none', stroke: '#ffffff', 'stroke-opacity': .35, 'stroke-dasharray': '4 6', 'stroke-width': 1.5, opacity: 0, r: 0 }, gRing);
  const links = [0, 1, 2].map(() => el('line', { class: 'link', stroke: '#ffffff', 'stroke-opacity': .5, 'stroke-width': 1.5, opacity: 0 }, gLinks));
  const dots = pts.map(p => el('circle', { class: 'pt', cx: W / 2, cy: H / 2, r: 7, fill: COLORS[p.g], opacity: 0 }, gPts));
  // Leader lines connect each dot to its label when a topic's labels are stacked in a column.
  const leaders = pts.map(() => el('line', { class: 'lbl', stroke: '#ffffff', 'stroke-opacity': .28, 'stroke-width': 1, opacity: 0 }, gLbl));
  const labels = pts.map(p => { const t = el('text', { class: 'lbl', 'font-size': 17, fill: '#e8eaed', opacity: 0, 'paint-order': 'stroke', stroke: '#080b10', 'stroke-width': 6, 'stroke-linejoin': 'round' }, gLbl); t.textContent = p.t; return t; });
  const groupLabels = Object.keys(NAMES).map(g => {
    const m = pts.filter(p => p.g === g), cx = m.reduce((s, p) => s + X(p.pos[0]), 0) / m.length, top = Math.min(...m.map(p => Y(p.pos[1])));
    const t = el('text', { class: 'grp', x: cx, y: Math.max(28, top - 22), 'text-anchor': 'middle', 'font-size': 20, 'font-weight': 600, fill: COLORS[g], opacity: 0, 'paint-order': 'stroke', stroke: '#080b10', 'stroke-width': 6 }, gGrp);
    t.textContent = NAMES[g]; return t;
  });
  const badges = [0, 1, 2].map(k => { const g = el('g', { class: 'lbl', opacity: 0 }, gQ); el('circle', { r: 11, fill: '#f1f3f5' }, g); const t = el('text', { 'text-anchor': 'middle', y: 5, 'font-size': 14, 'font-weight': 700, fill: '#080b10' }, g); t.textContent = k + 1; return g; });
  const qMark = el('g', { class: 'q', opacity: 0 }, gQ);
  el('rect', { width: 18, height: 18, x: -9, y: -9, transform: 'rotate(45)', fill: '#ffffff' }, qMark);
  const qText = el('text', { x: 0, y: -24, 'font-size': 19, 'font-weight': 600, fill: '#ffffff', 'paint-order': 'stroke', stroke: '#080b10', 'stroke-width': 6 }, qMark);

  function placeLabel(t, x, y) {
    const right = x < W * .62; // flip to the left of the dot near the right edge
    t.setAttribute('x', right ? x + 14 : x - 14); t.setAttribute('y', y + 6);
    t.setAttribute('text-anchor', right ? 'start' : 'end');
  }
  let step = 0;
  function render() {
    const s = STEPS[step];
    // reveal: topics shown so far; the newest topic gets its sentence labels so every sentence is read on the map.
    const newest = s.reveal ? s.reveal[s.reveal.length - 1] : null;
    const visible = i => s.show === 'all' || (s.reveal ? s.reveal.includes(pts[i].g) : s.show.includes(i));
    const q = s.query != null ? qs[s.query] : null;
    const spotlight = q || (s.focus != null && s.show === 'all');
    const hub = q ? q.pos : spotlight ? pts[s.focus].pos : null;
    const near = q ? q.n : spotlight ? pts[s.focus].n : [];
    const lit = new Set(near.map(([j]) => j)); if (s.focus != null) lit.add(s.focus);
    dots.forEach((d, i) => {
      d.setAttribute('cx', visible(i) ? X(pts[i].pos[0]) : W / 2); d.setAttribute('cy', visible(i) ? Y(pts[i].pos[1]) : H / 2);
      d.setAttribute('opacity', !visible(i) ? 0 : spotlight && !lit.has(i) ? .16 : newest && pts[i].g !== newest ? .45 : 1);
      d.setAttribute('r', i === s.focus ? 10 : spotlight && lit.has(i) ? 8.5 : 7);
    });
    // Stacked layout for the newest topic: labels in an evenly spaced column beside the cluster, so none overlap.
    const stack = new Map();
    if (newest) {
      const members = pts.map((p, i) => i).filter(i => pts[i].g === newest).sort((a, b) => Y(pts[a].pos[1]) - Y(pts[b].pos[1]));
      const xs = members.map(i => X(pts[i].pos[0])), cy = members.reduce((t, i) => t + Y(pts[i].pos[1]), 0) / members.length;
      const right = (Math.min(...xs) + Math.max(...xs)) / 2 < W * .55, gap = 27;
      const colX = right ? Math.max(...xs) + 44 : Math.min(...xs) - 44;
      const top = Math.min(Math.max(cy - (members.length - 1) * gap / 2, 30), H - 30 - (members.length - 1) * gap);
      members.forEach((i, k) => stack.set(i, { x: colX, y: top + k * gap, anchor: right ? 'start' : 'end' }));
    }
    labels.forEach((t, i) => {
      const show = visible(i) && ((newest && pts[i].g === newest) || i === s.focus);
      const st = stack.get(i), dx = X(pts[i].pos[0]), dy = Y(pts[i].pos[1]);
      if (st) {
        t.setAttribute('x', st.x); t.setAttribute('y', st.y + 6); t.setAttribute('text-anchor', st.anchor);
        const l = leaders[i]; l.setAttribute('x1', dx); l.setAttribute('y1', dy); l.setAttribute('x2', st.x + (st.anchor === 'start' ? -6 : 6)); l.setAttribute('y2', st.y);
      } else placeLabel(t, dx, dy);
      leaders[i].setAttribute('opacity', show && st ? 1 : 0);
      t.setAttribute('opacity', show ? 1 : 0);
    });
    groupLabels.forEach(t => t.setAttribute('opacity', s.groups ? 1 : 0));
    badges.forEach((b, k) => {
      const j = spotlight && near[k] ? near[k][0] : null;
      if (j == null) { b.setAttribute('opacity', 0); return; }
      const x = X(pts[j].pos[0]), y = Y(pts[j].pos[1]), dx = hub ? x - X(hub[0]) : 1, dy = hub ? y - Y(hub[1]) : 0, d = Math.hypot(dx, dy) || 1;
      b.setAttribute('transform', `translate(${x + dx / d * 20},${y + dy / d * 20})`); b.setAttribute('opacity', 1);
    });
    links.forEach((l, k) => {
      const j = near[k] ? near[k][0] : null;
      if (hub && j != null) { l.setAttribute('x1', X(hub[0])); l.setAttribute('y1', Y(hub[1])); l.setAttribute('x2', X(pts[j].pos[0])); l.setAttribute('y2', Y(pts[j].pos[1])); l.setAttribute('opacity', 1); }
      else l.setAttribute('opacity', 0);
    });
    if (q) {
      const qx = X(q.pos[0]), qy = Y(q.pos[1]);
      const far = Math.max(...q.n.map(([j]) => Math.hypot(X(pts[j].pos[0]) - qx, Y(pts[j].pos[1]) - qy)));
      ring.setAttribute('cx', qx); ring.setAttribute('cy', qy); ring.setAttribute('r', far + 14); ring.setAttribute('opacity', 1);
      qMark.setAttribute('transform', `translate(${qx},${qy})`); qMark.setAttribute('opacity', 1);
      qText.textContent = `“${q.t}”`;
      qText.setAttribute('text-anchor', qx > W * .7 ? 'end' : qx < W * .3 ? 'start' : 'middle');
    } else { ring.setAttribute('opacity', 0); qMark.setAttribute('opacity', 0); }

    caption.innerHTML = s.say;
    results.innerHTML = spotlight ? near.map(([j, sc], k) => `<div><b class="rank">${k + 1}</b><i class="dot" style="background:${COLORS[pts[j].g]}"></i>${pts[j].t}<span>${sc != null ? sc.toFixed(2) : ''}</span></div>`).join('') : '';
    progress.innerHTML = STEPS.map((_, k) => `<i class="${k <= step ? 'on' : ''}"></i>`).join('') + `<span>${step + 1} / ${STEPS.length}</span>`;
    slide.stepInfo = { i: step + 1, n: STEPS.length };
  }
  slide.step = dir => { const n = step + dir; if (n < 0 || n >= STEPS.length) return false; step = n; render(); return true; };
  slide.onEnter = dir => { step = dir < 0 ? STEPS.length - 1 : 0; render(); };
  render();
})();
