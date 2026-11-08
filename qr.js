// Draws the Colab QR code from the data-url on #colab-qr (uses the vendored qrcode-generator, works offline).
(() => {
  const box = document.getElementById('colab-qr');
  const caption = document.getElementById('colab-qr-caption');
  if (!box) return;
  const url = (box.dataset.url || '').trim();
  if (!url || typeof qrcode !== 'function') {
    box.classList.add('qr-missing');
    box.textContent = 'Add the Colab link to data-url on #colab-qr in index.html';
    return;
  }
  const qr = qrcode(0, 'L');
  qr.addData(url); qr.make();
  // Level L keeps the code at fewer, larger modules (easier to scan from the back of the room). Dark modules on a white tile: phone cameras read black-on-white most reliably.
  box.innerHTML = qr.createSvgTag({ cellSize: 10, margin: 4, scalable: true });
  const svg = box.querySelector('svg'); if (svg) svg.setAttribute('shape-rendering', 'crispEdges');
  caption.textContent = url.replace(/^https?:\/\//, '');
})();
