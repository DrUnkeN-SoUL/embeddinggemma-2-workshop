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
  const qr = qrcode(0, 'M');
  qr.addData(url); qr.make();
  // Dark modules on a white tile: phone cameras read black-on-white most reliably.
  box.innerHTML = qr.createSvgTag({ cellSize: 8, margin: 4, scalable: true });
  caption.textContent = url.replace(/^https?:\/\//, '');
})();
