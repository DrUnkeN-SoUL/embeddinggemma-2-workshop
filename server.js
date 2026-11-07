#!/usr/bin/env node
// Presenter server: serves the deck and relays a WebSocket remote control. No dependencies.
//
//   node server.js            (PORT=8080 and REMOTE_KEY=<random> by default)
//
// Open the printed "Display" URL on the projector and the "Remote" URL on your phone or laptop.
// Only connections carrying the remote key can control the slides.
const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const os = require('os');

const PORT = Number(process.env.PORT) || 8080;
const KEY = process.env.REMOTE_KEY || crypto.randomBytes(3).toString('hex');
const ROOT = __dirname;
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.json': 'application/json', '.ipynb': 'application/json' };

// ---------- static files ----------
const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  let rel = decodeURIComponent(url.pathname);
  if (rel === '/') rel = '/index.html';
  if (rel === '/remote') rel = '/remote.html';
  const file = path.normalize(path.join(ROOT, rel));
  if (!file.startsWith(ROOT + path.sep) || path.basename(file).startsWith('.')) { res.writeHead(403).end(); return; }
  fs.readFile(file, (err, body) => {
    if (err) { res.writeHead(404).end('Not found'); return; }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    res.end(body);
  });
});

// ---------- minimal WebSocket (RFC 6455, text frames only) ----------
const displays = new Set(), remotes = new Set(), previews = new Set();
let lastState = null;

function sendFrame(sock, text) {
  const data = Buffer.from(text);
  const len = data.length;
  const head = len < 126 ? Buffer.from([0x81, len])
    : len < 65536 ? Buffer.from([0x81, 126, len >> 8, len & 255])
    : Buffer.concat([Buffer.from([0x81, 127]), (() => { const b = Buffer.alloc(8); b.writeBigUInt64BE(BigInt(len)); return b; })()]);
  if (!sock.destroyed) sock.write(Buffer.concat([head, data]));
}
function control(sock, opcode, payload = Buffer.alloc(0)) {
  if (!sock.destroyed) sock.write(Buffer.concat([Buffer.from([0x80 | opcode, payload.length]), payload]));
}
function onMessage(sock, role, text) {
  let m; try { m = JSON.parse(text); } catch { return; }
  if (role === 'display' && m.type === 'state') {
    lastState = text;
    remotes.forEach(r => sendFrame(r, text));
    previews.forEach(r => sendFrame(r, text));
  } else if (role === 'remote' && ['nav', 'goto', 'blank'].includes(m.type)) {
    displays.forEach(d => sendFrame(d, JSON.stringify(m)));
  }
}

server.on('upgrade', (req, sock) => {
  const url = new URL(req.url, 'http://x');
  const role = url.searchParams.get('role');
  const wsKey = req.headers['sec-websocket-key'];
  if (url.pathname !== '/ws' || !wsKey || !['display', 'remote', 'preview'].includes(role) ||
      (role !== 'display' && url.searchParams.get('key') !== KEY)) {
    sock.end('HTTP/1.1 403 Forbidden\r\n\r\n'); return;
  }
  const accept = crypto.createHash('sha1').update(wsKey + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
  sock.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
  sock.setNoDelay(true);
  const group = role === 'display' ? displays : role === 'remote' ? remotes : previews;
  group.add(sock);
  if (role !== 'display' && lastState) sendFrame(sock, lastState);
  if (role === 'remote') {
    displays.forEach(d => sendFrame(d, JSON.stringify({ type: 'hello' })));
  }
  console.log(`+ ${role} connected (${displays.size} display, ${remotes.size} remote)`);

  let buf = Buffer.alloc(0), fragments = [];
  sock.on('data', chunk => {
    buf = Buffer.concat([buf, chunk]);
    while (buf.length >= 2) {
      const fin = buf[0] & 0x80, opcode = buf[0] & 0x0f, masked = buf[1] & 0x80;
      let len = buf[1] & 0x7f, off = 2;
      if (len === 126) { if (buf.length < 4) return; len = buf.readUInt16BE(2); off = 4; }
      else if (len === 127) { if (buf.length < 10) return; len = Number(buf.readBigUInt64BE(2)); off = 10; }
      if (len > 1 << 20) { sock.destroy(); return; } // 1 MB cap
      const maskOff = off; if (masked) off += 4;
      if (buf.length < off + len) return;
      const payload = Buffer.from(buf.subarray(off, off + len));
      if (masked) for (let i = 0; i < len; i++) payload[i] ^= buf[maskOff + (i & 3)];
      buf = buf.subarray(off + len);
      if (opcode === 0x8) { control(sock, 0x8); sock.end(); return; }
      if (opcode === 0x9) { control(sock, 0xA, payload); continue; }
      if (opcode === 0x1 || opcode === 0x0) {
        fragments.push(payload);
        if (fin) { onMessage(sock, role, Buffer.concat(fragments).toString('utf8')); fragments = []; }
      }
    }
  });
  const drop = () => { if (group.delete(sock)) console.log(`- ${role} left (${displays.size} display, ${remotes.size} remote)`); };
  sock.on('close', drop); sock.on('error', drop);
});

server.listen(PORT, '0.0.0.0', () => {
  const lan = Object.values(os.networkInterfaces()).flat().filter(i => i && i.family === 'IPv4' && !i.internal).map(i => i.address);
  console.log(`\nEmbeddingGemma 2 deck\n`);
  console.log(`  Display (projector): http://localhost:${PORT}/`);
  console.log(`  Remote (this laptop): http://localhost:${PORT}/remote?key=${KEY}`);
  lan.forEach(ip => console.log(`  Remote (phone, same Wi-Fi): http://${ip}:${PORT}/remote?key=${KEY}`));
  console.log(`\n  Remote key: ${KEY}   (set REMOTE_KEY to choose your own)\n`);
});
