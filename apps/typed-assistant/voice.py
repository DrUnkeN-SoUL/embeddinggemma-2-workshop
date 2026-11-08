"""Voice sidecar for the type-safe assistant: two ways to hear a spoken command, both on the laptop CPU.

    demo/.venv-audio/bin/python voice.py            # http://127.0.0.1:8092 (app.py talks to it)
    demo/.venv-audio/bin/python voice.py --test DIR # score DIR/*.wav, with DIR/list.txt lines "N|what was said"

1. Meaning straight from the sound. EmbeddingGemma 2's audio encoder (300M) turns the waveform into the
   same 768-D space as text. The vector is compared with the 8 tool prototypes, built from the same
   tool descriptions + example phrases as app.py, but embedded with this (full-precision) model. No words
   are involved, so this picks the *action* only.
2. Words, for the values. Whisper (faster-whisper "base", int8) writes a transcript. app.py runs its
   usual text pipeline on it, so "22", "bedroom" and "cool" come out as typed values and get validated.

Both run for each clip, one after the other (~1.2 s for a 3 s command). Needs a separate venv with torch (CPU), see README.
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from app import DESC_WEIGHT, PREFIX, TOOLS  # noqa: E402  (same tool table as the text path)

MODEL_ID = os.environ.get("VOICE_EMBED_MODEL", "google/embeddinggemma-2")
WHISPER = os.environ.get("VOICE_WHISPER", "base")
# Fewer threads than cores is faster here. Audio encoder on a Ryzen 7 5800H (8 cores / 16 threads),
# float32: 16 threads 2.1 s, 8 threads 1.3 s, 6 threads 0.56-0.65 s, 4 threads 0.6 s.
# bfloat16 is 4x slower on this CPU (no native bf16), so float32 it is.
THREADS = int(os.environ.get("VOICE_THREADS", "6"))
PROTO_CACHE = os.path.join(HERE, "data", "audio_protos.npz")

# Audio vs text-prototype scores sit lower than text vs text (about 0.73-0.81 for a clear command,
# 0.64-0.65 for off-topic speech on our test clips), so the sound path has its own thresholds.
AUDIO_MIN_SCORE = 0.70
AUDIO_MIN_GAP = 0.05

# Whisper hint: the words this house uses, so "Achan" is not written "Aiken" and "AC" not "may C".
HINT = {
    "en": "Smart home: AC, lights, living room, bedroom, kitchen, balcony, front door, back door. "
          "Message Amma, Achan, Rahul, Office. Play Carnatic music, melody, film songs, lo-fi, devotional songs. "
          "Biryani, appam and stew, puttu and kadala, porotta and beef, sadya.",
    "ml": "ലൈറ്റ്, ഏസി, ബെഡ്റൂം, അടുക്കള, ബാൽക്കണി, അമ്മ, അച്ഛൻ, മെസ്സേജ്, ബിരിയാണി, സദ്യ, പാട്ട്, ടൈമർ.",
}


def log(*a):
    print(*a, flush=True)


def read_wav(data):
    """16-bit PCM WAV bytes -> float32 mono at 16 kHz (resampled linearly if needed)."""
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("expected 16-bit PCM")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        n = int(len(a) * 16000 / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return a


class Ears:
    def __init__(self):
        import torch
        from faster_whisper import WhisperModel
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(THREADS)
        t0 = time.perf_counter()
        self.model = SentenceTransformer(MODEL_ID, device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                         config_kwargs={"vision_config": None})  # text + audio: 570M
        self.embed_load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.whisper = WhisperModel(WHISPER, device="cpu", compute_type="int8", cpu_threads=THREADS)
        self.whisper_load_s = time.perf_counter() - t0
        self.names = [t["name"] for t in TOOLS]
        self.protos = self._protos()
        # All inference runs on one long-lived thread with the thread count set on it (OpenMP
        # settings are per thread).
        self.worker = ThreadPoolExecutor(1, initializer=torch.set_num_threads, initargs=(THREADS,))
        self.hear(np.zeros(16000, dtype=np.float32) + 1e-4)  # warm-up, on the worker thread

    def _enc(self, x):
        return self.model.encode(x, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def _protos(self):
        """Tool prototypes in this model's space, same recipe as app.py. Cached by the tool table's hash."""
        texts = [(t["description"], t["examples"]) for t in TOOLS]
        key = hashlib.sha1(json.dumps([MODEL_ID, PREFIX, DESC_WEIGHT, texts]).encode()).hexdigest()
        if os.path.exists(PROTO_CACHE):
            z = np.load(PROTO_CACHE)
            if str(z["key"]) == key:
                return z["protos"]
        t0 = time.perf_counter()
        rows = []
        for desc, ex in texts:
            e = self._enc([PREFIX + x for x in ex]).mean(axis=0)
            e /= np.linalg.norm(e)
            v = e + DESC_WEIGHT * self._enc([PREFIX + desc])[0]
            rows.append(v / np.linalg.norm(v))
        protos = np.stack(rows).astype(np.float32)
        os.makedirs(os.path.dirname(PROTO_CACHE), exist_ok=True)
        np.savez(PROTO_CACHE, protos=protos, key=np.asarray(key))
        log(f"Audio-side tool prototypes built in {time.perf_counter() - t0:.1f} s")
        return protos

    def _by_sound(self, audio):
        t0 = time.perf_counter()
        v = self._enc([{"audio": {"array": audio, "sampling_rate": 16000}}])[0]
        s = self.protos @ v
        o = np.argsort(-s)
        top, second = float(s[o[0]]), float(s[o[1]])
        if top < AUDIO_MIN_SCORE:
            conf, why = False, f"top score {top:.3f} < {AUDIO_MIN_SCORE:.2f}"
        elif top - second < AUDIO_MIN_GAP:
            conf, why = False, f"gap to #2 is {top - second:.3f} < {AUDIO_MIN_GAP:.2f}"
        else:
            conf, why = True, f"top {top:.3f}, {top - second:.3f} ahead of #2"
        return dict(ranked=[dict(tool=self.names[i], score=round(float(s[i]), 4)) for i in o],
                    confident=conf, reason=why, ms=round((time.perf_counter() - t0) * 1000))

    def _words(self, audio, lang):
        t0 = time.perf_counter()
        segs, info = self.whisper.transcribe(audio, language=None if lang == "auto" else lang, beam_size=1,
                                             initial_prompt=HINT.get(lang, HINT["en"]),
                                             condition_on_previous_text=False, without_timestamps=True,
                                             vad_filter=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        return dict(text=text, lang=info.language, ms=round((time.perf_counter() - t0) * 1000))

    def hear(self, audio, lang="en"):
        return self.worker.submit(self._hear, audio, lang).result()

    def _hear(self, audio, lang):
        t0 = time.perf_counter()
        # One after the other: on a laptop CPU the two models only slow each other down in parallel.
        sound, words = self._by_sound(audio), self._words(audio, lang)
        return dict(sound=sound, words=words, seconds=round(len(audio) / 16000, 2),
                    ms=round((time.perf_counter() - t0) * 1000))

    def info(self):
        return dict(embed_model=MODEL_ID, whisper=WHISPER, min_score=AUDIO_MIN_SCORE, min_gap=AUDIO_MIN_GAP,
                    embed_load_s=round(self.embed_load_s, 1), whisper_load_s=round(self.whisper_load_s, 1))


def make_handler(ears):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                return self.json(ears.info())
            self.json({"error": "not found"}, 404)

        def do_POST(self):
            if not self.path.startswith("/hear"):
                return self.json({"error": "not found"}, 404)
            try:
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= 16000 * 2 * 30 + 1024:
                    return self.json({"error": "send 0.1-30 s of 16-bit WAV"}, 400)
                lang = (self.path.split("lang=")[1].split("&")[0] if "lang=" in self.path else "en")
                audio = read_wav(self.rfile.read(n))
                self.json(ears.hear(audio, lang if lang in ("en", "ml", "auto") else "en"))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # bad upload: report, keep serving
                self.json({"error": repr(e)[:200]}, 400)

    return Handler


def run_test(folder):
    ears = Ears()
    for line in open(os.path.join(folder, "list.txt"), encoding="utf-8"):
        i, said = line.strip().split("|", 1)
        r = ears.hear(read_wav(open(os.path.join(folder, f"c{i}.wav"), "rb").read()))
        s = r["sound"]
        top, sec = s["ranked"][0], s["ranked"][1]
        log(f"{said[:44]:<45} sound→ {top['tool']:<13}{top['score']:.3f} (#2 {sec['score']:.3f}) "
            f"{'✓' if s['confident'] else '?'}  words→ {r['words']['text']!r}  {r['ms']} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--test", metavar="DIR")
    args = ap.parse_args()
    if args.test:
        return run_test(args.test)
    ears = Ears()
    log(f"Voice ready: {MODEL_ID} audio encoder ({ears.embed_load_s:.1f} s) + Whisper {WHISPER} "
        f"({ears.whisper_load_s:.1f} s) on http://{args.host}:{args.port}")
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(ears))
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
