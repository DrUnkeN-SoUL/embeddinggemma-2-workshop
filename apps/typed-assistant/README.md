# Type-safe on-device assistant

Type a command for a small Kerala smart home + phone. The result is always exactly one of:

- ✅ a **schema-valid typed function call**, e.g. `set_ac(room="bedroom", temperature=22, mode="cool")` plus its JSON, which then runs against an in-memory "home";
- ❓ a **clarifying question** naming the missing or invalid field ("Which room? living room / bedroom / kitchen / balcony", "Temperature must be between 16 and 30; you said 40.");
- 🤷 **"unsure which action"** with the top 2 tools to pick from.

It never produces an invalid call. **There is no LLM.** EmbeddingGemma 2 (270M, 165 MB LiteRT-LM file) runs on
the laptop CPU and decides by meaning: which tool, and (as a fallback) which value of an enum field. Everything else is
deterministic and explainable: regex for numbers, durations and times, a synonym lexicon (English, Manglish, Malayalam)
for enums and on/off words, and a validator. Nothing leaves the laptop.

## Run

```bash
apps/typed-assistant/run.sh            # http://127.0.0.1:8091  (the intent matcher uses 8090)
apps/typed-assistant/run.sh --eval     # accuracy on tests.json + threshold table + per-utterance results
```

`run.sh` uses `demo/.venv` (litert-lm 0.18.0 + numpy) and `demo/models/embeddinggemma-2-text-270m.litertlm`
(run `demo/laptop_demo.sh` once if it is missing). The first run embeds 95 texts (tool examples, tool descriptions,
enum descriptions) in about 9 s and caches them in `data/vectors.npz` (gitignored). Changing the tool table only
re-embeds the texts that changed. After that, startup takes under a second.

- The page opens on a 3-step intro for developers (`→` / `←`, `Enter` skips to the demo, `T` brings it back):
  the 8 tools as typed Python signatures, one tool as a function-calling JSON Schema next to its prototype
  recipe and validator rules, and the per-message pipeline (text and voice lanes) with the real thresholds.
- `?q=...` pre-fills the box and runs it (and skips the intro); `?intro=0` skips the intro; `?step=1` opens a step;
  `?details=1` opens the details drawer; `?voice=1` turns voice on.
- Keyboard: `Enter` runs (nothing runs while you type), `Shift+Enter` new line, `Alt+1`…`Alt+6` examples,
  `D` details drawer, `V` voice on/off, `Esc` clear, `/` focus. Tab reaches every button.
- After a ❓ question, the next thing you type (or the option you click) is the answer: "turn on the light" → "bedroom"
  completes `set_light(room="bedroom", on=True)`. If the answer does not fill the asked field and clearly reads as a new
  command ("order 2 biryani"), it is treated as a new command.
- After a 🤷, clicking one of the two tools re-parses the same sentence with that tool.
- "Reset the house" puts the state back (all lights off, doors locked).

## Voice (optional toggle)

Voice is **off by default** and the app works exactly the same without it. `V` (or the 🎙 button in the header) turns
it on; if the voice server is not running the button says so and typing carries on. With voice on:

- `Enter` on an empty box (or `Alt+V`, or the 🎙 next to Run) starts listening. It stops by itself after a ~0.9 s
  pause (max 10 s), or press `Enter` again. `L` switches Whisper between English and Malayalam.
- Each clip goes two ways, both on the laptop CPU (`voice.py`, port 8092, started by `run.sh`):
  1. **Meaning straight from the sound.** EmbeddingGemma 2's audio encoder embeds the waveform into the same
     768-D space as text and compares it with the same 8 tool prototypes (built from the same descriptions and
     examples). No words involved, so it picks the *action* only. Own thresholds: top ≥ 0.70 and ≥ 0.05 ahead.
  2. **Words for the values.** Whisper `base` (int8, with a vocabulary hint: Amma, Achan, Carnatic, biryani, sadya…)
     writes a transcript, which goes through the normal text pipeline, so values are extracted and validated as usual.
- The text path decides the call. If the words alone are 🤷 but the sound is confident, the sound's action is used.
  The card shows both: "Heard “…”" and "From the sound alone: AC 0.76 · agrees ✓".
- Replies are spoken with the browser's local voice. After a ❓ question it listens again for the answer, so
  "turn on the light" → "Which room?" → "the balcony" completes `set_light(room="balcony", on=True)` hands-free.

Setup (once, ~1 GB download; the model is already in the Hugging Face cache if the notebook ran here):

```bash
python3 -m venv demo/.venv-audio
demo/.venv-audio/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
demo/.venv-audio/bin/pip install sentence-transformers transformers faster-whisper soundfile librosa pillow numpy
apps/typed-assistant/run.sh        # starts app.py (8091) and voice.py (8092); NO_VOICE=1 skips voice
```

Measured on this laptop (Ryzen 7 5800H, WSL2, CPU only), 10 test clips spoken by the Windows voice
(`data/voice-test/`, regenerate with `say.ps1`): all 8 commands give the right typed call, both off-topic
clips give 🤷, the sound alone picks the right action for all 8 (0.73–0.82, 0.09–0.20 ahead of #2; off-topic 0.64–0.66).

| Step | Time per 2–4 s command |
|---|---|
| Audio encoder (EmbeddingGemma 2, PyTorch float32, 6 threads) | 0.47–0.8 s |
| Whisper `base` int8 | 0.55–0.69 s |
| Text pipeline | ~0.1 s |
| **Total** | **1.0–1.5 s** |

Thread count matters: the audio encoder takes 2.1 s on 16 threads, 1.3 s on 8 and ~0.56 s on 6 (`VOICE_THREADS`).
bfloat16 is 4× slower on this CPU (no native bf16), so it stays float32. The voice server uses ~1.7 GB RAM;
running a second copy of the model at the same time pushes an 8 GB laptop into swap and makes both crawl.

Voice limits: tested with a synthetic English voice, not a room full of real accents; Whisper `base` Malayalam
is weak (the sound path still picks the action); the audio thresholds come from 10 clips; spoken replies use
whatever English voice the OS has.

## How it works

1. **Tool selection by meaning.** Each of the 8 tools has a description and 7–9 example utterances (some Manglish and
   Malayalam). Prototype = `normalize(mean(example vectors) + 0.5 × description vector)`, with the symmetric prefix
   `task: classification | text: ` on both sides. The message is embedded once (~80–90 ms) and compared by dot product.
   **Confident only if top score ≥ 0.82 and ≥ 0.03 ahead of #2**, otherwise 🤷 with the top 2.
2. **Typed schemas** (`TOOLS` in `app.py`): enums, bounded ints, bools, `"HH:MM"` times, free text; required or optional.
   ```
   set_light(room: "living room"|"bedroom"|"kitchen"|"balcony", on: bool, brightness: int[0..100] = None)
   set_ac(room: …, temperature: int[16..30], mode: "cool"|"dry"|"fan"|"auto" = None)
   set_timer(minutes: int[1..180], label: str = None)
   play_music(genre: "melody"|"carnatic"|"film songs"|"lo-fi"|"devotional", volume: int[0..100] = None)
   send_message(contact: "Amma"|"Achan"|"Rahul"|"Office", text: str)
   set_reminder(text: str, time: "HH:MM")
   lock_door(door: "front"|"back", locked: bool)
   order_food(dish: "biryani"|"appam & stew"|"puttu & kadala"|"porotta & beef"|"sadya", quantity: int[1..10])
   ```
3. **Argument extraction (no LLM), in this order:**
   - free text first, from the original words: message body after `saying` / `that` / `:` / "tell Rahul …" /
     Malayalam `അയക്കൂ …` or `… എന്ന്`; reminder text after "remind me to …" (time words stripped);
   - number words to digits ("twenty four", Manglish "pathu", Malayalam "രണ്ട്");
   - time of day: `7:30 pm`, `6 am`, `18:00`, `7 മണിക്ക്`, day-part words (രാവിലെ, evening, rathri). A bare
     "at 7" is **not guessed**: it becomes "morning or evening? 07:00 / 19:00";
   - durations: "10 minutes", "half an hour", "an hour and a half", "അഞ്ച് മിനിറ്റ്", summed;
   - enums: synonym lexicon (Latin words whole-word or prefix, Malayalam as substrings because suffixes attach:
     അമ്മയ്ക്ക്, അടുക്കളയിലെ). If no synonym matches, an **embedding fallback**: every enum value has a short
     description embedded once at startup; the message vector (already computed) is scored against them *relative to
     the field's average*, and the best value is accepted only if it leads the runner-up by ≥ 0.05
     ("order the onam feast for 4" → `sadya`, "play something relaxing to study" → `lo-fi`; "turn on the light" stays
     "Which room?");
   - ints by cue ("40%", "volume 30", "22 degrees", "2 plates", "a sadya" → 1) or a lone remaining number;
   - booleans by on/off and lock/unlock words (off, kill, ഓഫ്, കെടുത്ത / lock, unlock, open, പൂട്ട, തുറ).
     `set_light` with only a brightness implies `on=True`.
4. **Validation.** Every field is checked against type, bounds, enum and required-ness. Values the user actually said
   but got wrong are asked about before missing ones. Only a call with zero issues executes.
5. **Visible effect.** Lights per room, AC per room, door locks, a live timer countdown, now playing, last message,
   reminders, last order, and the last executed calls. State is in memory on the server.

The UI shows the chosen tool and score, the call as code and JSON, how each field was filled (synonym / embedding with
its lead / regex / free text), the top-3 tool scores, and latency (embed, extract + validate, total).

## Measured numbers (this laptop, WSL2, CPU only)

`run.sh --eval` on `tests.json` (67 hand-written utterances). The main set (53) was written together with the tools.
The hard set (14) are unseen phrasings found while probing the running app; two Manglish/Malayalam
`send_message` examples were added to the tool table after seeing them, nothing else was tuned on them.

| | Main set (53) | Hard set (14) | All (67) |
|---|---|---|---|
| Tool selection, top-1 | 49/49 = 100% | 9/11 = 82% | 58/60 = 97% |
| Tool selection, confident and right | 49/49 = 100% | 7/11 = 64% | 56/60 = 93% |
| Exact typed call (tool + every argument) | 33/33 = 100% | 5/8 = 62% | 38/41 = 93% |
| Clarifying question on the right field | 16/16 = 100% | 2/3 = 67% | 18/19 = 95% |
| "Unsure" when expected (off-topic) | 4/4 | 2/3 | 6/7 |
| Wrong call executed | 0 | 1 | 1 |

Every hard-set miss except one is a *safe* miss: it became 🤷 "which action?" instead of a call.
The one wrong executed call is "is the back door locked?" → `lock_door(door="back", locked=True)`: the schema has no
read-only "query state" tool, so a question about a door is read as a command.

- Embedding latency: median 88 ms, p90 91 ms in `--eval`; 77–110 ms in the live server.
  Extraction + validation: about 0.1 ms. Tool scoring (8 × 768 dot products): under 0.1 ms.
- Model load 0.2–0.4 s. First index build 8.5 s (90 ms per text), then cached.
- Server memory: ~330 MB RSS.

Live server checks (each a fresh sentence):

| Message | Result | Embed |
|---|---|---|
| set the bedroom AC to 22 in cool mode | `set_ac(room="bedroom", temperature=22, mode="cool")` (0.946) | 77 ms |
| dim the living room lights to 30% | `set_light(room="living room", on=True, brightness=30)` | 80 ms |
| remind me to call Achan at 7:30 pm | `set_reminder(text="call Achan", time="19:30")` | 84 ms |
| bedroom light off aakku | `set_light(room="bedroom", on=False)` | 84 ms |
| അമ്മയ്ക്ക് മെസ്സേജ് അയക്കൂ ഞാൻ വൈകും | `send_message(contact="Amma", text="ഞാൻ വൈകും")` (0.910) | 87 ms |
| രണ്ട് ബിരിയാണി ഓർഡർ ചെയ്യൂ | `order_food(dish="biryani", quantity=2)` | 93 ms |
| ബാൽക്കണിയിലെ ലൈറ്റ് ഓൺ ചെയ്യൂ | `set_light(room="balcony", on=True)` | 92 ms |
| set AC to 40 | ❓ Temperature must be between 16 and 30; you said 40. (then: which room?) | 106 ms |
| turn on the light | ❓ Which room? living room / bedroom / kitchen / balcony | 108 ms |
| play devotional songs at 120 volume | ❓ Volume must be between 0 and 100; you said 120. | 94 ms |
| what's the weather in Kochi | 🤷 top score 0.745 < 0.82: set_ac or lock_door? | 88 ms |
| it's getting stuffy in the hall, AC 23 please | `set_ac(room="living room", temperature=23)` | 87 ms |
| timer 25 min for the dosa batter | `set_timer(minutes=25, label="dosa batter")` | 87 ms |
| let Office know I'm working from home today | `send_message(contact="Office", text="I'm working from home today")` | 94 ms |
| 2 puttu kadala venam | `order_food(dish="puttu & kadala", quantity=2)` | 94 ms |
| order the onam feast for 4 | `order_food(dish="sadya", quantity=4)`, dish by embedding (lead 0.063) | 77 ms |
| put on some MS Subbulakshmi | ❓ Which kind of music? (embedding not sure enough, so it asks) | 89 ms |

Follow-ups: "set AC to 40" → "22" → "living room" gives `set_ac(room="living room", temperature=22)`;
"remind me at 7 to water the plants" → pick `19:00` → `set_reminder(text="water the plants", time="19:00")`;
"send a message to Achan" → "call me when you're free" fills `text`.

## Talking points

- **Types are the safety net, not the model.** The embedding model only ranks; the schema decides what may run. A
  wrong-but-valid-looking value (AC 40 °C, 15 plates, a 5-hour timer) is caught by the validator and turned into a question.
- **Ask, don't guess.** "at 7" is ambiguous, so it asks 07:00 or 19:00. "turn on the light" asks which room. The only
  way to get a call is to have every required field, in range.
- **Meaning where words fail.** "order the onam feast" has no synonym for any dish, but the embedding picks `sadya`
  because it leads the other dishes by a clear margin. When the lead is small ("put on some MS Subbulakshmi") it asks.
- **Cross-lingual for free.** Malayalam script commands work with a mostly English tool table, because the model maps
  them near the English examples. The lexicon only has to cover the enum words.
- **Explainable.** Every field shows how it was filled. Extraction + validation take about 0.1 ms; the one embedding
  (~85 ms) is the whole cost.
- **Adding a tool is data, not training**: a description, a few examples, a typed parameter list, maybe some synonyms.

## Honest limits

- It is a demo with 8 tools and a hand-written lexicon. Phrasings outside the lexicon and regexes are missed, and the
  hand-written test set is small; the main-set 100% says the rules cover what they were written for, the hard set
  (64% confident and right) is closer to what an unprepared audience will type.
- Romanised Malayalam (Manglish) sentences without English keywords are weak: "Achanod parayu njan ethi" and
  "7 manikku marunnu kazhikkan ormippikku" fall to 🤷 (safe, but not helpful). Malayalam script does better.
- One command per message; no dates ("tomorrow" is kept in the reminder text, not scheduled); no "turn off the AC"
  (the schema has no AC power field, so it asks for room and temperature); no state questions ("is the door locked?" is
  read as a command, the one wrong call in the eval).
- Thresholds (0.82 / 0.03 for tools, 0.05 for enum fallback) were chosen on a handful of probes, not a benchmark.
  Off-topic text that scores above 0.82 ("hello" sits at 0.80, "open YouTube" at 0.82) can reach a clarifying question,
  which is harmless but odd.
- The model runs one request at a time behind a lock; the home state is a single in-memory object shared by every
  browser tab.
- In this WSL setup, headless Chrome needs the Windows fonts for Malayalam; Windows browsers use Nirmala UI.
