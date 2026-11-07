# On-device intent matcher

Type a customer-support message; EmbeddingGemma 2 (270M, quantized LiteRT-LM file, 165 MB) embeds
it on the laptop CPU and picks the closest of 77 real Banking77 intents. Results update as you type.
No network once the data is cached; nothing leaves the laptop.

## Run

```bash
apps/intent-matcher/run.sh            # opens on http://127.0.0.1:8090
apps/intent-matcher/run.sh --eval     # accuracy + latency (add --fresh to re-embed the test set)
```

`run.sh` uses `demo/.venv` (litert-lm 0.18.0 + numpy) and `demo/models/embeddinggemma-2-text-270m.litertlm`
(run `demo/laptop_demo.sh` once if the model is missing). First run only: downloads Banking77
(`mteb/banking77` JSONL, 1.6 MB) into `data/` and embeds the intent index (847 texts, ~76 s). After that,
startup takes about 1 s. `data/` is gitignored. `?q=...` in the URL pre-fills the box.

Keyboard: type in the box; `Alt+1`…`Alt+8` load the examples; `Esc` clears; `/` focuses the box; Tab
reaches every example button.

## How it works

- Both sides use the symmetric prefix `task: classification | text: `. Vectors are normalized; score = dot product.
- Each intent gets a prototype: `normalize(mean(10 example vectors) + 0.5 × name vector)`, where the name is
  the readable label (`card_arrival` → "card arrival") and the examples are 10 random training messages (seed 0).
- Each message is embedded once and compared with 77 prototypes (a 77×768 matrix multiply, well under 1 ms).
- Confidence rule: **confident only if top score ≥ 0.82 and top − second ≥ 0.03**; otherwise the UI says
  "Unsure, ask a clarifying question: did you mean A or B?".

## Measured numbers (this laptop, WSL2, CPU only)

`run.sh --eval`: 500 random Banking77 test messages (seed 0), 77 intents, chance = 1.3%.

| Prototype | Top-1 | Top-3 |
|---|---|---|
| Intent name only (zero examples) | 91.6% | 98.6% |
| Name + 1 example | 88.8% | 97.4% |
| Name + 3 examples | 91.2% | 99.2% |
| 10 examples, no name | 91.0% | 99.4% |
| **Name + 10 examples (app default)** | **92.6%** | **99.2%** |

- Query embedding latency: median 85 ms, p90 89 ms (in the live server: 72–77 ms).
- Model load 0.25 s. First index build 75.7 s (89 ms per text), then cached.
- Server memory: ~330 MB RSS.

Confidence rule on the same 500 messages:

| Rule | Marked confident | Accuracy when confident | Top-1 still right when unsure |
|---|---|---|---|
| No rule | 100% | 92.6% | – |
| Gap ≥ 0.02 | 92% | 95.2% | 64% |
| **Score ≥ 0.82 and gap ≥ 0.03 (app)** | **87%** | **97.5%** | 60% |

The gap does most of the work on in-scope messages (correct answers have median gap 0.108, wrong ones
0.024). The score floor of 0.82 catches off-topic text: in-scope messages almost never score below 0.85
(5th percentile 0.852), while off-topic probes scored 0.67–0.80.

Live server checks:

| Message | Result | Embed |
|---|---|---|
| my card still hasn't arrived | card arrival, 0.958 (gap 0.138) | 74 ms |
| bro where's my card, ordered it like 2 weeks ago | card arrival, 0.931 | 72 ms |
| atm ate my card lol | card swallowed, 0.924 | 72 ms |
| got charged 2x for the same coffee?? | transaction charged twice, 0.875 | 75 ms |
| എന്റെ കാർഡ് ഇതുവരെ വന്നില്ല (my card hasn't arrived yet) | card arrival, 0.871 (gap 0.037) | 74 ms |
| ഒരേ സാധനത്തിന് രണ്ടുതവണ പണം പിടിച്ചു (charged twice for the same item) | transaction charged twice, 0.834 | 73 ms |
| പണം രണ്ടുതവണ പിടിച്ചു (money taken twice) | unsure: 0.799; top guess still "transaction charged twice" | 75 ms |
| card | unsure: Visa or Mastercard 0.854 vs card linking 0.844 | 98 ms (Chrome rendering at the same time) |
| what's the weather in Kochi | unsure: 0.670 | 73 ms |
| how do I reset my wifi router | unsure: 0.801 (would-be "passcode forgotten") | 77 ms |

## Talking points

- **No training run.** A new intent is one line of text, or a few example messages. Re-indexing 77 intents takes about a minute on a laptop CPU.
- **The name alone gets you most of the way.** The intent name by itself scores 91.6% top-1 on 77 classes. Ten real examples raise that to 92.6% and make the top-3 almost perfect (99.2%), so a "did you mean A or B?" fallback nearly always offers the right answer.
- **Fewer examples isn't always better than none.** One example scored *worse* than the name alone (88.8%), because a single message is a noisy picture of an intent. Average several examples.
- **Know when you don't know.** Scores are compressed (unrelated text still scores about 0.7), so use the gap to #2 together with a floor. With this rule, 87% of messages are answered at 97.5% accuracy and the rest get a clarifying question.
- **Cross-lingual for free.** The prototypes are English only, yet Malayalam messages land on the right intent. Short or vague Malayalam ("പണം രണ്ടുതവണ പിടിച്ചു") gets top-1 right but falls under the floor, which is the safe behaviour.
- **Private and offline.** About 80 ms per message on a laptop CPU, with a 165 MB model and about 330 MB of RAM. Customer text never leaves the device.

## Caveats

- The model runs one request at a time behind a lock. The UI debounces at 150 ms and aborts stale fetches, but a request that has already reached the model still finishes.
- Banking77 has no out-of-scope class, so the 0.82 floor comes from a handful of hand-written off-topic probes, not a benchmark.
- In this WSL setup, headless Chrome has no Malayalam font. Windows browsers use Nirmala UI.
