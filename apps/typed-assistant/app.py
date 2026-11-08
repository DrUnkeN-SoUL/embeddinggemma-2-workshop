"""Type-safe on-device assistant: EmbeddingGemma 2 (LiteRT-LM, laptop CPU, offline).

    python app.py                 # serve the demo UI on http://127.0.0.1:8091
    python app.py --eval          # tool selection / exact args / clarifying questions on tests.json
    python app.py --build         # embed tool examples + enum descriptions into data/ and exit

Natural language -> exactly one of:
  * a schema-valid typed function call (then it runs against an in-memory "home"),
  * a clarifying question naming the missing or invalid field,
  * "unsure which action" with the top 2 tools.

There is no LLM. EmbeddingGemma 2 picks the tool by meaning (one embedding per message, compared
with one prototype vector per tool) and, when no synonym matches, the value of an enum field.
Everything else (numbers, durations, times, on/off, free text) is regex + a small lexicon,
and a validator checks every argument against its type, bounds and enum before anything runs.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE_FILE = os.path.join(DATA, "vectors.npz")
TESTS = os.path.join(HERE, "tests.json")
MODEL = os.environ.get(
    "ASSIST_MODEL",
    os.path.join(HERE, "..", "..", "demo", "models", "embeddinggemma-2-text-270m.litertlm"),
)

# Symmetric task prefix: tool examples, enum descriptions and the user message all use it.
PREFIX = "task: classification | text: "

DESC_WEIGHT = 0.5   # tool prototype = normalize(mean(example vecs) + DESC_WEIGHT * description vec)

# Tool choice is confident only if top score >= MIN_SCORE and (top - second) >= MIN_GAP
# (chosen from `python app.py --eval`, see README).
MIN_SCORE = 0.82
MIN_GAP = 0.03

# Enum fallback: score each value against the message *relative to the field's average value*
# (contrastive), accept the best only if it leads the runner-up by ENUM_GAP.
ENUM_GAP = 0.05


def log(*a):
    print(*a, flush=True)


# ================================================================ schema

ROOMS = ["living room", "bedroom", "kitchen", "balcony"]


def enum(name, values, ask, **kw):
    return dict(name=name, type="enum", values=values, ask=ask, **kw)


def integer(name, lo, hi, ask, label, **kw):
    return dict(name=name, type="int", min=lo, max=hi, ask=ask, label=label, **kw)


TOOLS = [
    dict(
        name="set_light",
        description="Turn the lights in a room on or off, or change their brightness",
        params=[
            enum("room", ROOMS, "Which room?"),
            dict(name="on", type="bool", ask="On or off?", words=("on", "off")),
            integer("brightness", 0, 100, "What brightness?", "Brightness", required=False),
        ],
        examples=[
            "turn on the living room lights",
            "switch off the bedroom light",
            "dim the kitchen lights to 30 percent",
            "lights off in the balcony",
            "make the hall lights brighter",
            "it's too dark in here",
            "kitchen light on aakku",
            "ലൈറ്റ് ഓഫ് ചെയ്യൂ",
        ],
    ),
    dict(
        name="set_ac",
        description="Set the air conditioner temperature and mode in a room",
        params=[
            enum("room", ROOMS, "Which room?"),
            integer("temperature", 16, 30, "What temperature?", "Temperature", unit="°C"),
            enum("mode", ["cool", "dry", "fan", "auto"], "Which mode?", required=False),
        ],
        examples=[
            "set the bedroom AC to 22",
            "make the AC 24 degrees in cool mode",
            "it's too hot, turn the AC down to 20",
            "AC dry mode in the living room",
            "increase the air conditioner temperature",
            "AC 23 aakku",
            "ഏസി 24 ഡിഗ്രി ആക്കൂ",
            "bhayankara choodu, AC idu",
        ],
    ),
    dict(
        name="set_timer",
        description="Start a countdown timer for some minutes",
        params=[
            integer("minutes", 1, 180, "For how long?", "Timer length", unit="min"),
            dict(name="label", type="str", ask="What is the timer for?", required=False),
        ],
        examples=[
            "set a timer for 10 minutes",
            "timer for half an hour",
            "start a 5 minute timer for the rice",
            "countdown 45 minutes",
            "let me know when 15 minutes are up",
            "wake me up in 20 minutes",
            "10 minute timer vekku",
            "പത്ത് മിനിറ്റ് ടൈമർ വെക്കൂ",
        ],
    ),
    dict(
        name="play_music",
        description="Play music of a genre, optionally at a volume",
        params=[
            enum("genre", ["melody", "carnatic", "film songs", "lo-fi", "devotional"], "Which kind of music?"),
            integer("volume", 0, 100, "What volume?", "Volume", required=False),
        ],
        examples=[
            "play some melody songs",
            "put on carnatic music",
            "play lo-fi beats at volume 30",
            "play devotional songs",
            "I want to hear some film songs",
            "play something relaxing",
            "oru paattu idu",
            "പാട്ട് വെക്കൂ",
        ],
    ),
    dict(
        name="send_message",
        description="Send a text message to a contact",
        params=[
            enum("contact", ["Amma", "Achan", "Rahul", "Office"], "Who should I send it to?"),
            dict(name="text", type="str", ask="What should the message say?"),
        ],
        examples=[
            "message Amma saying I'll be late",
            "text Rahul that the meeting moved to 5",
            "tell Achan I reached safely",
            "send a WhatsApp to the office: running late",
            "let Rahul know I'm on my way",
            "Ammakku oru message ayakku",
            "Rahulinodu parayu njan varunnu",
            "അമ്മയ്ക്ക് മെസ്സേജ് അയക്കൂ",
            "അച്ഛനോട് പറയൂ ഞാൻ ട്രെയിനിൽ കയറി",
        ],
    ),
    dict(
        name="set_reminder",
        description="Remind me about something at a time of day",
        params=[
            dict(name="text", type="str", ask="What should I remind you about?"),
            dict(name="time", type="time", ask="At what time?"),
        ],
        examples=[
            "remind me to call Achan at 7 pm",
            "set a reminder for 9:30 to take my medicine",
            "remind me at 6 am to go for a walk",
            "don't let me forget the electricity bill at 18:00",
            "reminder: pay the rent at 10 am",
            "7 manikku marunnu kazhikkan ormippikku",
            "രാവിലെ 8 മണിക്ക് ഓർമ്മിപ്പിക്കൂ",
        ],
    ),
    dict(
        name="lock_door",
        description="Lock or unlock a door of the house",
        params=[
            enum("door", ["front", "back"], "Which door?"),
            dict(name="locked", type="bool", ask="Lock or unlock?", words=("lock", "unlock")),
        ],
        examples=[
            "lock the front door",
            "unlock the back door",
            "open the back door for the delivery",
            "lock up the house for the night",
            "let the guests in, unlock the front",
            "front door lock cheyyu",
            "മുൻവാതിൽ പൂട്ടൂ",
        ],
    ),
    dict(
        name="order_food",
        description="Order food for delivery",
        params=[
            enum("dish", ["biryani", "appam & stew", "puttu & kadala", "porotta & beef", "sadya"], "Which dish?"),
            integer("quantity", 1, 10, "How many?", "Quantity", unit="plates"),
        ],
        examples=[
            "order two biryanis",
            "get me appam and stew",
            "I'm hungry, order puttu and kadala",
            "order 3 porotta and beef",
            "book a sadya for 5 people",
            "oru biriyani order cheyyu",
            "രണ്ട് ബിരിയാണി ഓർഡർ ചെയ്യൂ",
            "food order cheyyanam",
        ],
    ),
]
TOOL_BY_NAME = {t["name"]: t for t in TOOLS}

# Synonyms per enum value. Latin entries match whole words; a trailing * means "word prefix"
# (ammakku, biryanis). Malayalam entries match as substrings (the language glues suffixes on:
# അമ്മയ്ക്ക്, അടുക്കളയിലെ).
LEXICON = {
    "room": {
        "living room": ["living room", "living", "hall", "sitting room", "drawing room", "lounge", "ഹാൾ", "സ്വീകരണമുറി"],
        "bedroom": ["bedroom*", "bed room", "bed-room", "kidappumuri*", "ബെഡ്റൂ", "ബെഡ്‌റൂ", "കിടപ്പുമുറി"],
        "kitchen": ["kitchen*", "adukkala*", "അടുക്കള", "കിച്ച"],
        "balcony": ["balcony", "balconi", "sit-out", "sit out", "sitout", "verandah", "veranda", "porch", "terrace",
                    "ബാൽക്കണി", "വരാന്ത", "സിറ്റൗട്ട്"],
    },
    "mode": {
        "cool": ["cool", "cooling", "cold"],
        "dry": ["dry", "dehumidify*"],
        "fan": ["fan", "fan only"],
        "auto": ["auto", "automatic"],
    },
    "genre": {
        "melody": ["melody", "melodies", "melodious", "romantic", "soft songs"],
        "carnatic": ["carnatic", "karnatic", "classical", "kriti*", "keerthana*", "raga*", "കർണാടക", "ശാസ്ത്രീയ"],
        "film songs": ["film", "films", "filmy", "movie", "movies", "cinema", "cinematic", "sinima", "സിനിമ"],
        "lo-fi": ["lo-fi", "lofi", "lo fi", "chill", "chillhop", "study beats"],
        "devotional": ["devotional", "bhajan*", "bhakti", "bhakthi", "temple", "prayer", "ഭക്തി", "ഭജന"],
    },
    "contact": {
        "Amma": ["amma*", "mom", "mum", "mummy", "mother", "അമ്മ"],
        "Achan": ["achan*", "acha", "achaa", "dad", "daddy", "papa", "father", "അച്ഛ", "അച്ചൻ"],
        "Rahul": ["rahul*", "രാഹുൽ", "രാഹുലി"],
        "Office": ["office", "ofice", "boss", "manager", "work", "ഓഫീസ", "ഓഫിസ"],
    },
    "door": {
        "front": ["front", "main door", "main entrance", "munvathil", "മുൻവാതിൽ", "മുൻവശ", "മുന്നിലെ"],
        "back": ["back", "rear", "kitchen door", "back door", "pinvathil", "പിൻവാതിൽ", "പിന്നിലെ", "പുറകിലെ"],
    },
    "dish": {
        "biryani": ["biryani*", "biriyani*", "briyani*", "ബിരിയാണി", "ബിരിയാണ"],
        "appam & stew": ["appam*", "stew", "അപ്പം", "സ്റ്റൂ"],
        "puttu & kadala": ["puttu", "kadala", "പുട്ട്", "കടല"],
        "porotta & beef": ["porotta*", "parotta*", "barotta*", "beef", "പൊറോട്ട", "ബീഫ്"],
        "sadya": ["sadya", "sadhya", "onam meal", "സദ്യ"],
    },
}

# One short description per enum value, embedded once, for the meaning-based fallback.
ENUM_DESC = {
    "room": {"living room": "the living room, the hall where we sit",
             "bedroom": "the bedroom where I sleep",
             "kitchen": "the kitchen where I cook",
             "balcony": "the balcony, the sit-out outside"},
    "mode": {"cool": "AC cooling mode, make it cold",
             "dry": "AC dry mode, remove humidity",
             "fan": "AC fan only mode, just circulate air",
             "auto": "AC auto mode, decide by itself"},
    "genre": {"melody": "soft romantic melody songs",
              "carnatic": "Carnatic classical music, kritis and ragas",
              "film songs": "Malayalam and Tamil film songs from movies",
              "lo-fi": "lo-fi chill beats for studying and relaxing",
              "devotional": "devotional songs, bhajans and temple music"},
    "contact": {"Amma": "Amma, my mother", "Achan": "Achan, my father",
                "Rahul": "Rahul, my friend", "Office": "the office, my boss at work"},
    "door": {"front": "the front door, the main entrance", "back": "the back door behind the kitchen"},
    "dish": {"biryani": "biryani, spiced rice with chicken or mutton",
             "appam & stew": "appam with stew, a Kerala breakfast",
             "puttu & kadala": "puttu with kadala curry",
             "porotta & beef": "porotta with beef fry",
             "sadya": "sadya, the Onam vegetarian feast on a banana leaf"},
}

# Boolean cue words. Checked in order; the first group with a hit wins.
BOOL_WORDS = {
    ("on", "off"): [
        (False, [r"\boff\b", r"\bkill\b", r"\bkedu?thu?\w*", r"ഓഫ്", r"കെടുത്ത", r"അണയ്ക്ക", r"അണക്ക"]),
        (True, [r"\bon\b", r"\bbright\w*", r"ഓൺ", r"\bittu\b", r"\bidu\b", r"തെളിയ"]),
    ],
    ("lock", "unlock"): [
        (False, [r"\bunlock\w*", r"\bopen\w*", r"\blet\b.*\bin\b", r"\bthura\w*", r"തുറ", r"അൺലോക്ക്"]),
        (True, [r"\block\w*", r"\bsecure\b", r"\bclose\b", r"\bshut\b", r"\bpoottu?\w*", r"പൂട്ട", r"ലോക്ക്", r"അടയ്ക്ക", r"അടക്ക"]),
    ],
}

# ================================================================ text helpers

ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
MANGLISH = {"onnu": 1, "randu": 2, "rendu": 2, "moonu": 3, "moonnu": 3, "naalu": 4, "nalu": 4, "anju": 5,
            "aaru": 6, "ezhu": 7, "ettu": 8, "ombathu": 9, "pathu": 10, "irupathu": 20, "muppathu": 30}
MALAYALAM_NUM = [("ഒന്ന", 1), ("രണ്ട", 2), ("മൂന്ന", 3), ("നാല", 4), ("അഞ്ച", 5), ("ആറ", 6), ("ഏഴ", 7),
                 ("എട്ട", 8), ("ഒമ്പത", 9), ("ഒൻപത", 9), ("പത്ത", 10), ("ഇരുപത", 20), ("മുപ്പത", 30)]
NUM_WORD_RE = re.compile(
    r"\b(?:(%s)(?:[\s-]+(%s))?|(%s)|(%s))\b" % (
        "|".join(TENS), "|".join(w for w in ONES if 0 < ONES[w] < 10), "|".join(ONES), "|".join(MANGLISH)),
    re.I)
ML_NUM_RE = re.compile(r"(?<!\S)(%s)\S*" % "|".join(w for w, _ in MALAYALAM_NUM))


def numbers_to_digits(s):
    """'twenty two' -> '22', 'pathu' -> '10', 'രണ്ട്' -> '2'. Case of other text is kept."""
    def en(m):
        if m.group(1):
            return str(TENS[m.group(1).lower()] + (ONES[m.group(2).lower()] if m.group(2) else 0))
        if m.group(3):
            return str(ONES[m.group(3).lower()])
        return str(MANGLISH[m.group(4).lower()])
    s = NUM_WORD_RE.sub(en, s)
    return ML_NUM_RE.sub(lambda m: str(dict(MALAYALAM_NUM)[m.group(1)]), s)


def is_latin(word):
    return all(ord(c) < 0x250 for c in word)


def syn_regex(syn):
    if not is_latin(syn):
        return re.escape(syn)
    prefix = syn.endswith("*")
    body = re.escape(syn.rstrip("*")).replace(r"\ ", r"[\s-]*").replace(r"\-", r"[\s-]*")
    return r"\b" + body + (r"\w*" if prefix else r"\b")


SYN_RE = {field: {val: re.compile("|".join(syn_regex(s) for s in syns), re.I) for val, syns in vals.items()}
          for field, vals in LEXICON.items()}
ALL_CONTACTS = "|".join(syn_regex(s) for syns in LEXICON["contact"].values() for s in syns)


def blank(s, span):
    """Replace s[span] with spaces so offsets stay valid and the text is not re-used."""
    a, b = span
    return s[:a] + " " * (b - a) + s[b:]


def tidy(t):
    t = re.sub(r"\s+", " ", t or "").strip(" \t,.;:!?-\"'“”‘’")
    return t or None


# ================================================================ time / duration / numbers

AM = r"(a\.?\s?m\.?|morning|raavile|ravile|രാവിലെ)"
PM = r"(p\.?\s?m\.?|evening|night|tonight|afternoon|vaikunneram|vaikittu|rathri|raathri|uchakku|വൈകുന്നേരം|വൈകിട്ട്|രാത്രി|ഉച്ച)"
TIME_PATTERNS = [
    re.compile(r"\b(\d{1,2})[:.](\d{2})\s*(?:%s|%s)?" % (AM, PM), re.I),
    re.compile(r"\b(\d{1,2})()\s*(?:%s|%s)" % (r"(a\.?\s?m\b\.?)", r"(p\.?\s?m\b\.?)"), re.I),
    re.compile(r"\b(\d{1,2})()\s*(?:o'?\s?clock|mani\w*|മണിക്ക്|മണി)(?!ക്കൂ)", re.I),
    re.compile(r"(?:\bat|@)\s*(\d{1,2})()\b(?!\s*(?:min|hour|hr|%|percent|degree|°|plate|people))", re.I),
]
NOON = re.compile(r"\b(noon|midday|midnight)\b", re.I)


def parse_time(s):
    """Returns (value, span, problem). value is 'HH:MM'; problem is a dict for an invalid/ambiguous time."""
    m = NOON.search(s)
    if m:
        return ("00:00" if m.group(1).lower() == "midnight" else "12:00"), m.span(), None
    for i, rx in enumerate(TIME_PATTERNS):
        m = rx.search(s)
        if not m:
            continue
        h, mm = int(m.group(1)), int(m.group(2) or 0)
        span = m.span()
        if i == 1:
            ampm = "am" if m.group(3) else "pm"
        elif i == 0:
            ampm = "am" if m.group(3) else ("pm" if m.group(4) else None)
        else:
            ampm = None
        if ampm is None:  # look for a day-part word anywhere: "രാവിലെ 8 മണിക്ക്", "7 at night"
            if re.search(PM, s, re.I):
                ampm = "pm"
            elif re.search(AM, s, re.I):
                ampm = "am"
        if mm > 59 or h > 23 or (ampm and not 1 <= h <= 12):
            return None, span, dict(kind="range", said=m.group(0).strip())
        if ampm == "pm" and h != 12:
            h += 12
        elif ampm == "am" and h == 12:
            h = 0
        elif ampm is None and 1 <= h <= 12:
            # "at 7" or "7:30" with no am/pm: ask instead of guessing.
            opts = [f"{h % 12:02d}:{mm:02d}", f"{h % 12 + 12:02d}:{mm:02d}"]
            if h == 12:
                opts = ["12:%02d" % mm, "00:%02d" % mm]
            return None, span, dict(kind="ambiguous", said=m.group(0).strip(), options=opts)
        return f"{h:02d}:{mm:02d}", span, None
    return None, None, None


DURATION_PATTERNS = [
    (re.compile(r"\b(?:an?|1)\s+hours?\s+and\s+a\s+half\b|\b1\.5\s*(?:hours?|hrs?)\b|\bone and a half hours?\b", re.I), lambda m: 90),
    (re.compile(r"\bhalf\s+(?:an?\s+)?hour\b|അര\s*മണിക്കൂ\S*|\bara\s+manikk?oo?r\w*", re.I), lambda m: 30),
    (re.compile(r"\bquarter\s+(?:of\s+)?(?:an?\s+)?hour\b", re.I), lambda m: 15),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b|(\d+(?:\.\d+)?)\s*(?:മണിക്കൂ\S*|manikk?oo?r\w*)", re.I),
     lambda m: round(float(m.group(1) or m.group(2)) * 60)),
    (re.compile(r"\ban?\s+hour\b", re.I), lambda m: 60),
    (re.compile(r"\b(\d+)\s*(?:minutes?|mins?|m)\b|(\d+)\s*(?:മിനിറ്റ\S*|മിനുട്ട\S*|minute\w*)", re.I),
     lambda m: int(m.group(1) or m.group(2))),
    (re.compile(r"\ba\s+minute\b", re.I), lambda m: 1),
]


def parse_duration(s):
    """Sum every duration phrase: '1 hour 30 minutes' -> 90. Returns (minutes|None, spans)."""
    total, spans = 0, []
    for rx, val in DURATION_PATTERNS:
        for m in rx.finditer(s):
            if any(a < m.end() and m.start() < b for a, b in spans):
                continue
            total += val(m)
            spans.append(m.span())
    return (total if spans else None), spans


DISH_ANY = "|".join(syn_regex(s) for syns in LEXICON["dish"].values() for s in syns)
INT_CUES = {
    "temperature": [r"(\d+)\s*(?:°|degrees?|deg\b|c\b|ഡിഗ്രി|digri)", r"\b(?:to|at)\s+(\d+)\b"],
    "brightness": [r"(\d+)\s*(?:%|percent|per cent)", r"\bbrightness\s+(?:to\s+|at\s+|of\s+)?(\d+)", r"\bto\s+(\d+)\b"],
    "volume": [r"\bvolume\s+(?:to\s+|at\s+|of\s+)?(\d+)", r"(\d+)\s*(?:%|percent)", r"\b(?:at|to)\s+(\d+)\b"],
    "quantity": [r"(\d+)\s*(?:x\s*)?(?:plates?|portions?|packets?|servings?|people|persons?|nos?\b|pieces?)",
                 r"(\d+)\s+(?:plates?\s+of\s+)?(?=%s)" % DISH_ANY],
}
ONE_WORDS = re.compile(r"(?:\b(?:a|an|oru)|(?<!\S)ഒരു)\s+(?:plate\s+(?:of\s+)?)?(?=%s)" % DISH_ANY, re.I)


def parse_int(name, s):
    """Returns (int|None, span). Cue patterns first, then a lone number if exactly one is left."""
    for cue in INT_CUES.get(name, []):
        m = re.search(cue, s, re.I)
        if m:
            return int(m.group(1)), m.span(1)
    if name == "quantity":
        m = ONE_WORDS.search(s)
        if m:
            return 1, m.span()
    nums = list(re.finditer(r"(?<![\d.:])\d+(?![\d.:])", s))
    if len(nums) == 1:
        return int(nums[0].group()), nums[0].span()
    return None, None


# ================================================================ free text

def contact_tail(s):
    return re.search(r"(?:\b(?:tell|text|message|msg|whatsapp|ping|inform|let|send)\s+(?:a\s+\w+\s+to\s+|to\s+)?)"
                     r"(?:%s)\s+(?:know\s+)?(?:that\s+)?(.+)$" % ALL_CONTACTS, s, re.I)


def extract_message(s):
    for rx in (r"\b(?:saying|that says|says|stating)\s+(.+)$",
               r":\s*(.+)$",
               r"\bthat\s+(.+)$"):
        m = re.search(rx, s, re.I)
        if m:
            return m.group(1), m.span(1)
    m = contact_tail(s)
    if m:
        return m.group(1), m.span(1)
    # Malayalam / Manglish: "... അയക്കൂ ഞാൻ വൈകും", "ഞാൻ വൈകും എന്ന് അമ്മയോട് പറയൂ"
    m = re.search(r"(?:അയക്ക\S*|അയയ്ക്ക\S*|പറയൂ|പറയ്\S*|\bayakk\w*|\bparay\w*)\s*[:,]?\s+(.+)$", s, re.I)
    if m:
        return m.group(1), m.span(1)
    m = re.search(r"^(.+?)\s*(?:എന്ന്|\bennu\b|\benn\b)\s", s, re.I)
    if m:
        return m.group(1), m.span(1)
    return None, None


REMIND_KW = r"(?:\bremind\w*|\bforget\b|\bormippik\w*|ഓർമ്മിപ്പിക്ക|ഓർമിപ്പിക്ക|ഓർമ്മപ്പെടുത്ത)"


def strip_time_words(t):
    t = numbers_to_digits(t)
    _, span, _ = parse_time(t)
    if span:
        t = blank(t, span)
    t = re.sub(r"\b(?:at|by|@)\s*$", "", t.strip(), flags=re.I)
    t = re.sub(r"(?:\b(?:today|tonight|in the (?:morning|evening))\b|%s|%s)" % (
        r"(?<!\S)(?:രാവിലെ|വൈകുന്നേരം|രാത്രി)", r"\b(?:raavile|vaikunneram|rathri)\b"), " ", t, flags=re.I)
    t = re.sub(r"\s+\b(?:at|by)\s*$|^\s*(?:at|by)\b", "", t.strip(), flags=re.I)
    return tidy(t)


def extract_reminder(s):
    m = re.search(r"\breminder\s*[:\-]\s*(.+)$", s, re.I)
    if not m:
        kw = re.search(REMIND_KW, s, re.I)
        if kw:
            m = re.compile(r"\b(?:to|about|that)\s+(.+)$", re.I).search(s, kw.end())
            if not m and not is_latin(kw.group(0)) or (not m and kw.group(0).lower().startswith("ormippik")):
                m = re.search(r"^(.+?)\s*(?:എന്ന്\s*)?$", s[:kw.start()])
    if not m:
        return None, None
    return strip_time_words(m.group(1)), m.span(1)


def extract_label(s):
    m = re.search(r".*\b(?:for|called|named|labell?ed|label)\s+(?:the\s+)?(?!\d|an?\s+(?:hour|minute)|half|quarter)"
                  r"([^\d\s][^\d]*?)\s*$", s, re.I)
    if m:
        return tidy(m.group(1)), m.span(1)
    return None, None


# ================================================================ model

class Embedder:
    """One LiteRT-LM EmbeddingEngine, shared across threads behind a lock."""

    def __init__(self, path=MODEL):
        if not os.path.exists(path):
            sys.exit(f"Model not found: {path}\nRun demo/laptop_demo.sh once to download it.")
        from litert_lm import Backend, EmbeddingEngine
        t0 = time.perf_counter()
        self._ctx = EmbeddingEngine(path, backend=Backend.CPU())
        self.engine = self._ctx.__enter__()
        self.load_s = time.perf_counter() - t0
        self.lock = threading.Lock()

    def embed(self, text):
        with self.lock:
            vec = np.asarray(self.engine.compute_embedding(PREFIX + text).embedding, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def close(self):
        self._ctx.__exit__(None, None, None)


def normalize(m):
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)


def index_texts():
    texts = []
    for t in TOOLS:
        texts += [t["description"]] + t["examples"]
    for vals in ENUM_DESC.values():
        texts += list(vals.values())
    return texts


def load_vectors(embedder_factory):
    """text -> vector cache in data/vectors.npz. Only texts not seen before are embedded."""
    vecs, build_s, new = {}, 0.0, 0
    if os.path.exists(CACHE_FILE):
        z = np.load(CACHE_FILE, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("prefix") == PREFIX and meta.get("model") == os.path.basename(MODEL):
            vecs = dict(zip(meta["texts"], z["vecs"]))
            build_s = meta.get("build_s", 0.0)
    missing = [t for t in OrderedDict.fromkeys(index_texts()) if t not in vecs]
    emb = None
    if missing:
        emb = embedder_factory()
        log(f"Embedding {len(missing)} tool examples / enum descriptions ...")
        t0 = time.perf_counter()
        for t in missing:
            vecs[t] = emb.embed(t)
        dt = time.perf_counter() - t0
        build_s += dt
        new = len(missing)
        log(f"  {dt:.1f} s ({dt / len(missing) * 1000:.0f} ms per text). Cached to {CACHE_FILE}")
        os.makedirs(DATA, exist_ok=True)
        keep = list(OrderedDict.fromkeys(index_texts()))
        meta = dict(texts=keep, prefix=PREFIX, model=os.path.basename(MODEL), build_s=build_s)
        np.savez(CACHE_FILE, vecs=np.stack([vecs[t] for t in keep]), meta=np.asarray(json.dumps(meta)))
    return vecs, build_s, new, emb


# ================================================================ the assistant

class Assistant:
    def __init__(self, embedder=None):
        holder = {}

        def factory():
            holder["e"] = embedder or Embedder()
            return holder["e"]
        vecs, self.build_s, self.new_vecs, emb = load_vectors(factory)
        self.emb = emb or embedder or Embedder()
        self.names = [t["name"] for t in TOOLS]
        self.protos = np.stack([
            normalize(normalize(np.mean([vecs[e] for e in t["examples"]], axis=0)) + DESC_WEIGHT * vecs[t["description"]])
            for t in TOOLS])
        self.enum_vals, self.enum_mat = {}, {}
        for field, vals in ENUM_DESC.items():
            m = np.stack([vecs[d] for d in vals.values()])
            self.enum_vals[field] = list(vals)
            self.enum_mat[field] = m - m.mean(axis=0)  # contrastive: value minus field average
        self.cache = OrderedDict()
        self.emb.embed("warm up")
        self.reset()

    # ---------------------------------------------------------- embedding

    def embed(self, text):
        hit = text in self.cache
        if hit:
            self.cache.move_to_end(text)
            return self.cache[text], True
        v = self.emb.embed(text)
        self.cache[text] = v
        if len(self.cache) > 256:
            self.cache.popitem(last=False)
        return v, False

    def select(self, vec):
        scores = self.protos @ vec
        order = np.argsort(-scores)
        top, second = float(scores[order[0]]), float(scores[order[1]])
        if top < MIN_SCORE:
            conf, why = False, f"top score {top:.3f} < {MIN_SCORE:.2f}"
        elif top - second < MIN_GAP:
            conf, why = False, f"gap to #2 is {top - second:.3f} < {MIN_GAP:.2f}"
        else:
            conf, why = True, f"top {top:.3f}, {top - second:.3f} ahead of #2"
        ranked = [dict(tool=self.names[i], score=round(float(scores[i]), 4)) for i in order]
        return conf, why, ranked

    # ---------------------------------------------------------- extraction

    def enum_fallback(self, field, vec):
        if vec is None or field not in self.enum_mat:
            return None, None
        s = self.enum_mat[field] @ vec
        o = np.argsort(-s)
        gap = float(s[o[0]] - s[o[1]])
        detail = dict(value=self.enum_vals[field][o[0]], gap=round(gap, 3), accepted=gap >= ENUM_GAP)
        return (self.enum_vals[field][o[0]] if gap >= ENUM_GAP else None), detail

    def extract(self, tool, text, vec):
        """Deterministic argument extraction. Returns (args, problems, trace)."""
        args, problems, trace = {}, {}, []
        rest = text
        # 1. leading free text (message body, reminder text): taken from the original words.
        for p in tool["params"]:
            if p["type"] == "str" and p["name"] == "text":
                fn = extract_message if tool["name"] == "send_message" else extract_reminder
                val, span = fn(rest)
                if span:
                    rest = blank(rest, span)
                    if tool["name"] == "send_message":
                        val = tidy(val)
                if val:
                    args["text"] = val
                    trace.append(("text", "free text", val))
        norm = numbers_to_digits(rest)
        # 2. time of day
        for p in tool["params"]:
            if p["type"] == "time":
                full = numbers_to_digits(text)
                val, span, prob = parse_time(full)
                if val:
                    args[p["name"]] = val
                    trace.append((p["name"], "time regex", val))
                elif prob:
                    problems[p["name"]] = prob
                    trace.append((p["name"], "time regex", prob["said"]))
                if span:
                    norm = blank(norm, span) if len(norm) == len(full) else norm
        # 3. durations, then integers
        for p in tool["params"]:
            if p["type"] == "int" and p["name"] == "minutes":
                val, spans = parse_duration(norm)
                for sp in spans:
                    norm = blank(norm, sp)
                if val is None:
                    m = re.search(r"\b(?:timer|countdown|for|in)\s+(?:of\s+|for\s+)?(\d+)\b(?![:.]\d)", norm, re.I)
                    if m:
                        val = int(m.group(1))
                        norm = blank(norm, m.span(1))
                if val is not None:
                    args["minutes"] = val
                    trace.append(("minutes", "duration regex", val))
        lower = norm.lower()
        before_enums = lower
        for p in tool["params"]:
            if p["type"] == "enum":
                hits = []
                for val, rx in SYN_RE[p["name"]].items():
                    m = rx.search(lower)
                    if m:
                        hits.append((m.start(), val, m))
                if hits:
                    hits.sort(key=lambda h: h[0])
                    _, val, m = hits[0]
                    args[p["name"]] = val
                    lower = blank(lower, m.span())
                    trace.append((p["name"], "synonym", f"“{m.group(0).strip()}” → {val}"))
                else:
                    val, detail = self.enum_fallback(p["name"], vec)
                    if detail and (detail["accepted"] or p.get("required", True)):
                        trace.append((p["name"], "embedding",
                                      f"{detail['value']} (lead {detail['gap']:.3f}"
                                      f"{' ≥' if detail['accepted'] else ' <'} {ENUM_GAP})"))
                    if val:
                        args[p["name"]] = val
        for p in tool["params"]:
            if p["type"] == "int" and p["name"] != "minutes":
                val, span = parse_int(p["name"], before_enums)
                if val is not None:
                    args[p["name"]] = val
                    before_enums = blank(before_enums, span)
                    lower = blank(lower, span)
                    trace.append((p["name"], "number regex", val))
        for p in tool["params"]:
            if p["type"] == "bool":
                for val, pats in BOOL_WORDS[p["words"]]:
                    m = next((m for m in (re.search(x, lower, re.I) for x in pats) if m), None)
                    if m:
                        args[p["name"]] = val
                        trace.append((p["name"], "on/off words", f"“{m.group(0).strip()}” → {val}"))
                        break
                if p["name"] == "on" and "on" not in args and "brightness" in args:
                    args["on"] = args["brightness"] > 0
                    trace.append(("on", "inferred", "brightness given → on"))
        # 4. trailing free text (timer label) from whatever is left
        for p in tool["params"]:
            if p["type"] == "str" and p["name"] == "label":
                val, span = extract_label(norm)
                if val:
                    args["label"] = val
                    trace.append(("label", "free text", val))
        return args, problems, trace

    # ---------------------------------------------------------- validation

    @staticmethod
    def validate(tool, args, problems):
        """Every param checked against type, bounds, enum and required-ness. Returns issue list."""
        issues = []
        for p in tool["params"]:
            n, req = p["name"], p.get("required", True)
            v = args.get(n)
            if n in problems:
                prob = problems[n]
                if prob["kind"] == "ambiguous":
                    issues.append(dict(field=n, kind="ambiguous", options=prob["options"],
                                       question=f"“{prob['said']}”: morning or evening? {prob['options'][0]} / {prob['options'][1]}"))
                else:
                    issues.append(dict(field=n, kind="invalid", options=[],
                                       question=f"“{prob['said']}” is not a valid time; use HH:MM (24h)."))
                continue
            if v is None:
                if req:
                    issues.append(dict(field=n, kind="missing", options=Assistant.options(p), question=Assistant.ask(p)))
                continue
            t = p["type"]
            if t == "enum" and v not in p["values"]:
                issues.append(dict(field=n, kind="invalid", options=p["values"], question=f"{n} must be one of {', '.join(p['values'])}."))
            elif t == "int" and (not isinstance(v, int) or isinstance(v, bool) or not p["min"] <= v <= p["max"]):
                issues.append(dict(field=n, kind="invalid", options=[],
                                   question=f"{p['label']} must be between {p['min']} and {p['max']}; you said {v}."))
            elif t == "bool" and not isinstance(v, bool):
                issues.append(dict(field=n, kind="invalid", options=list(p["words"]), question=p["ask"]))
            elif t == "str" and (not isinstance(v, str) or not v.strip() or len(v) > 200):
                issues.append(dict(field=n, kind="invalid", options=[], question=p["ask"]))
            elif t == "time" and not (isinstance(v, str) and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v)):
                issues.append(dict(field=n, kind="invalid", options=[], question="Time must be HH:MM (24h)."))
        # Ask about a value the user actually said (and got wrong) before asking for a missing one.
        return sorted(issues, key=lambda i: i["kind"] == "missing")

    @staticmethod
    def options(p):
        if p["type"] == "enum":
            return p["values"]
        if p["type"] == "bool":
            return list(p["words"])
        return []

    @staticmethod
    def ask(p):
        if p["type"] == "enum":
            return f"{p['ask']} {' / '.join(p['values'])}"
        if p["type"] == "int":
            unit = f" {p['unit']}" if p.get("unit") else ""
            return f"{p['ask']} ({p['min']}–{p['max']}{unit})"
        if p["type"] == "bool":
            return f"{p['ask']} {' / '.join(p['words'])}"
        if p["type"] == "time":
            return f"{p['ask']} (HH:MM, 24h)"
        return p["ask"]

    # ---------------------------------------------------------- the pipeline

    def parse(self, text, tool_name=None, prev=None, field=None, execute=True):
        """tool_name forces a tool (picked from the 'unsure' options or a pending question);
        prev = args already collected for that tool and field = the field we asked about
        (answering a clarifying question)."""
        text = text.strip()[:500]
        t0 = time.perf_counter()
        vec, cached = self.embed(text)
        embed_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        conf, why, ranked = self.select(vec)
        select_ms = (time.perf_counter() - t1) * 1000
        mode = "new"
        t2 = time.perf_counter()
        if tool_name in TOOL_BY_NAME:
            tool = TOOL_BY_NAME[tool_name]
            args, problems, trace = self.extract(tool, text, vec)
            if prev is not None:
                # Answer to a clarifying question. A free-text field takes the whole answer.
                # Otherwise: if the answer does not fill the field we asked about and reads like a
                # different command, start over; else merge it with what we had.
                fp = next((p for p in tool["params"] if p["name"] == field), None)
                if fp and fp["type"] == "str" and not args.get(field):
                    args[field] = tidy(text)
                    trace.append((field, "free text", "the whole answer"))
                if field not in args and field not in problems and conf and ranked[0]["tool"] != tool_name:
                    tool = TOOL_BY_NAME[ranked[0]["tool"]]
                    args, problems, trace = self.extract(tool, text, vec)
                else:
                    mode = "answer"
                    merged = {k: v for k, v in prev.items() if k in {p["name"] for p in tool["params"]}}
                    merged.update(args)
                    args = merged
                    trace.insert(0, ("·", "kept from before", ", ".join(f"{k}={v}" for k, v in prev.items()) or "nothing"))
            else:
                mode = "forced"
        elif not conf:
            extract_ms = (time.perf_counter() - t2) * 1000
            top2 = ranked[:2]
            return self._out(text, "unsure", None, {}, [], [], ranked, why, cached, embed_ms, select_ms, extract_ms, 0,
                             t0, mode, question=f"Which action did you mean: {top2[0]['tool']} or {top2[1]['tool']}?",
                             options=[r["tool"] for r in top2])
        else:
            tool = TOOL_BY_NAME[ranked[0]["tool"]]
            args, problems, trace = self.extract(tool, text, vec)
        extract_ms = (time.perf_counter() - t2) * 1000
        t3 = time.perf_counter()
        issues = self.validate(tool, args, problems)
        validate_ms = (time.perf_counter() - t3) * 1000
        if issues:
            first = issues[0]
            return self._out(text, "clarify", tool, args, trace, issues, ranked, why, cached, embed_ms, select_ms,
                             extract_ms, validate_ms, t0, mode, question=first["question"], options=first["options"])
        call = {p["name"]: args[p["name"]] for p in tool["params"] if args.get(p["name"]) is not None}
        effect = self.execute(tool["name"], call) if execute else None
        return self._out(text, "ok", tool, call, trace, [], ranked, why, cached, embed_ms, select_ms, extract_ms,
                         validate_ms, t0, mode, effect=effect)

    def _out(self, text, status, tool, args, trace, issues, ranked, why, cached, embed_ms, select_ms, extract_ms,
             validate_ms, t0, mode, question=None, options=None, effect=None):
        out = dict(query=text, status=status, mode=mode, reason=why, cached=cached,
                   tool=tool["name"] if tool else None, args=args, ranked=ranked[:4],
                   trace=[dict(field=f, how=h, value=v) for f, h, v in trace],
                   issues=issues, question=question, options=options or [], effect=effect,
                   ms=dict(embed=round(embed_ms, 1), select=round(select_ms, 2), extract=round(extract_ms, 2),
                           validate=round(validate_ms, 2), total=round((time.perf_counter() - t0) * 1000, 1)))
        if tool:
            score = next(r["score"] for r in ranked if r["tool"] == tool["name"])
            out["score"] = score
        if status == "ok":
            out["code"] = format_call(tool["name"], args)
            out["json"] = dict(name=tool["name"], arguments=args)
        out["state"] = self.state
        return out

    # ---------------------------------------------------------- the "home"

    def reset(self):
        self.state = dict(
            lights={r: dict(on=False, brightness=100) for r in ROOMS},
            ac={},
            doors=dict(front=True, back=True),
            timer=None, music=None, message=None, reminders=[], order=None, log=[],
        )

    def execute(self, name, a):
        s = self.state
        if name == "set_light":
            s["lights"][a["room"]]["on"] = a["on"]
            if "brightness" in a:
                s["lights"][a["room"]]["brightness"] = a["brightness"]
            effect = f"{a['room']} light {'on' if a['on'] else 'off'}"
        elif name == "set_ac":
            prev = s["ac"].get(a["room"], {})
            s["ac"][a["room"]] = dict(temperature=a["temperature"], mode=a.get("mode", prev.get("mode", "auto")))
            effect = f"{a['room']} AC {a['temperature']}°C {s['ac'][a['room']]['mode']}"
        elif name == "set_timer":
            s["timer"] = dict(minutes=a["minutes"], label=a.get("label"), ends_at=time.time() + a["minutes"] * 60)
            effect = f"timer {a['minutes']} min"
        elif name == "play_music":
            s["music"] = dict(genre=a["genre"], volume=a.get("volume", (s["music"] or {}).get("volume", 50)))
            effect = f"playing {a['genre']}"
        elif name == "send_message":
            s["message"] = dict(contact=a["contact"], text=a["text"], at=time.strftime("%H:%M"))
            effect = f"sent to {a['contact']}"
        elif name == "set_reminder":
            new = dict(time=a["time"], text=a["text"])
            s["reminders"] = sorted([r for r in s["reminders"] if r != new] + [new], key=lambda r: r["time"])[-4:]
            effect = f"reminder at {a['time']}"
        elif name == "lock_door":
            s["doors"][a["door"]] = a["locked"]
            effect = f"{a['door']} door {'locked' if a['locked'] else 'unlocked'}"
        elif name == "order_food":
            s["order"] = dict(dish=a["dish"], quantity=a["quantity"], at=time.strftime("%H:%M"))
            effect = f"ordered {a['quantity']} × {a['dish']}"
        s["log"] = ([dict(code=format_call(name, a), at=time.strftime("%H:%M:%S"))] + s["log"])[:6]
        return effect

    def info(self):
        return dict(model=os.path.basename(MODEL), dims=int(self.protos.shape[1]), tools=[signature(t) for t in TOOLS],
                    min_score=MIN_SCORE, min_gap=MIN_GAP, enum_gap=ENUM_GAP, desc_weight=DESC_WEIGHT,
                    load_s=round(self.emb.load_s, 2), build_s=round(self.build_s, 1), prefix=PREFIX,
                    examples=sum(len(t["examples"]) for t in TOOLS))


def py_value(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def format_call(name, args):
    return f"{name}({', '.join(f'{k}={py_value(v)}' for k, v in args.items())})"


def signature(t):
    params = []
    for p in t["params"]:
        if p["type"] == "enum":
            ty = " | ".join(json.dumps(v) for v in p["values"])
        elif p["type"] == "int":
            ty = f"int[{p['min']}..{p['max']}]"
        elif p["type"] == "time":
            ty = '"HH:MM"'
        else:
            ty = p["type"]
        params.append(dict(name=p["name"], type=ty, required=p.get("required", True), ask=p["ask"]))
    return dict(name=t["name"], description=t["description"], params=params, examples=t["examples"],
                schema=json_schema(t))


def json_schema(t):
    """The tool as a function-calling JSON Schema (what an LLM tool API would be given)."""
    props, req = {}, []
    for p in t["params"]:
        if p["type"] == "enum":
            sch = {"type": "string", "enum": p["values"]}
        elif p["type"] == "int":
            sch = {"type": "integer", "minimum": p["min"], "maximum": p["max"]}
        elif p["type"] == "bool":
            sch = {"type": "boolean"}
        elif p["type"] == "time":
            sch = {"type": "string", "pattern": "^([01]\\d|2[0-3]):[0-5]\\d$"}
        else:
            sch = {"type": "string"}
        props[p["name"]] = sch
        if p.get("required", True):
            req.append(p["name"])
    return {"name": t["name"], "description": t["description"],
            "parameters": {"type": "object", "properties": props, "required": req, "additionalProperties": False}}


# ================================================================ eval

def run_eval():
    with open(TESTS, encoding="utf-8") as f:
        tests = json.load(f)
    a = Assistant()
    log(f"Model loaded in {a.emb.load_s:.2f} s. {len(TOOLS)} tools, {sum(len(t['examples']) for t in TOOLS)} example"
        f" utterances. Running {len(tests)} test utterances ...\n")
    rows, lat = [], []
    for t in tests:
        a.reset()
        a.cache.clear()
        r = a.parse(t["q"], execute=False)
        lat.append(r["ms"])
        rows.append((t, r))

    def tool_ok(t, r):
        return r["status"] != "unsure" and r["tool"] == t.get("tool")

    def is_good(t, r):
        if t["expect"] == "call":
            return r["status"] == "ok" and r["tool"] == t["tool"] and r["args"] == t["args"]
        if t["expect"] == "clarify":
            return r["status"] == "clarify" and r["tool"] == t["tool"] and r["issues"][0]["field"] == t["field"]
        return r["status"] == "unsure"

    pct = lambda k, n: f"{k}/{n} = {k / n * 100:.0f}%" if n else "-"
    sets = [("main set (written with the tools)", [x for x in rows if x[0].get("set") != "hard"]),
            ("hard set (unseen phrasings found while probing)", [x for x in rows if x[0].get("set") == "hard"]),
            ("all", rows)]
    for title, rs in sets:
        with_tool = [(t, r) for t, r in rs if t.get("tool")]
        calls = [(t, r) for t, r in rs if t["expect"] == "call"]
        clar = [(t, r) for t, r in rs if t["expect"] == "clarify"]
        uns = [(t, r) for t, r in rs if t["expect"] == "unsure"]
        wrong = sum(r["status"] == "ok" and not is_good(t, r) for t, r in rs)
        log(f"{title}: {len(rs)} utterances")
        log(f"  Tool selection, top-1 (ignoring the unsure rule):  {pct(sum(r['ranked'][0]['tool'] == t['tool'] for t, r in with_tool), len(with_tool))}")
        log(f"  Tool selection, confident and right:               {pct(sum(tool_ok(t, r) for t, r in with_tool), len(with_tool))}")
        log(f"  Exact typed call (tool + every argument):          {pct(sum(is_good(t, r) for t, r in calls), len(calls))}")
        log(f"  Clarifying question on the right field:            {pct(sum(is_good(t, r) for t, r in clar), len(clar))}")
        log(f"  'Unsure which action' when expected:               {pct(sum(is_good(t, r) for t, r in uns), len(uns))}")
        log(f"  Wrong calls executed (valid call, wrong content):  {wrong}")
    emb = [m["embed"] for m in lat]
    tot = [m["total"] for m in lat]
    log(f"\nLatency per utterance: embed median {np.median(emb):.0f} ms (p90 {np.percentile(emb, 90):.0f}),"
        f" total median {np.median(tot):.0f} ms; extraction + validation median "
        f"{np.median([m['extract'] + m['validate'] for m in lat]):.2f} ms")
    log(f"Index: {len(index_texts())} texts, first build {a.build_s:.1f} s, then cached.\n")

    # threshold table: how the unsure rule trades off
    tops = np.array([r["ranked"][0]["score"] for _, r in rows])
    gaps = np.array([r["ranked"][0]["score"] - r["ranked"][1]["score"] for _, r in rows])
    right = np.array([bool(t.get("tool")) and r["ranked"][0]["tool"] == t["tool"] for t, r in rows])
    want_unsure = np.array([t["expect"] == "unsure" for t, _ in rows])
    log(f"{'min score':>10}{'min gap':>9}{'in-scope confident+right':>27}{'off-topic caught':>18}")
    for ms in (0.0, 0.75, 0.78, 0.80, 0.82):
        for mg in (0.0, 0.02, 0.03, 0.04):
            conf = (tops >= ms) & (gaps >= mg)
            mark = "  <- app" if (ms, mg) == (MIN_SCORE, MIN_GAP) else ""
            log(f"{ms:>10.2f}{mg:>9.2f}{(conf & right)[~want_unsure].sum():>21}/{(~want_unsure).sum()}"
                f"{(~conf)[want_unsure].sum():>14}/{want_unsure.sum()}{mark}")

    log("\nPer utterance (x = not what tests.json expects, h = hard set):")
    for t, r in rows:
        good = is_good(t, r)
        got = r.get("code") or f"{r['status']} {r['tool'] or ''}: {r['question']}"
        sc = r["ranked"][0]
        log(f" {' ' if good else 'x'}{'h' if t.get('set') == 'hard' else ' '} {t['q'][:56]:<56} {sc['tool']:<12} {sc['score']:.3f}"
            f" gap {sc['score'] - r['ranked'][1]['score']:.3f}  -> {got}")
        if not good and t["expect"] == "call":
            log(f"       expected {format_call(t['tool'], t['args'])}")
    a.emb.close()


# ================================================================ server

VOICE_URL = os.environ.get("VOICE_URL", "http://127.0.0.1:8092")


def voice_text(t):
    """Whisper writes '7.30pm' and ends with a full stop; the text pipeline wants '7:30 pm'."""
    t = re.sub(r"\b(\d{1,2})\.(\d{2})\s*([ap])\.?m\.?", r"\1:\2 \3m", t, flags=re.I)
    t = re.sub(r"\b(\d{1,2})\s*([ap])\.m\.", r"\1 \2m", t, flags=re.I)
    return t.strip().rstrip(".!?。").strip()


def voice_call(path, body=None, timeout=30):
    import urllib.request
    req = urllib.request.Request(VOICE_URL + path, data=body, headers={"Content-Type": "audio/wav"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def hear(asst, wav, qs, lock):
    """Spoken command: the sound picks the action on its own, Whisper's words fill the values.
    The text path stays in charge of the call (it validates every value); when the words alone
    are not confident about the action, the sound's confident pick is used instead."""
    lang = qs.get("lang", ["en"])[0]
    h = voice_call("/hear?lang=" + urllib.parse.quote(lang), wav)
    text = voice_text(h["words"]["text"])
    sound = h["sound"]
    pick = sound["ranked"][0]["tool"] if sound["confident"] else None
    voice = dict(transcript=h["words"]["text"], text=text, lang=h["words"]["lang"], seconds=h["seconds"],
                 sound=sound, stt_ms=h["words"]["ms"], sound_ms=sound["ms"], used="words")
    if not text:
        return dict(status="unheard", voice=voice, state=asst.state)
    prev = None
    if "prev" in qs:
        try:
            prev = json.loads(qs["prev"][0])
            prev = prev if isinstance(prev, dict) else None
        except ValueError:
            prev = None
    with lock:
        if prev is not None:   # answering a clarifying question
            out = asst.parse(text, qs.get("tool", [None])[0], prev, qs.get("field", [None])[0])
            voice["used"] = "answer"
        else:
            first = asst.parse(text, execute=False)
            if first["status"] == "unsure" and pick:
                out = asst.parse(text, pick)
                voice["used"] = "sound"
            else:
                out = asst.parse(text)
            out["ms"], out["cached"] = first["ms"], first["cached"]  # second pass reuses the cached vector
    voice["agree"] = bool(out.get("tool")) and out.get("tool") == sound["ranked"][0]["tool"]
    out["voice"] = voice
    return out


def make_handler(asst, page):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code=200):
            self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    with open(page, "rb") as f:  # re-read so UI edits show on reload
                        self.send(200, f.read(), "text/html; charset=utf-8")
                elif url.path == "/api/parse":
                    q = qs.get("q", [""])[0]
                    if not q.strip():
                        return self.json({"error": "empty query"}, 400)
                    prev = None
                    if "prev" in qs:
                        try:
                            prev = json.loads(qs["prev"][0])
                            prev = prev if isinstance(prev, dict) else None
                        except ValueError:
                            prev = None
                    with lock:
                        out = asst.parse(q, qs.get("tool", [None])[0], prev, qs.get("field", [None])[0])
                    self.json(out)
                elif url.path == "/api/state":
                    self.json(asst.state)
                elif url.path == "/api/reset":
                    with lock:
                        asst.reset()
                    self.json(asst.state)
                elif url.path == "/api/info":
                    self.json(asst.info())
                elif url.path == "/api/voice":
                    try:
                        self.json(dict(ok=True, **voice_call("/health", timeout=2)))
                    except Exception:
                        self.json(dict(ok=False, url=VOICE_URL))
                else:
                    self.send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            if url.path != "/api/hear":
                return self.send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= 2_000_000:
                return self.json({"error": "send up to 30 s of 16-bit 16 kHz WAV"}, 400)
            wav = self.rfile.read(n)
            try:
                self.json(hear(asst, wav, urllib.parse.parse_qs(url.query), lock))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                self.json({"error": "voice server not reachable: " + repr(e)[:160]}, 503)

    return Handler


def serve(host, port):
    page = os.path.join(HERE, "index.html")
    asst = Assistant()
    log(f"Model loaded in {asst.emb.load_s:.2f} s; {len(TOOLS)} typed tools ready.")
    srv = ThreadingHTTPServer((host, port), make_handler(asst, page))
    srv.daemon_threads = True
    log(f"\n  Typed assistant running at  http://{host}:{port}\n  (Ctrl+C to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        asst.emb.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true", help="run tests.json and report accuracy")
    ap.add_argument("--build", action="store_true", help="embed the tool/enum index into data/ and exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    if args.build:
        _, _, n, emb = load_vectors(Embedder)
        log(f"Index ready ({n} new embeddings).")
        if emb:
            emb.close()
    elif args.eval:
        run_eval()
    else:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
