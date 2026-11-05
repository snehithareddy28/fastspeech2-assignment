"""Telugu text frontend (new, not in ming024).

Telugu script is an abugida that maps almost one-to-one onto pronunciation, so
a rule-based grapheme-to-phoneme (G2P) converter is adequate for TTS:

  * every consonant letter carries the inherent vowel /a/ unless it is followed
    by a vowel sign (matra) or the virama (్), which kills the vowel;
  * conjuncts are written as consonant + virama + consonant, so they fall out of
    the same rule (e.g. పట్టు -> p a tx tx u);
  * the anusvara (ం) becomes a nasal homorganic with the next consonant
    (k/g -> ng, c/j -> ny, retroflex -> nx, dental -> n, labial -> m) and /m/
    word-finally or before sonorants/fricatives (సమావేశం -> ... sh a m);
  * visarga (ః) -> /h/; candrabindu is dropped.

Known simplifications (documented in the report): no word-initial e -> ye
glide, no vowel harmony, no sandhi across words, no English loanword
handling. Words containing Latin letters are rejected by `is_telugu_text` and
such utterances are filtered out of training data.

Phones are ASCII strings so they can be used as MFA phones and as symbols in
the model's embedding table. Retroflexes use an "x" suffix.
"""
import re
import sys
import unicodedata

# ----------------------------------------------------------------------------
# Phone inventory
# ----------------------------------------------------------------------------
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tx", "ఠ": "txh", "డ": "dx", "ఢ": "dxh", "ణ": "nx",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "rx", "ల": "l", "ళ": "lx", "ఴ": "zh",
    "వ": "v", "శ": "sh", "ష": "sx", "స": "s", "హ": "h",
    "ౘ": "ts", "ౙ": "dz", "ౚ": "rx",
}
INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఌ": ["l", "u"], "ౡ": ["l", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
MATRAS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ౢ": ["l", "u"], "ౣ": ["l", "uu"],
    "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}
VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"
SILENT_MARKS = {"ఀ", "ఁ", "ఄ", "఼", "ఽ", "ౕ", "ౖ",
                "‌", "‍"}  # candrabindu, nukta, avagraha, length marks, ZWNJ/ZWJ

VOWEL_PHONES = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
NASAL_PHONES = ["ng", "ny", "nx", "n", "m"]
PHONES = sorted(set(VOWEL_PHONES) | set(CONSONANTS.values()) | set(NASAL_PHONES))
SILENCES = ["sp", "spn", "sil"]

_VELAR = {"k", "kh", "g", "gh"}
_PALATAL = {"c", "ch", "j", "jh"}
_RETRO = {"tx", "txh", "dx", "dxh"}
_DENTAL = {"t", "th", "d", "dh"}
_LABIAL = {"p", "ph", "b", "bh"}

TELUGU_BLOCK = re.compile(r"[ఀ-౿]")
TELUGU_DIGITS = {chr(0x0C66 + i): str(i) for i in range(10)}


def _anusvara_phone(next_cons):
    if next_cons in _VELAR:
        return "ng"
    if next_cons in _PALATAL:
        return "ny"
    if next_cons in _RETRO:
        return "nx"
    if next_cons in _DENTAL:
        return "n"
    if next_cons in _LABIAL:
        return "m"
    return "m"


def word_to_phones(word):
    """Rule-based G2P for one Telugu word (Telugu letters only)."""
    word = unicodedata.normalize("NFC", word)
    out = []
    pending = False  # a consonant waiting for its vowel
    anusvara_slots = []
    chars = list(word)
    for ch in chars:
        if ch in CONSONANTS:
            if pending:
                out.append("a")
            out.append(CONSONANTS[ch])
            pending = True
        elif ch in MATRAS:
            out.extend(MATRAS[ch])
            pending = False
        elif ch == VIRAMA:
            pending = False
        elif ch in INDEPENDENT_VOWELS:
            if pending:
                out.append("a")
            out.extend(INDEPENDENT_VOWELS[ch])
            pending = False
        elif ch == ANUSVARA:
            if pending:
                out.append("a")
                pending = False
            anusvara_slots.append(len(out))
            out.append("<N>")
        elif ch == VISARGA:
            if pending:
                out.append("a")
                pending = False
            out.append("h")
        elif ch in SILENT_MARKS:
            continue
        else:
            raise ValueError("Unsupported character {!r} (U+{:04X}) in word {!r}".format(
                ch, ord(ch), word))
    if pending:
        out.append("a")
    for i in anusvara_slots:
        nxt = out[i + 1] if i + 1 < len(out) else None
        out[i] = _anusvara_phone(nxt)
    return out


# ----------------------------------------------------------------------------
# Number normalisation
# ----------------------------------------------------------------------------
_ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది",
         "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను",
         "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై",
         8: "ఎనభై", 9: "తొంభై"}


def _below_100(n):
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] if o == 0 else _TENS[t] + " " + _ONES[o]


def _below_1000(n):
    if n < 100:
        return _below_100(n)
    h, r = divmod(n, 100)
    if h == 1:
        head_alone, head_comb = "వంద", "నూట"
    else:
        head_alone, head_comb = _ONES[h] + " వందలు", _ONES[h] + " వందల"
    return head_alone if r == 0 else head_comb + " " + _below_100(r)


def telugu_number(n):
    """Cardinal number words, 0 <= n < 10^9 (Indian grouping: వేలు, లక్షలు, కోట్లు)."""
    n = int(n)
    if n < 1000:
        return _below_1000(n)
    for unit, alone_1, comb_1, alone_n, comb_n in [
        (10 ** 7, "కోటి", "కోటి", "కోట్లు", "కోట్ల"),
        (10 ** 5, "లక్ష", "లక్ష", "లక్షలు", "లక్షల"),
        (10 ** 3, "వెయ్యి", "వెయ్యి", "వేలు", "వేల"),
    ]:
        if n >= unit:
            q, r = divmod(n, unit)
            if q == 1:
                head_alone, head_comb = alone_1, comb_1
            else:
                qw = telugu_number(q)
                head_alone, head_comb = qw + " " + alone_n, qw + " " + comb_n
            return head_alone if r == 0 else head_comb + " " + telugu_number(r)
    raise ValueError(n)


def telugu_ordinal(n):
    """Ordinal used before the suffix వ, e.g. 12వ -> పన్నెండవ, 1వ -> ఒకటవ."""
    words = telugu_number(n).split(" ")
    last = words[-1]
    if last[-1] in ("ు", "ి"):
        last = last[:-1] + "వ"
    elif last[-1] in ("ై", "ా"):
        last = last + "యవ"
    else:
        last = last + "వ"
    words[-1] = last
    return " ".join(words)


_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_ORD_RE = re.compile(r"(\d+)\s?వ(?![ా-్ౕౖ])")  # 12వ (not 12వా ...)
_DEC_RE = re.compile(r"(\d+)\.(\d+)")
_NUM_RE = re.compile(r"\d+")


def normalize_telugu(text):
    """Expand digits, times and ordinals into Telugu words.

    Rules (documented in the report):
      * Telugu digits (౦-౯) are mapped to ASCII digits first.
      * HH:MM -> "<hours> <minutes>" read as cardinals (10:45 -> పది నలభై ఐదు);
        ":00" is dropped.
      * <number>వ -> ordinal form (12వ -> పన్నెండవ).
      * a.b -> "<a> పాయింట్ <digits of b>".
      * other digit runs -> cardinals with Indian grouping (2026 -> రెండు వేల ఇరవై ఆరు).
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(TELUGU_DIGITS.get(c, c) for c in text)
    text = re.sub(r"(?<=\d),(?=\d{2,3}\b)", "", text)  # 1,000 -> 1000

    def _time(m):
        h, mnt = int(m.group(1)), int(m.group(2))
        return telugu_number(h) if mnt == 0 else telugu_number(h) + " " + telugu_number(mnt)

    text = _TIME_RE.sub(_time, text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?%", r"\1 శాతం", text)  # 50% -> యాభై శాతం
    text = _ORD_RE.sub(lambda m: telugu_ordinal(int(m.group(1))), text)
    text = _DEC_RE.sub(lambda m: telugu_number(int(m.group(1))) + " పాయింట్ " +
                       " ".join(_ONES[int(d)] for d in m.group(2)), text)
    text = _NUM_RE.sub(lambda m: telugu_number(int(m.group(0))), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------------
# Sentence level
# ----------------------------------------------------------------------------
_PAUSE_PUNCT = set(",.?!;:।॥-–—")
_TOKEN_RE = re.compile(r"[ఀ-౿‌‍]+|[^\sఀ-౿]")


def is_telugu_text(text):
    """True if, after normalisation, only Telugu letters, spaces and punctuation remain."""
    for ch in text:
        if ch.isspace() or ch in _PAUSE_PUNCT or ch in "\"'()[]“”‘’":
            continue
        if not TELUGU_BLOCK.match(ch) and ch not in "‌‍":
            return False
    return True


def words_for_alignment(normalized_text):
    """Space-separated Telugu words with punctuation removed (for MFA .lab files)."""
    return " ".join(_TOKEN_RE.findall(re.sub(r"[^ఀ-౿‌‍\s]", " ",
                                                normalized_text)))


def text_to_phones(text, verbose=False):
    """Raw Telugu sentence -> list of phones, with 'sp' at punctuation.

    Returns (phones, normalized_text, word_phones) where word_phones is a list of
    (word, phones, source) triples for logging and plots.
    """
    norm = normalize_telugu(text)
    phones, word_phones, skipped = [], [], []
    for tok in _TOKEN_RE.findall(norm):
        if TELUGU_BLOCK.match(tok[0]):
            ph = word_to_phones(tok)
            phones.extend(ph)
            word_phones.append((tok, ph, "rule"))
        elif tok in _PAUSE_PUNCT:
            if phones and phones[-1] != "sp":
                phones.append("sp")
                word_phones.append((tok, ["sp"], "pause"))
        elif tok not in "\"'()[]“”‘’":
            skipped.append(tok)
    if skipped:
        # not silent: Latin words or symbols in Telugu input are dropped
        print("[telugu g2p] warning: dropped unsupported characters {!r}".format(
            "".join(skipped)), file=sys.stderr)
    while phones and phones[-1] == "sp":
        phones.pop()
        word_phones.pop()
    return phones, norm, word_phones
