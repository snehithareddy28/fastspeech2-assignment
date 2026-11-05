"""English text frontend (new, replaces ming024's preprocess_english).

Two jobs:
  1. normalize_english: expand times, dates, years, ordinals, cardinals and
     a few abbreviations into words. The same function is used to write the
     MFA transcripts (training) and at inference, so both see identical text.
  2. text_to_phones: sentence -> ARPAbet phones. Uses g2p_en on the whole
     normalised sentence, so its POS-tagger based homograph table can pick
     e.g. CONDUCT (noun, K AA1 N D AH0 K T) vs CONDUCT (verb, K AH0 N D AH1 K T).
     ming024 instead looked words up one at a time in a lexicon, which always
     returns the first pronunciation and cannot disambiguate heteronyms.
     Out-of-vocabulary words (e.g. "andaroo", "rhyth") go through g2p_en's
     neural G2P.

Letter-name spellings (AM, PM, acronyms) are emitted inline as ARPAbet in
curly braces, the convention upstream text_to_sequence already supports.
"""
import re

import inflect
from unidecode import unidecode

_inflect = inflect.engine()

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
_MONTH_RE = "|".join(MONTHS)

LETTER_ARPA = {
    "a": "EY1", "b": "B IY1", "c": "S IY1", "d": "D IY1", "e": "IY1", "f": "EH1 F",
    "g": "JH IY1", "h": "EY1 CH", "i": "AY1", "j": "JH EY1", "k": "K EY1",
    "l": "EH1 L", "m": "EH1 M", "n": "EH1 N", "o": "OW1", "p": "P IY1",
    "q": "K Y UW1", "r": "AA1 R", "s": "EH1 S", "t": "T IY1", "u": "Y UW1",
    "v": "V IY1", "w": "D AH1 B AH0 L Y UW0", "x": "EH1 K S", "y": "W AY1",
    "z": "Z IY1",
}

_ABBREV = [(re.compile(r"\b%s\." % a, re.IGNORECASE), b) for a, b in [
    ("mrs", "missus"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"),
    ("jr", "junior"), ("prof", "professor"), ("etc", "et cetera"), ("vs", "versus"),
]]


def _num(n):
    return _inflect.number_to_words(int(n), andword="").replace("-", " ").replace(",", "")


def _ordinal(n):
    return _inflect.number_to_words(_inflect.ordinal(int(n)), andword="").replace(
        "-", " ").replace(",", "")


def _year(y):
    y = int(y)
    if 2000 <= y <= 2009:
        return _num(y)
    if 1000 <= y <= 2099 and y % 100 == 0:
        return _num(y // 100) + " hundred"
    if 1000 <= y <= 2099:
        hi, lo = divmod(y, 100)
        lo_w = ("oh " + _num(lo)) if lo < 10 else _num(lo)
        return _num(hi) + " " + lo_w
    return _num(y)


_PROTECT = []


def _spell(letters, for_alignment):
    letters = [c for c in letters.lower() if c.isalpha()]
    if for_alignment:
        return " ".join(letters)
    # protected placeholder, so later digit expansion cannot touch the stress digits
    _PROTECT.append("{" + " ".join(LETTER_ARPA[c] for c in letters) + "}")
    # index encoded with letters: digit expansion runs later and would rewrite digits
    return " QXPROTECT%sQX " % "".join("ABCDEFGHIJ"[int(c)] for c in str(len(_PROTECT) - 1))


def normalize_english(text, for_alignment=False):
    """Expand non-standard words.

    Rules (documented in the report):
      * H:MM [AM|PM] -> "<hour> <minutes>" (+ letter names for AM/PM);
        ":00" -> "o'clock"; minutes 01-09 -> "oh <n>".
      * <day> <Month> <year> -> "<ordinal day> <Month> <year>"
        (12 November 2026 -> twelfth November twenty twenty six).
      * <Month> <day>, <year> -> "<Month> <ordinal day> <year>".
      * 4-digit numbers 1000-2099 elsewhere are read as years; other digit runs
        as cardinals; 1st/2nd/3rd/4th as ordinals; "%" -> percent.
    With for_alignment=True, letter names are written as plain letters so MFA
    can look them up in its lexicon.
    """
    text = unidecode(text)
    text = text.replace("’", "'")
    for regex, rep in _ABBREV:
        text = regex.sub(rep, text)

    def _time(m):
        h, mnt, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
        if mnt == 0:
            words = _num(h) + " o'clock"
        elif mnt < 10:
            words = _num(h) + " oh " + _num(mnt)
        else:
            words = _num(h) + " " + _num(mnt)
        if ampm:
            words += " " + _spell(ampm, for_alignment)
        return words

    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?\s?[Mm]\.?)?(?![A-Za-z])", _time, text)
    text = re.sub(r"\b(\d{1,2})\s+(%s)\s+(\d{4})\b" % _MONTH_RE,
                  lambda m: "%s %s %s" % (_ordinal(m.group(1)), m.group(2), _year(m.group(3))),
                  text, flags=re.IGNORECASE)
    text = re.sub(r"\b(%s)\s+(\d{1,2}),?\s+(\d{4})\b" % _MONTH_RE,
                  lambda m: "%s %s %s" % (m.group(1), _ordinal(m.group(2)), _year(m.group(3))),
                  text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d{1,2})\s+(%s)\b" % _MONTH_RE,
                  lambda m: "%s %s" % (_ordinal(m.group(1)), m.group(2)), text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _ordinal(m.group(1)), text)
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?%", r"\1 percent", text)
    text = re.sub(r"\b(\d+)\.(\d+)\b",
                  lambda m: _num(m.group(1)) + " point " + " ".join(_num(d) for d in m.group(2)),
                  text)
    text = re.sub(r"\b(1\d{3}|20\d{2})\b", lambda m: _year(m.group(1)), text)
    text = re.sub(r"\d+", lambda m: " " + _num(m.group(0)) + " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"QXPROTECT([A-J]+)QX",
                  lambda m: _PROTECT[int("".join(str("ABCDEFGHIJ".index(c)) for c in m.group(1)))],
                  text)
    del _PROTECT[:]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def words_for_alignment(raw_text):
    """Lower-cased words, apostrophes kept, other punctuation removed (MFA .lab)."""
    t = normalize_english(raw_text, for_alignment=True).lower()
    t = re.sub(r"[^a-z' ]", " ", t)
    t = re.sub(r"(^|\s)'+|'+(\s|$)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        import nltk
        for res, path in [("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
                          ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                          ("cmudict", "corpora/cmudict"), ("punkt", "tokenizers/punkt")]:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(res, quiet=True)
                except Exception:
                    pass
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


_PUNCT_PAUSE = set(",.;:?!")

# Inference-only rule: re-join a word split by a stray space ("rhyth m" -> "rhythm").
# Two adjacent alphabetic tokens are joined when the joined form is in CMUdict and
# one part is not a real word on its own (not in CMUdict, or a single letter other
# than "a"/"i"). Words that are both valid ("in to", "a way") are left alone.
# Set to False (synthesize --no_join) to reproduce the unjoined behaviour.
JOIN_SPLIT_WORDS = True


def _join_split_words(words, cmu):
    def lone(w):
        return w not in cmu or (len(w) == 1 and w not in ("a", "i"))
    out, joined, i = [], set(), 0
    while i < len(words):
        a = words[i]
        b = words[i + 1] if i + 1 < len(words) else None
        if (b is not None and a.isalpha() and b.isalpha() and (a + b) in cmu
                and (lone(a) or lone(b))):
            out.append(a + b)
            joined.add(len(out) - 1)
            i += 2
        else:
            out.append(a)
            i += 1
    return out, joined


def _g2p_words(segment):
    """Mirror of g2p_en.G2p.__call__ (v2.1.0) that also reports, per word, which
    source produced the pronunciation: 'homograph' (POS-based), 'cmudict' or
    'neural' (OOV prediction). Same output phones as g2p_en."""
    from nltk import pos_tag
    from nltk.tokenize import TweetTokenizer

    g2p = _get_g2p()
    text = segment.lower()
    text = re.sub(r"[^ a-z'.,?!\-]", "", text)
    text = text.replace("i.e.", "that is").replace("e.g.", "for example")
    try:
        from nltk import word_tokenize
        words = word_tokenize(text)
    except LookupError:
        words = TweetTokenizer().tokenize(text)
    joined = set()
    if JOIN_SPLIT_WORDS:
        words, joined = _join_split_words(words, g2p.cmu)
    result = []
    for k, (word, pos) in enumerate(pos_tag(words)):
        if re.search("[a-z]", word) is None:
            result.append((word, [word], "punct"))
            continue
        if word in g2p.homograph2features:
            pron1, pron2, pos1 = g2p.homograph2features[word]
            pron = pron1 if pos.startswith(pos1) else pron2
            src = "homograph(%s)" % pos
        elif word in g2p.cmu:
            pron = g2p.cmu[word][0]
            src = "cmudict"
        else:
            pron = g2p.predict(word)
            src = "neural"
        if k in joined:
            src = "joined+" + src
        result.append((word, list(pron), src))
    return result


def text_to_phones(text, verbose=False, return_sources=False):
    """Raw English sentence -> (phones, normalized_text, word_phones).

    word_phones is a list of (word, phones, source) triples.
    """
    norm = normalize_english(text)
    phones, word_phones = [], []
    for seg in re.split(r"(\{[^}]*\})", norm):
        if not seg.strip():
            continue
        if seg.startswith("{"):
            ph = seg[1:-1].split()
            phones.extend(ph)
            word_phones.append((seg, ph, "letters"))
            continue
        for word, pron, src in _g2p_words(seg):
            if src == "punct":
                if word in _PUNCT_PAUSE and phones and phones[-1] != "sp":
                    phones.append("sp")
                    word_phones.append((word, ["sp"], "pause"))
                continue
            pron = [p for p in pron if re.fullmatch(r"[A-Z]+[012]?", p)]
            phones.extend(pron)
            word_phones.append((word, pron, src))
    while phones and phones[-1] == "sp":
        phones.pop()
        word_phones.pop()
    if verbose:
        for w, p, s in word_phones:
            print("  {:<28s} {:<16s} {}".format(w, s, " ".join(p)))
    return phones, norm, word_phones


def build_lexicon_entries(words, base_lexicon_path):
    """Pronunciations for MFA: every variant from the base lexicon, g2p_en for OOVs.

    Keeping all variants lets MFA choose the acoustically best one for
    heteronyms in the training data.
    """
    base = {}
    with open(base_lexicon_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            w = parts[0].lower()
            pron = " ".join(parts[1:])
            base.setdefault(w, [])
            if pron not in base[w]:
                base[w].append(pron)
    g2p = _get_g2p()
    entries, oov = [], []
    for w in sorted(set(words)):
        if w in base:
            for pron in base[w]:
                entries.append((w, pron))
        else:
            ph = [p for p in g2p(w) if re.fullmatch(r"[A-Z]+[012]?", p)]
            if len(w) == 1 and w in LETTER_ARPA:
                ph = LETTER_ARPA[w].split()
            if ph:
                entries.append((w, " ".join(ph)))
                oov.append(w)
    return entries, oov
