""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import get_symbols

# MODIFIED: per-language symbol tables (upstream had one global table).
_symbol_to_id = {}
_id_to_symbol = {}
for _lang in ("en", "te"):
    _syms = get_symbols(_lang)
    _symbol_to_id[_lang] = {s: i for i, s in enumerate(_syms)}
    _id_to_symbol[_lang] = {i: s for i, s in enumerate(_syms)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names, lang="en"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)

        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names), lang)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names), lang)
        sequence += _arpabet_to_sequence(m.group(2), lang)
        text = m.group(3)

    return sequence


def sequence_to_text(sequence, lang="en"):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol[lang]:
            s = _id_to_symbol[lang][symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols, lang="en"):
    out = []
    for s in symbols:
        if s in _symbol_to_id[lang] and s != "_" and s != "~":
            out.append(_symbol_to_id[lang][s])
        elif s.startswith("@"):
            # MODIFIED: upstream silently dropped unknown phones; fail loudly instead
            raise KeyError("Unknown phone {!r} for language {!r}".format(s[1:], lang))
    return out


def _arpabet_to_sequence(text, lang="en"):
    return _symbols_to_sequence(["@" + s for s in text.split()], lang)
