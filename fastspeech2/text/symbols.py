""" from https://github.com/keithito/tacotron

MODIFIED for this project: symbol tables are now per language.
  * "en": upstream table minus pinyin (pad, punctuation, letters, @ARPAbet, @silences)
  * "te": pad + @Telugu phones (see text/telugu.py) + @silences
Each model only allocates embeddings for its own language. Upstream used one
global table (ARPAbet + pinyin) for every dataset.
"""

from text import cmudict
from text import telugu

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_telugu = ["@" + s for s in telugu.PHONES]

SYMBOL_SETS = {
    "en": [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet + _silences,
    "te": [_pad] + _telugu + _silences,
}


def get_symbols(lang):
    return SYMBOL_SETS[lang]


# Backwards compatibility with upstream imports
symbols = SYMBOL_SETS["en"]
