"""Title cleaning and tokenisation shared by all lexical components."""

from __future__ import annotations

import codecs
import html
import re

# Units that should stick to the number before them: "128 GB" and "128gb" both become "128gb".
UNITS = {
    "gb", "tb", "mb", "ml", "l", "lt", "liter", "g", "gr", "gram", "kg", "mg",
    "cm", "mm", "m", "inch", "w", "watt", "mah", "v", "pcs", "pc", "x",
}

# Rung 1b: spellings of one unit mapped to a canonical unit and multiplier, so "1 kg", "1000 gr" and
# "1000gram" all become "1000g". Only used when canonical_units=True, so rung 1 stays reproducible.
CANONICAL_UNITS = {
    **dict.fromkeys(["g", "gr", "grm", "gram", "grams", "gramm"], ("g", 1)),
    **dict.fromkeys(["kg", "kgs", "kilo", "kilogram"], ("g", 1000)),
    **dict.fromkeys(["ml", "mls", "mililiter", "milliliter"], ("ml", 1)),
    **dict.fromkeys(["l", "lt", "ltr", "liter", "litre"], ("ml", 1000)),
    **dict.fromkeys(["pcs", "pc"], ("pcs", 1)),
    **dict.fromkeys(["w", "watt"], ("w", 1)),
}
_NUM_UNIT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)([a-z]+)")

_ESCAPE_RE = re.compile(r"\\x[0-9a-fA-F]{2}")
# Decimals with an optional unit ("1,8l") first, then any alphanumeric run so model numbers like "a52" stay whole.
_TOKEN_RE = re.compile(r"[0-9]+[.,][0-9]+[a-z]*|[a-z0-9]+")


def decode_title(title: str) -> str:
    """Undo the literal byte escapes Shopee titles contain, e.g. '\\xe2\\x80\\x9c' -> '“'."""
    if not isinstance(title, str):
        return ""
    if _ESCAPE_RE.search(title):
        try:
            title = codecs.decode(title, "unicode_escape").encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass  # leave it as-is rather than mangling it further
    return html.unescape(title)


def tokenize(title: str, canonical_units: bool = False) -> list[str]:
    """Lowercase word/number tokens with units glued onto their numbers.

    With canonical_units, unit spellings are unified too: "400 gram", "400 gr" and "0.4kg" -> "400g".
    """
    units = UNITS | CANONICAL_UNITS.keys() if canonical_units else UNITS
    raw = _TOKEN_RE.findall(decode_title(title).lower())
    tokens: list[str] = []
    for tok in raw:
        tok = tok.replace(",", ".")
        if tokens and tok in units and tokens[-1].replace(".", "").isdigit():
            tokens[-1] += tok
        else:
            tokens.append(tok)
    return [canonical_unit(t) for t in tokens] if canonical_units else tokens


def canonical_unit(token: str) -> str:
    m = _NUM_UNIT_RE.fullmatch(token)
    if not m or m.group(2) not in CANONICAL_UNITS:
        return token
    unit, factor = CANONICAL_UNITS[m.group(2)]
    value = round(float(m.group(1)) * factor, 3)
    return f"{int(value) if value.is_integer() else value}{unit}"
