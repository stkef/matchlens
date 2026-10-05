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


def tokenize(title: str) -> list[str]:
    """Lowercase word/number tokens with units glued onto their numbers."""
    raw = _TOKEN_RE.findall(decode_title(title).lower())
    tokens: list[str] = []
    for tok in raw:
        tok = tok.replace(",", ".")
        if tokens and tok in UNITS and tokens[-1].replace(".", "").isdigit():
            tokens[-1] += tok
        else:
            tokens.append(tok)
    return tokens
