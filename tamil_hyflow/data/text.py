import re
import unicodedata
from dataclasses import dataclass

@dataclass
class TamilToken:
    text: str
    cons_id: int
    vowel_id: int
    length_id: int
    class_id: int
    word_bound_id: int
    punct_id: int

_PUNCT = {"": 0, ",": 1, ".": 2, "?": 3}
_TAMIL_VOWELS = {"அ": 0, "ஆ": 1, "இ": 2, "ஈ": 3, "உ": 4, "ஊ": 5, "எ": 6, "ஏ": 7, "ஐ": 8, "ஒ": 9, "ஓ": 10, "ஔ": 11}
_MODIFIERS = {"": 0, "ா": 1, "ி": 2, "ீ": 3, "ு": 4, "ூ": 5, "ெ": 6, "ே": 7, "ை": 8, "ொ": 9, "ோ": 10, "ௌ": 11, "்": 12}
_BASE = {"அ": 0, "ஆ": 0, "இ": 0, "ஈ": 0, "உ": 0, "ஊ": 0, "எ": 0, "ஏ": 0, "ஐ": 0, "ஒ": 0, "ஓ": 0, "ஔ": 0, "ஃ": 18}
_CONS_BASES = ["க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம", "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன"]
_BASE.update({c: i + 1 for i, c in enumerate(_CONS_BASES)})
_CLASS_MAP = {c: (i % 6) for i, c in enumerate(_CONS_BASES)}
_LENGTH_MAP = {"அ": 0, "ஆ": 1, "இ": 0, "ஈ": 1, "உ": 0, "ஊ": 1, "எ": 0, "ஏ": 1, "ஐ": 2, "ஒ": 0, "ஓ": 1, "ஔ": 2}
_MOD_LENGTH = {"": 0, "ா": 1, "ி": 0, "ீ": 1, "ு": 0, "ூ": 1, "ெ": 0, "ே": 1, "ை": 2, "ொ": 0, "ோ": 1, "ௌ": 2, "்": 0}

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text

def _split_units(text: str) -> list[tuple[str, str, str, str]]:
    units: list[tuple[str, str, str, str]] = []
    word_tokens = text.split(" ")
    for wi, word in enumerate(word_tokens):
        chars = list(word)
        i = 0
        while i < len(chars):
            ch = chars[i]
            if ch in _PUNCT:
                units.append((ch, ch, "", "punct"))
                i += 1
                continue
            if ch in _BASE and ch not in _TAMIL_VOWELS:
                modifier = ""
                if i + 1 < len(chars) and chars[i + 1] in _MODIFIERS:
                    modifier = chars[i + 1]
                    i += 1
                units.append((ch + modifier, ch, modifier, "tamil"))
            elif ch in _TAMIL_VOWELS:
                units.append((ch, ch, "", "tamil"))
            else:
                units.append((ch, ch, "", "other"))
            i += 1
        if wi < len(word_tokens) - 1:
            units.append((" ", "", "", "boundary"))
    return units

def tokenize_tamil(text: str) -> list[TamilToken]:
    text = normalize_text(text)
    out: list[TamilToken] = []
    units = _split_units(text)
    for idx, (surface, base, modifier, kind) in enumerate(units):
        if kind == "punct":
            out.append(TamilToken(surface, 0, 0, 0, 0, 1, _PUNCT.get(surface, 0)))
            continue
        if kind == "boundary":
            continue
        cons_id = _BASE.get(base, 19)
        vowel_id = _MODIFIERS.get(modifier, 0)
        length_id = _LENGTH_MAP.get(base, _MOD_LENGTH.get(modifier, 0))
        class_id = _CLASS_MAP.get(base, 5)
        word_bound = 1 if idx + 1 < len(units) and units[idx + 1][3] == "boundary" else 0
        punct_id = 0
        out.append(TamilToken(surface, cons_id, vowel_id, length_id, class_id, word_bound, punct_id))
    return out
