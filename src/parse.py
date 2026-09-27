"""Phase 2: extract a single answer letter (A-E) from a model reply.

Returns None when no single valid letter can be found; the caller counts that as wrong.
"""
import re

LETTERS = "ABCDE"

# Arabic MCQ labels sometimes used instead of Latin letters (أ ب ج د هـ).
ARABIC_TO_LATIN = {"أ": "A", "ا": "A", "ب": "B", "ج": "C", "د": "D", "ه": "E", "هـ": "E"}

# Explicit answer phrases: "Answer: B", "the answer is (C)", "الإجابة: د", "الجواب هو ب".
_ANSWER_PHRASE = re.compile(
    r"(?:answer(?:\s+is)?|الإجابة(?:\s+الصحيحة)?(?:\s+هي)?|الاجابة(?:\s+هي)?|الجواب(?:\s+هو)?)"
    r"\s*[:：\-]?\s*[\(\[]?\s*([A-Ea-eأابجده]ـ?)(?![A-Za-z؀-ۿ])",
    re.IGNORECASE,
)
# A standalone label: not glued to other letters (so "Bus" or "بيت" do not count).
_STANDALONE = re.compile(r"(?<![A-Za-z؀-ۿ])([A-Ea-eأابجده]ـ?)(?![A-Za-z؀-ۿ])")


def _to_latin(token: str) -> str:
    token = token.strip()
    return ARABIC_TO_LATIN.get(token, ARABIC_TO_LATIN.get(token.rstrip("ـ"), token.upper()))


def parse_letter(reply: str | None, n_options: int = 5) -> str | None:
    """Return 'A'..'E' (limited to the first n_options letters) or None."""
    if not reply:
        return None
    valid = set(LETTERS[:n_options])
    text = reply.strip()

    # 1) Explicit phrase such as "Answer: B" wins.
    m = _ANSWER_PHRASE.search(text)
    if m and _to_latin(m.group(1)) in valid:
        return _to_latin(m.group(1))

    # Distinct uppercase Latin labels anywhere in the reply. Two or more ("A or B")
    # means the model hedged, so we refuse to guess. (Lowercase / Arabic single
    # letters mid-sentence are too often ordinary words, so they are not counted.)
    found = {c for c in re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", text) if c in valid}
    if len(found) > 1:
        return None

    # 2) Reply starts with a label: "B", "B.", "(B)", "B) text", "ب - ...".
    m = _STANDALONE.match(text.lstrip("([*\"' "))
    if m and _to_latin(m.group(1)) in valid:
        return _to_latin(m.group(1))

    # 3) Otherwise accept a single Latin label found mid-reply.
    return found.pop() if found else None
