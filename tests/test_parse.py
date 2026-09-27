import pytest

from src.parse import parse_letter


@pytest.mark.parametrize("reply, expected", [
    ("B", "B"),
    ("  c  ", "C"),
    ("B.", "B"),
    ("(D)", "D"),
    ("A) الأولي", "A"),
    ("Answer: B", "B"),
    ("The answer is (C).", "C"),
    ("**D**", "D"),
    ("الإجابة: B", "B"),
    ("الإجابة الصحيحة هي د", "D"),
    ("الجواب هو ب", "B"),
    ("ج", "C"),
    ("أ", "A"),
    ("هـ", "E"),
    ("الخيار C هو الصحيح", "C"),
])
def test_valid_replies(reply, expected):
    assert parse_letter(reply) == expected


@pytest.mark.parametrize("reply", [
    "",
    None,
    "لا أعرف",                 # "I don't know"
    "A or B",                  # two different letters -> ambiguous
    "Bus",                     # letter glued inside a word
    "The correct option is",   # cut off before the letter
])
def test_invalid_replies_return_none(reply):
    assert parse_letter(reply) is None


def test_letter_outside_number_of_options_is_rejected():
    assert parse_letter("D", n_options=3) is None
    assert parse_letter("C", n_options=3) == "C"
