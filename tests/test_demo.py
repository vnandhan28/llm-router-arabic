from src.demo import parse_question


def test_parse_options_on_separate_lines():
    q, opts = parse_question("ما عاصمة الإمارات؟\nA. دبي\nB. أبوظبي\nC. الشارقة")
    assert q == "ما عاصمة الإمارات؟"
    assert opts == ["دبي", "أبوظبي", "الشارقة"]


def test_parse_arabic_option_labels():
    q, opts = parse_question("كم عدد سور القرآن؟\nأ) 113\nب) 114")
    assert q == "كم عدد سور القرآن؟"
    assert opts == ["113", "114"]


def test_parse_options_on_one_line():
    q, opts = parse_question("ما عاصمة الإمارات؟ A) دبي B) أبوظبي C) الشارقة")
    assert q == "ما عاصمة الإمارات؟"
    assert opts == ["دبي", "أبوظبي", "الشارقة"]
