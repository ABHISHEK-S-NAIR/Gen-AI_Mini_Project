from app.services.section_detector import detect_sections


def test_detect_sections_with_synonyms() -> None:
    raw = """
    Abstract
    This is abstract text.
    Introduction
    Intro details.
    Method
    Method details.
    Experiments
    Result details.
    Conclusion
    Final notes.
    """

    sections = detect_sections(raw)
    assert sections["abstract"]
    assert sections["intro"]
    assert sections["method"]
    assert sections["results"]
    assert sections["conclusion"]
