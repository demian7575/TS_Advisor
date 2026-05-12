from ts_advisor.text import clean_text, extract_observation_sections


def test_clean_text_handles_none_empty_and_normal_strings():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("UE traffic stops after HO 123!") == "ue traffic stops after ho"


def test_extract_observation_sections_returns_expected_outputs():
    text = """1 EFFECT
=====
Traffic stopped after handover.
2 TROUBLE DESCRIPTION
=====
2.2 Configuration Data
---
BB6630 21.Q3
2.3 Logs
3 MEASURES
=====
Restart workaround.
4 CSR
=====
CSR-123"""
    sections = extract_observation_sections(text)
    assert "Traffic stopped" in sections["effect"]
    assert "BB6630" in sections["config"]
    assert "Restart workaround" in sections["measures"]
