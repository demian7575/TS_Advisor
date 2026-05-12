import re
import pandas as pd

SECTION_KEYS = ["effect", "trouble", "config", "measures", "csr"]

def clean_text(text):
    """Lowercase, keep only letters and whitespace, collapse runs of whitespace."""
    if text is None or pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    return " ".join(text.lower().split())


def clean_text_keep_alphanum(text):
    """Notebook-compatible cleaner that preserves digits and dots for configuration tokens."""
    if text is None or pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z0-9.\s]", " ", str(text))
    return " ".join(text.lower().split())


def extract_effect_section(text):
    """Extract the notebook's '1 EFFECT' section; fall back to full text if structure is absent."""
    if text is None or pd.isna(text) or text == "":
        return ""
    pat = r"1\s+E\s*F\s*F\s*E\s*C\s*T\s*.*?\n={5,}(.*?)(?=2\s+T\s*R\s*O\s*U\s*B\s*L\s*E)"
    m = re.search(pat, str(text), re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else str(text)


def extract_observation_sections(text):
    """Split Observation.Observation into the same major sections used by the source notebook."""
    if text is None or pd.isna(text) or text == "":
        return {k: "" for k in SECTION_KEYS}
    t = str(text)
    def grab(pattern):
        m = re.search(pattern, t, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
    return {
        "effect": grab(r"1\s+E\s*F\s*F\s*E\s*C\s*T.*?\n={3,}(.*?)(?=\n\s*2\s+T\s*R\s*O\s*U\s*B\s*L\s*E|$)"),
        "trouble": grab(r"2\s+T\s*R\s*O\s*U\s*B\s*L\s*E.*?\n={3,}(.*?)(?=\n\s*3\s+M\s*E\s*A\s*S\s*U\s*R\s*E|$)"),
        "config": grab(r"2\.2\s+Configuration\s+Data.*?\n-{3,}(.*?)(?=\n\s*2\.3|\n\s*3\s+M\s*E\s*A\s*S|$)"),
        "measures": grab(r"3\s+M\s*E\s*A\s*S\s*U\s*R\s*E.*?\n={3,}(.*?)(?=\n\s*4|$)"),
        "csr": grab(r"4\s+CSR.*?\n={3,}(.*?)(?=$)"),
    }


def extract_cause_of_fault(text):
    """Extract 'CAUSE OF FAULT' from Answer.Answer; fall back to full answer."""
    if text is None or pd.isna(text) or text == "":
        return ""
    m = re.search(r"CAUSE OF FAULT\s*\n-{10,}(.*?)(?=\n-{10,})", str(text), re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else str(text)


def build_text_column(dataframe, config, use_answer=False):
    heading = dataframe.get(config.heading_column, pd.Series([""] * len(dataframe), index=dataframe.index)).fillna("")
    effect = dataframe.get(config.observation_column, pd.Series([""] * len(dataframe), index=dataframe.index)).apply(extract_effect_section)
    if use_answer:
        cause = dataframe.get(config.answer_column, pd.Series([""] * len(dataframe), index=dataframe.index)).apply(extract_cause_of_fault)
        combined = heading + " [SEP] " + effect + " [SEP] " + cause
    else:
        combined = heading + " [SEP] " + effect
    return combined.apply(clean_text).tolist()
