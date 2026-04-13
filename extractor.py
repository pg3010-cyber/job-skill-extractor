"""Core NLP logic for extracting skills from job descriptions."""

import re
from collections import Counter

import spacy

from skills_data import SKILLS_LIST, SYNONYMS

nlp = spacy.load("en_core_web_sm")


def preprocess(text):
    """Lowercase text and remove most punctuation for matching."""
    lowered_text = text.lower()
    # Keep a few tech-friendly symbols so terms like C++, CI/CD, and Node.js survive.
    cleaned_text = re.sub(r"[^a-z0-9\s\+\#\./-]", " ", lowered_text)
    return re.sub(r"\s+", " ", cleaned_text).strip()


def get_pos_tags(text):
    """Return noun-like tokens because technical skills are often nouns."""
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc if token.pos_ in {"NOUN", "PROPN"}]


def dependency_parse(text):
    """Return simple dependency relationships linked to likely skill phrases."""
    doc = nlp(text)
    dep_rows = []
    for token in doc:
        if token.dep_ in {"pobj", "dobj", "compound"}:
            dep_rows.append((token.text, token.dep_, token.head.text))
    return dep_rows


def extract_skills(text):
    """Extract skills with simple matching, synonym resolution, and dependency hints."""
    cleaned_text = preprocess(text)
    tokens = cleaned_text.split()
    candidates = tokens + [" ".join(pair) for pair in zip(tokens, tokens[1:])]
    skill_lookup = {skill.lower(): skill for skill in SKILLS_LIST}

    matched_skills = []
    for candidate in candidates:
        canonical_candidate = SYNONYMS.get(candidate.lower(), candidate)
        if canonical_candidate.lower() in skill_lookup:
            matched_skills.append(skill_lookup[canonical_candidate.lower()])

    dep_info = dependency_parse(text)
    for token_text, dep_label, head_word in dep_info:
        dep_candidate = SYNONYMS.get(token_text.lower(), token_text)
        if dep_candidate.lower() in skill_lookup:
            matched_skills.append(skill_lookup[dep_candidate.lower()])

        # Compound relations often split multi-word skills such as "Machine Learning".
        if dep_label == "compound":
            compound_phrase = f"{token_text} {head_word}".lower()
            canonical_phrase = SYNONYMS.get(compound_phrase, compound_phrase)
            if canonical_phrase.lower() in skill_lookup:
                matched_skills.append(skill_lookup[canonical_phrase.lower()])

    frequency = Counter(matched_skills)
    unique_skills = sorted(frequency.keys())

    return {
        "skills": unique_skills,
        "frequency": dict(sorted(frequency.items(), key=lambda item: (-item[1], item[0]))),
        "pos_tags": get_pos_tags(text),
        "dep_info": dep_info,
    }
