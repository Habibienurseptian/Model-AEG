import re

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'\b\d+\s*[\.\)]\s*', '', text)
    text = text.replace("/", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def word_tokenize(text):
    text = normalize_text(text)
    return [w for w in text.split() if w]