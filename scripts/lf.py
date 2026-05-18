import re
import numpy as np

from scripts.preprocessing import word_tokenize

def extract_linguistic_features(texts):
    features = []
    for t in texts:
        try:
            t = t.strip()
            words = word_tokenize(t)
            sentences = re.split(r'[.!?]+', t)
            sentences = [s for s in sentences if s.strip()]
            num_words = len(words)
            num_sentences = max(len(sentences), 1)
            avg_sentence_len = num_words / num_sentences
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            unique_words = len(set(words))
            ttr = unique_words / (num_words + 1e-5)
            repetition = 1 - ttr
            long_word_ratio = sum(1 for w in words if len(w) > 6) / (num_words + 1e-5)
            punct_count = len(re.findall(r'[^\w\s]', t))
            punct_ratio = punct_count / (len(t) + 1e-5)
            f = [
                num_words,
                num_sentences,
                avg_sentence_len,
                avg_word_len,
                ttr,
                repetition,
                long_word_ratio,
                punct_ratio
            ]
        except Exception:
            f = [0] * 8
        features.append(f)
    return np.array(features)