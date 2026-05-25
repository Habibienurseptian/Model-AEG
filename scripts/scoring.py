import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scripts.preprocessing import normalize_text
from scripts.bert import get_bert_embeddings
from scripts.lf import extract_linguistic_features


class BertSemanticSearch:
    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_embeddings = get_bert_embeddings(corpus)

    def search(self, reference, top_k=5):
        reference_emb = get_bert_embeddings(
            [reference].reshape(1, -1),
        )[0]

        scores = cosine_similarity(
            reference_emb,
            self.corpus_embeddings
        )[0]

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "text": self.corpus[i],
                "score": float(scores[i] * 100)
            }
            for i in top_idx
        ]


def compute_score(answer, reference, bert_model, ling_model):

    if not answer or normalize_text(answer) == "":
        return {
            "semantic_score": 0.0,
            "syntactic_score": 0.0,
            "bert_score": 0.0,
            "ling_score": 0.0,
            "final_score": 0.0,
        }

    # BERT Embedding
    reference_emb = get_bert_embeddings([reference])
    answer_emb = get_bert_embeddings([answer])

    semantic_score = cosine_similarity(
        reference_emb.reshape(1, -1),
        answer_emb.reshape(1, -1)
    )[0][0] * 100

    # Linguistic features
    ling_features = extract_linguistic_features([answer])
    syntactic_score = (np.tanh(np.mean(ling_features)) * 50) + 50

    # ML models
    answer_feature = answer_emb.reshape(1, -1)
    bert_score = bert_model.predict(answer_feature)
    ling_score = ling_model.predict(ling_features)

    # final score
    final_score = 0.8 * semantic_score + 0.2 * syntactic_score

    return {
        "bert_score": round(float(bert_score[0]), 2),
        "ling_score": round(float(ling_score[0]), 2),
        "semantic_score": round(float(semantic_score), 2),
        "syntactic_score": round(float(syntactic_score), 2),
        "final_score": round(float(np.clip(final_score, 0, 100)), 2),
    }