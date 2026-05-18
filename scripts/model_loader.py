import os
import joblib

from scripts.config import MODEL_PATHS


def load_models():
    bert = None
    ling = None

    bert_path = MODEL_PATHS["bert_model"]
    ling_path = MODEL_PATHS["ling_model"]

    if os.path.exists(bert_path):
        bert = joblib.load(bert_path)
        print("BERT model loaded")

    if os.path.exists(ling_path):
        ling = joblib.load(ling_path)
        print("Linguistic model loaded")

    return bert, ling


def save_models(model_bert, model_ling):
    os.makedirs(
        os.path.dirname(MODEL_PATHS["bert_model"]),
        exist_ok=True
    )

    joblib.dump(model_bert, MODEL_PATHS["bert_model"])
    joblib.dump(model_ling, MODEL_PATHS["ling_model"])

    print("model berhasil disimpan")