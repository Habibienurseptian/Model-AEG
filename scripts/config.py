import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

MODEL_PATHS = {
    "bert_model": os.path.join(BASE_DIR, "model", "bert.pkl"),
    "ling_model": os.path.join(BASE_DIR, "model", "ling.pkl"),
}

# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "indobenchmark/indobert-base-p1"
# MODEL_NAME = "LazarusNLP/all-indo-e5-small-v4"

MAX_LENGTH = 128
BATCH_SIZE = 16

MIN_DATA = 100
MIN_SCORE, MAX_SCORE = 1, 5


device = torch.device("cpu")

for path in MODEL_PATHS.values():
    os.makedirs(os.path.dirname(path), exist_ok=True)