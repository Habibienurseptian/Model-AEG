import time
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.bert import get_bert_embeddings
from scripts.lf import extract_linguistic_features

from scripts.data_loader import load_dataset
from scripts.model_loader import save_models
from scripts.evaluation import ensemble_predict
from scripts.metrics import evaluate


def train_model():
    print("\n=== TRAINING START ===")

    df = load_dataset()

    if df is None:
        return

    print(f"total baris: {len(df)}")

    questions = df["questions"].astype(str).values
    answers = df["answers"].astype(str).values
    labels = df["scores"].astype(float).values

    

    X_train_q, X_test_q, X_train_a, X_test_a, y_train, y_test = train_test_split(
        questions,
        answers,
        labels,
        test_size=0.2,
        random_state=42
    )

    print(f"train: {len(y_train)}, test: {len(y_test)}")

    # BERT
    print("\n extracting BERT embeddings...")
    t0 = time.time()
    X_train_bert = get_bert_embeddings(X_train_q, X_train_a)
    X_test_bert = get_bert_embeddings(X_test_q, X_test_a)
    print(f"  selesai dalam {time.time() - t0:.2f}s" )

    # Linguistic
    print("\n extracting linguistic features...")
    t0 = time.time()
    X_train_ling = extract_linguistic_features(X_train_a)
    X_test_ling = extract_linguistic_features(X_test_a)
    print(f"  selesai dalam {time.time() - t0:.2f}s")

    #Ridge
    print("\n training models...")
    t0 = time.time()
    model_bert = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    model_bert.fit(X_train_bert, y_train)
    model_ling = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    model_ling.fit(X_train_ling, y_train)
    print(f"  selesai dalam {time.time() - t0:.2f}s")

    #Fusion
    final_pred = ensemble_predict(
        model_bert,
        model_ling,
        X_test_bert,
        X_test_ling
    )
    scores = evaluate(y_test, final_pred)
    
    print("\n=== EVALUATION ===")
    for key, value in scores.items():
        print(f"  {key:<5}: {value:.4f}")

    save_models(model_bert, model_ling)
    print("\n=== TRAINING DONE ===")