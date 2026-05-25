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
from scripts.metrics import evaluate


def ensemble_predict(model_bert, model_ling, X_bert, X_ling):
    return 0.8 * model_bert.predict(X_bert) + 0.2 * model_ling.predict(X_ling)


def train_model():

    print("\n=== TRAINING START ===")

    df = load_dataset()
    if df is None:
        return

    print(f"Total data: {len(df)}")

    questions = df["questions"].fillna("").astype(str).tolist()
    answers = df["answers"].fillna("").astype(str).tolist()
    labels = df["scores"].astype(float).values

    X_train_q, X_test_q, X_train_a, X_test_a, y_train, y_test = train_test_split(
        questions,
        answers,
        labels,
        test_size=0.2,
        random_state=42
    )

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # BERT
    print("\nExtracting BERT embeddings...")
    t0 = time.time()

    q_train_emb = get_bert_embeddings(X_train_q)
    a_train_emb = get_bert_embeddings(X_train_a)

    q_test_emb = get_bert_embeddings(X_test_q)
    a_test_emb = get_bert_embeddings(X_test_a)

    X_train_bert = np.abs(q_train_emb - a_train_emb)
    X_test_bert = np.abs(q_test_emb - a_test_emb)

    print(f"Done in {time.time() - t0:.2f}s")

    # LINGUISTIC FEATURES
    print("\nExtracting linguistic features...")
    t0 = time.time()

    X_train_ling = extract_linguistic_features(X_train_a)
    X_test_ling = extract_linguistic_features(X_test_a)

    print(f"Done in {time.time() - t0:.2f}s")

    # TRAIN MODEL
    print("\nTraining models...")
    t0 = time.time()

    model_bert = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    model_ling = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    model_bert.fit(X_train_bert, y_train)
    model_ling.fit(X_train_ling, y_train)

    print(f"Done in {time.time() - t0:.2f}s")

    # EVALUATION
    print("\nEvaluating...")

    final_pred = ensemble_predict(
        model_bert,
        model_ling,
        X_test_bert,
        X_test_ling
    )

    scores = evaluate(y_test, final_pred)

    print("\n=== EVALUATION ===")
    for k, v in scores.items():
        print(f"{k:<10}: {v:.4f}")

    # SAVE MODEL
    save_models(model_bert, model_ling)

    print("\n=== TRAINING DONE ===")