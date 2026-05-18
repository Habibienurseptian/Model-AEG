import numpy as np

from scripts.config import (
    MIN_SCORE,
    MAX_SCORE
)


def ensemble_predict(
    model_bert,
    model_ling,
    X_test_bert,
    X_test_ling
):
    semantic = model_bert.predict(X_test_bert)
    syntax = model_ling.predict(X_test_ling)

    final_pred = (
        0.8 * semantic +
        0.2 * syntax
    )

    final_pred = np.clip(
        final_pred,
        MIN_SCORE,
        MAX_SCORE
    )

    return final_pred