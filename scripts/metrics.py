import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    cohen_kappa_score
)

from scripts.config import (
    MIN_SCORE,
    MAX_SCORE
)

def qwk(y_true, y_pred):
    y_true = np.clip(
        np.round(y_true),
        MIN_SCORE,
        MAX_SCORE
    ).astype(int)

    y_pred = np.clip(
        np.round(y_pred),
        MIN_SCORE,
        MAX_SCORE
    ).astype(int)

    return cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic"
    )


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pcc = pearsonr(y_true, y_pred)[0]
    kappa = qwk(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "PCC": pcc,
        "QWK": kappa
    }