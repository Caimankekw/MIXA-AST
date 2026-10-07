# stats_regression.py
import numpy as np
from scipy.stats import pearsonr, spearmanr


def calculate_regression_stats(pred, target):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - target))

    if len(np.unique(pred)) < 2 or len(np.unique(target)) < 2:
        plcc = 0.0
        srcc = 0.0
    else:
        plcc = pearsonr(pred, target)[0]
        srcc = spearmanr(pred, target)[0]

    return {
        'PLCC': float(plcc),
        'SRCC': float(srcc),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MSE': float(mse),
    }