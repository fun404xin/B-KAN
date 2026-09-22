import numpy as np

def compute_rmse(y_true, y_pred):
    """
    Root Mean Square Error

    """
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)
    
def compute_mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))

def compute_r2(y_true, y_pred):
    """
    R-squared (Coefficient of Determination) 
    """
    ss_res = np.sum((y_true - y_pred) ** 2)    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) 
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot)

from scipy.special import ndtr


def compute_crps(y_true, y_pred, y_std):
    """
    Continuous Ranked Probability Score (Gaussian)
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_std = np.asarray(y_std, dtype=float).reshape(-1)

    if not (y_true.shape == y_pred.shape == y_std.shape):
        raise ValueError("y_true、y_pred 和 y_std 的样本数必须一致。")

    if y_true.size == 0:
        raise ValueError("输入不能为空。")

    if not all(np.all(np.isfinite(x))
               for x in (y_true, y_pred, y_std)):
        raise ValueError("输入不能包含 NaN 或无穷值。")

    if np.any(y_std < 0):
        raise ValueError("预测标准差不能为负数。")

    crps = np.abs(y_true - y_pred)
    mask = y_std > 0

    sigma = y_std[mask]
    z = (y_true[mask] - y_pred[mask]) / sigma

    cdf = ndtr(z)
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)

    crps[mask] = sigma * (
        z * (2.0 * cdf - 1.0)
        + 2.0 * pdf
        - 1.0 / np.sqrt(np.pi)
    )

    return float(np.mean(crps))

def compute_picp(y_true, y_pred, y_std, z_score=1.96):
   
    lower_bound = y_pred - z_score * y_std
    upper_bound = y_pred + z_score * y_std
    
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    return np.mean(in_interval)

def compute_mpiw(y_pred, y_std, z_score=1.96):
  
    lower_bound = y_pred - z_score * y_std
    upper_bound = y_pred + z_score * y_std
    
    width = upper_bound - lower_bound
    return np.mean(width)

def get_all_metrics(y_true, y_pred, y_std=None):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的样本数必须一致。")
    metrics = {
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "R2": compute_r2(y_true, y_pred)
    }

    if y_std is not None:
        y_std = np.asarray(y_std, dtype=float).reshape(-1)
        metrics["CRPS"] = compute_crps(y_true, y_pred, y_std)
        metrics["PICP"] = compute_picp(y_true, y_pred, y_std)
        metrics["MPIW"] = compute_mpiw(y_pred, y_std)

    return metrics