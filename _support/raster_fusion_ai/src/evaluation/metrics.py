"""Masked regression metrics and local SSIM with complete valid windows."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import uniform_filter, minimum_filter


def masked_ssim(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, window: int = 7) -> tuple[float | None, int]:
    if window < 3 or window % 2 == 0:
        raise ValueError("SSIM window must be odd and >=3")
    valid = mask & np.isfinite(target) & np.isfinite(prediction)
    centers = minimum_filter(valid.astype("uint8"), size=window, mode="constant", cval=0).astype(bool)
    if not centers.any():
        return None, 0
    a, b = np.where(valid, target, 0).astype(float), np.where(valid, prediction, 0).astype(float)
    mu_a, mu_b = uniform_filter(a, window), uniform_filter(b, window)
    var_a = np.maximum(0, uniform_filter(a*a, window)-mu_a**2)
    var_b = np.maximum(0, uniform_filter(b*b, window)-mu_b**2)
    cov = uniform_filter(a*b, window)-mu_a*mu_b
    score = ((2*mu_a*mu_b+.01**2)*(2*cov+.03**2))/((mu_a**2+mu_b**2+.01**2)*(var_a+var_b+.03**2))
    return float(score[centers].mean()), int(centers.sum())


def regression_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, ssim: bool = False, window: int = 7) -> dict:
    valid = mask & np.isfinite(target) & np.isfinite(prediction)
    a, b = target[valid].astype(float), prediction[valid].astype(float)
    if not len(a):
        raise ValueError("No valid evaluation pixels")
    error = a-b
    mse = float(np.mean(error**2))
    varying_target = len(a) > 1 and np.ptp(a) > 0
    varying_both = varying_target and np.ptp(b) > 0
    result = dict(n=len(a), mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(abs(error))),
                  r2=float(1-np.sum(error**2)/np.sum((a-a.mean())**2)) if varying_target else None,
                  pearson=float(pearsonr(a, b).statistic) if varying_both else None,
                  spearman=float(spearmanr(a, b).statistic) if varying_both else None)
    result["ssim"], result["ssim_n_windows"] = masked_ssim(target, prediction, valid, window) if ssim else (None, 0)
    return result
