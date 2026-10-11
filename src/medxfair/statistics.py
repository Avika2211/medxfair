import numpy as np
from scipy.stats import norm

def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")

def wilson_interval(successes: int, total: int, confidence: float = 0.95):
    if total == 0: return (float("nan"), float("nan"))
    z = norm.ppf((1 + confidence) / 2)
    p = successes / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    radius = z / den * np.sqrt(p*(1-p)/total + z*z/(4*total*total))
    return float(max(0, center-radius)), float(min(1, center+radius) if successes != total else 1.0)
