from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# =========================
# 1. Path configuration
# =========================
# File location: <repo_root>/routers/loage.py
# Resolve repo root and model directory paths deterministically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "age_model.pkl"      # Regression model for age prediction
QUANTILE_PATH = MODEL_DIR / "model.pkl"       # Quantile lookup table for percentile scoring

# Input feature order expected by the trained regression model
FEATURE_COLUMNS = [
    "flexibility",
    "grip_strength",
    "jump_power",
    "sex_M",
    "cardio_endurance",
    "sit_ups",
    "body_fat_pct",
    "sex_F",
    "rhr",
]

# Metrics used to compute percentile score (0-100)
METRICS = ["sit_ups", "flexibility", "jump_power", "cardio_endurance"]

# Metrics where a lower raw value indicates better performance (e.g., time-based measures)
LOWER_IS_BETTER = ["cardio_endurance"]

# =========================
# 2. Model & quantile loading
# =========================
try:
    age_model = joblib.load(MODEL_PATH)
    logger.info("age_model loaded successfully.")
except Exception as e:
    logger.warning("Failed to load age_model: %s", e)
    age_model = None

try:
    quantile_table = joblib.load(QUANTILE_PATH)
    logger.info("quantile_table loaded successfully.")
except Exception as e:
    logger.warning("Failed to load quantile_table: %s", e)
    quantile_table = None

# =========================
# 3. Request schema
# =========================
class PhysicalAgeRequest(BaseModel):
    flexibility: float
    jump_power: float
    cardio_endurance: float
    sit_ups: float

    sex: str  # "M" or "F"

    grip_strength: float | None = None
    body_fat_pct: float | None = None
    rhr: float | None = None

# =========================
# 4. Default values & missing feature handling
# =========================
# Sex-specific default values used when optional inputs are missing
DEFAULTS = {
    "M": {"grip_strength": 30.0, "body_fat_pct": 20.0, "rhr": 75.0},
    "F": {"grip_strength": 20.0, "body_fat_pct": 27.0, "rhr": 78.0},
}

def fill_missing_features(data: PhysicalAgeRequest) -> dict:
    """
    Fills missing optional features with sex-specific defaults
    to construct a complete model input dictionary.
    """
    sex = data.sex.upper()

    base = DEFAULTS.get(sex)
    if base is None:
        raise HTTPException(status_code=400, detail="sex must be either 'M' or 'F'.")

    return {
        "flexibility": data.flexibility,
        "grip_strength": data.grip_strength if data.grip_strength is not None else base["grip_strength"],
        "jump_power": data.jump_power,
        "cardio_endurance": data.cardio_endurance,
        "sit_ups": data.sit_ups,
        "body_fat_pct": data.body_fat_pct if data.body_fat_pct is not None else base["body_fat_pct"],
        "rhr": data.rhr if data.rhr is not None else base["rhr"],
        "sex_M": 1 if sex == "M" else 0,
        "sex_F": 1 if sex == "F" else 0,
    }

# =========================
# 5. Percentile & adjustment utilities
# =========================
def _is_true(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return x == 1
    if isinstance(x, str):
        return x.upper() == "TRUE"
    return False

def get_sex_from_row(row: pd.Series) -> str:
    """
    Maps one-hot sex columns to the quantile table sex key.
    Expected keys: 'Female' or 'Male'
    """
    if _is_true(row.get("sex_F", None)):
        return "Female"
    if _is_true(row.get("sex_M", None)):
        return "Male"
    return "Male"

def get_percentile_score(metric: str, value: float, sex: str, q_model: dict) -> float:
    """
    Converts a raw metric value into a 0-100 percentile score using a quantile table.
    """
    try:
        series = q_model[metric][sex]  # index: 0..100, values: thresholds
        idx = int(np.searchsorted(series.values, value, side="right"))
        score = float(idx)

        if metric in LOWER_IS_BETTER:
            score = 100.0 - score

        return max(0.0, min(100.0, score))
    except Exception:
        return 0.0

def get_percentile(user_row: pd.Series, quantile_model: dict) -> float:
    """
    Computes the final percentile as the mean of per-metric percentile scores.
    """
    sex = get_sex_from_row(user_row)
    scores: list[float] = []

    for metric in METRICS:
        if metric not in user_row.index:
            continue
        value = user_row[metric]
        if pd.isna(value):
            continue
        scores.append(get_percentile_score(metric, float(value), sex, quantile_model))

    return float(np.mean(scores)) if scores else 50.0

def adjust_age(age_pred: float, percentile: float) -> float:
    """
    Adjusts predicted age using a heuristic based on percentile bands.
    """
    if percentile < 20:
        return age_pred + 5
    if percentile < 40:
        return age_pred + 2
    if percentile < 60:
        return age_pred
    if percentile < 80:
        return age_pred - 2
    return age_pred - 4

# =========================
# 6. Main endpoint
# =========================
@router.post("/compute")
def compute_physical_age(payload: PhysicalAgeRequest):
    # Fail fast if required models are not loaded
    if age_model is None or quantile_table is None:
        raise HTTPException(status_code=500, detail="Required model artifacts are not loaded.")

    sex = payload.sex.upper()
    if sex not in ("M", "F"):
        raise HTTPException(status_code=400, detail="sex must be either 'M' or 'F'.")

    # Prepare model inputs (fill missing optional fields)
    row_dict = fill_missing_features(payload)
    row = pd.Series(row_dict)

    # Stage A: Predict age using regression model
    X = pd.DataFrame([row_dict])[FEATURE_COLUMNS]
    age_pred = float(age_model.predict(X)[0])

    # Stage B: Compute percentile using quantile table
    percentile = float(get_percentile(row, quantile_table))

    # Stage C: Adjust physical age based on percentile
    physical_age = float(adjust_age(age_pred, percentile))

    return {
        "age_pred": age_pred,
        "percentile": percentile,
        "physical_age": physical_age,
    }
