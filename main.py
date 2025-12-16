from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# =========================
# Environment & logging
# =========================

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

ENGINE_PATH = BASE_DIR / "models" / "model.pkl"

# Cached data (in-memory)
_facilities_df: Optional[pd.DataFrame] = None
_engine_cache: Optional[Dict[str, pd.DataFrame]] = None

# =========================
# Domain constants
# =========================

AGE_GRADES = [
    "19세 이하",
    "20대 초반",
    "20대 중반",
    "20대 후반",
    "30대 초반",
    "30대 중반",
    "30대 후반",
    "40대 초반",
    "40대 중반",
    "40대 후반",
    "50대 초반",
    "50대 중반",
    "50대 후반",
    "60대 초반",
    "60대 중반",
    "60대 후반",
    "70대 이상",
]

FACILITIES_SELECT = (
    "id,name,lat,lon,address,detail_equip,type,"
    "is_muscular_endurance,is_flexibility,is_cardio,quickness"
)

# =========================
# Supabase helpers
# =========================

def _require_env(value: Optional[str], name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _sb_json_headers(prefer_return: bool = False) -> Dict[str, str]:
    key = _require_env(SUPABASE_SERVICE_ROLE_KEY, "SUPABASE_SERVICE_ROLE_KEY")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer_return:
        headers["Prefer"] = "return=representation"
    return headers


def _sb_table_url(table: str) -> str:
    base = _require_env(SUPABASE_URL, "SUPABASE_URL")
    return f"{base}/rest/v1/{table}"


# =========================
# Loaders (engine / facilities)
# =========================

def load_engine() -> Dict[str, pd.DataFrame]:
    global _engine_cache

    if _engine_cache is not None:
        return _engine_cache

    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Engine file not found: {ENGINE_PATH}")

    obj = joblib.load(str(ENGINE_PATH))
    if not isinstance(obj, dict):
        raise ValueError("Engine artifact must be a dict of quantile tables.")

    _engine_cache = obj
    return _engine_cache


def load_facilities() -> pd.DataFrame:
    """
    Loads facility data from Supabase REST API into an in-memory DataFrame.
    Uses pagination (Range header) to fetch all rows.
    """
    global _facilities_df

    if _facilities_df is not None:
        return _facilities_df

    table_name = os.getenv("FACILITIES_TABLE", "facilities")
    url = _sb_table_url(table_name)

    key = _require_env(SUPABASE_SERVICE_ROLE_KEY, "SUPABASE_SERVICE_ROLE_KEY")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Range-Unit": "items",
        "Prefer": "count=exact",
    }

    page_size = 1000
    start = 0
    frames: List[pd.DataFrame] = []

    while True:
        end = start + page_size - 1
        headers["Range"] = f"{start}-{end}"
        params = {"select": FACILITIES_SELECT}

        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        frames.append(pd.DataFrame(data))

        if len(data) < page_size:
            break

        start += page_size

    if not frames:
        _facilities_df = pd.DataFrame(
            columns=[
                "id",
                "name",
                "lat",
                "lon",
                "address",
                "detail_equip",
                "is_cardio",
                "is_muscular_endurance",
                "is_flexibility",
                "quickness",
            ]
        )
        return _facilities_df

    df = pd.concat(frames, ignore_index=True).dropna(subset=["lat", "lon"])
    _facilities_df = df.astype({"lat": float, "lon": float})
    return _facilities_df


# =========================
# Utilities
# =========================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p = math.pi / 180.0
    a = (
        0.5
        - math.cos((lat2 - lat1) * p) / 2.0
        + math.cos(lat1 * p)
        * math.cos(lat2 * p)
        * (1.0 - math.cos((lon2 - lon1) * p))
        / 2.0
    )
    return r * 2.0 * math.asin(math.sqrt(a))


def infer_category(row: pd.Series) -> str:
    """
    Infers facility category based on boolean flags from Supabase.
    """
    categories: List[str] = []

    def _is_one(v) -> bool:
        return v == 1 or v == 1.0

    if _is_one(row.get("is_cardio")):
        categories.append("심폐지구력")
    if _is_one(row.get("is_muscular_endurance")):
        categories.append("근지구력")
    if _is_one(row.get("is_flexibility")):
        categories.append("유연성")
    if _is_one(row.get("quickness")):
        categories.append("순발력")

    return ", ".join(categories) if categories else "기타"


def get_quantile_from_table(df: pd.DataFrame, sex_col: str, value: float) -> float:
    """
    Interpolates quantile (0..1) for a given value using a quantile table.
    """
    if sex_col not in df.columns:
        raise KeyError(f"Missing column '{sex_col}' in engine table.")

    q = df.index.to_numpy(dtype=float)
    v = df[sex_col].to_numpy(dtype=float)

    order = np.argsort(v)
    return float(np.interp(value, v[order], q[order], left=0.0, right=1.0))


def grade_index_to_lo_age_value(idx: int) -> int:
    idx = max(0, min(idx, len(AGE_GRADES) - 1))
    mapping = {
        0: 18,
        1: 22,
        2: 25,
        3: 28,
        4: 32,
        5: 35,
        6: 38,
        7: 42,
        8: 45,
        9: 48,
        10: 52,
        11: 55,
        12: 58,
        13: 62,
        14: 65,
        15: 68,
        16: 72,
    }
    return mapping.get(idx, 40)


# =========================
# Pydantic models
# =========================

class PhysicalAgeRequest(BaseModel):
    user_id: Optional[str] = None
    sex: str
    flexibility: float
    jump_power: float
    cardio_endurance: float
    sit_ups: float

    @field_validator("sex")
    def normalize_sex(cls, v: str) -> str:
        s = v.strip().lower()
        if s in ("f", "female", "여"):
            return "Female"
        if s in ("m", "male", "남"):
            return "Male"
        raise ValueError("sex must be provided as Male/Female (M/F).")


class PhysicalAgeResponse(BaseModel):
    lo_age_value: int
    lo_age_tier_label: str
    percentile: float
    weak_point: str
    tier_index: int

    detail_quantiles: Dict[str, float]
    avg_quantile: float
    grade_index: int
    grade_label: str
    assessment_id: Optional[int] = None


class PhysicalAgeRecord(BaseModel):
    id: int
    user_id: str
    measured_at: datetime

    grade_index: Optional[int] = None
    grade_label: Optional[str] = None
    percentile: Optional[float] = None
    weak_point: Optional[str] = None
    avg_quantile: Optional[float] = None
    lo_age_value: Optional[int] = None
    lo_age_tier_label: Optional[str] = None
    detail_quantiles: Optional[Dict[str, float]] = None


class PhysicalAgeHistoryResponse(BaseModel):
    user_id: str
    records: List[PhysicalAgeRecord]


class FacilityOut(BaseModel):
    id: int
    name: str
    lat: float
    lon: float
    address: str
    mission: str
    category: str


class FavoriteToggleRequest(BaseModel):
    user_id: str
    facility_id: int
    is_favorite: bool


class MissionCompleteRequest(BaseModel):
    user_id: str
    facility_id: int
    status: str
    is_favorite: bool


# =========================
# FastAPI app
# =========================

app = FastAPI(title="Lo-Age API")


@app.on_event("startup")
def on_startup() -> None:
    try:
        load_engine()
        logger.info("Engine loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load engine: %s", e)

    try:
        load_facilities()
        logger.info("Facilities loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load facilities: %s", e)


@app.post("/predict/physical-age", response_model=PhysicalAgeResponse)
def predict_physical_age(req: PhysicalAgeRequest) -> PhysicalAgeResponse:
    engine = load_engine()

    # 1) Compute per-metric quantiles (0..1)
    q_dict = {
        "sit_ups": get_quantile_from_table(engine["sit_ups"], req.sex, req.sit_ups),
        "flexibility": get_quantile_from_table(engine["flexibility"], req.sex, req.flexibility),
        "jump_power": get_quantile_from_table(engine["jump_power"], req.sex, req.jump_power),
        "cardio_endurance": get_quantile_from_table(engine["cardio_endurance"], req.sex, req.cardio_endurance),
    }

    # Cardio endurance is treated as "lower is better" (time-like measure), so invert.
    q_dict["cardio_endurance"] = 1.0 - q_dict["cardio_endurance"]

    # 2) Average quantile -> grade index
    avg_q = float(np.mean(list(q_dict.values())))
    n_grades = len(AGE_GRADES)

    idx_from_low = int(avg_q * n_grades)
    if idx_from_low == n_grades:
        idx_from_low -= 1

    grade_idx = (n_grades - 1) - idx_from_low
    grade_idx = max(0, min(grade_idx, n_grades - 1))

    lo_age_tier_label = AGE_GRADES[grade_idx]
    lo_age_value = grade_index_to_lo_age_value(grade_idx)
    weak_point = min(q_dict, key=q_dict.get)

    # 3) Persist to Supabase (optional)
    assessment_id: Optional[int] = None

    if req.user_id:
        row = {
            "user_id": req.user_id,
            "sex": req.sex,
            "sit_ups": req.sit_ups,
            "flexibility": req.flexibility,
            "jump_power": req.jump_power,
            "cardio_endurance": req.cardio_endurance,
            "lo_age_tier_label": lo_age_tier_label,
            "tier_index": grade_idx,
            "percentile": avg_q * 100.0,
            "weak_point": weak_point,
            "detail_quantiles": q_dict,
            "measured_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            r = requests.post(
                _sb_table_url("physical_age_assessments"),
                headers=_sb_json_headers(prefer_return=True),
                json=row,
                timeout=10,
            )

            if r.ok:
                payload = r.json()
                if isinstance(payload, list) and payload:
                    assessment_id = payload[0].get("id")
                elif isinstance(payload, dict):
                    assessment_id = payload.get("id")
            else:
                logger.warning("Supabase insert failed: status=%s body=%s", r.status_code, r.text)

        except Exception as e:
            logger.warning("Supabase insert raised an exception: %s", e)

    # 4) Return prediction regardless of persistence outcome
    return PhysicalAgeResponse(
        lo_age_value=lo_age_value,
        lo_age_tier_label=lo_age_tier_label,
        percentile=avg_q * 100.0,
        weak_point=weak_point,
        tier_index=grade_idx,
        detail_quantiles=q_dict,
        avg_quantile=avg_q,
        grade_index=grade_idx,
        grade_label=lo_age_tier_label,
        assessment_id=assessment_id,
    )


@app.get("/users/{user_id}/physical-age/history", response_model=PhysicalAgeHistoryResponse)
def get_physical_age_history(user_id: str, limit: int = 20) -> PhysicalAgeHistoryResponse:
    url = _sb_table_url("physical_age_assessments")
    key = _require_env(SUPABASE_SERVICE_ROLE_KEY, "SUPABASE_SERVICE_ROLE_KEY")

    params = {
        "user_id": f"eq.{user_id}",
        "order": "measured_at.desc",
        "limit": str(limit),
    }
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")

    records = [
        PhysicalAgeRecord(
            id=row["id"],
            user_id=row["user_id"],
            measured_at=row["measured_at"],
            grade_index=row.get("tier_index"),
            grade_label=row.get("lo_age_tier_label"),
            percentile=row.get("percentile"),
            weak_point=row.get("weak_point"),
            avg_quantile=(row.get("percentile") or 0) / 100.0,
            lo_age_value=row.get("lo_age_value"),
            lo_age_tier_label=row.get("lo_age_tier_label"),
            detail_quantiles=row.get("detail_quantiles"),
        )
        for row in rows
    ]

    return PhysicalAgeHistoryResponse(user_id=user_id, records=records)


@app.get("/facilities/near", response_model=List[FacilityOut])
def get_near_facilities(lat: float, lon: float, radius_km: float = 2.0) -> List[FacilityOut]:
    global _facilities_df

    if _facilities_df is None:
        _facilities_df = load_facilities()

    if _facilities_df is None or _facilities_df.empty:
        return []

    df = _facilities_df
    results: List[FacilityOut] = []

    for _, row in df.iterrows():
        try:
            if haversine_km(lat, lon, float(row["lat"]), float(row["lon"])) > radius_km:
                continue

            name_val = row.get("name")
            name = "Unnamed facility" if pd.isna(name_val) else str(name_val)

            addr_val = row.get("address")
            address = "" if pd.isna(addr_val) else str(addr_val)

            category = infer_category(row)

            detail_equip = row.get("detail_equip")
            if pd.isna(detail_equip) or str(detail_equip).strip() == "":
                first_cat = category.split(",")[0].strip()
                mission = f"{first_cat} 운동"
            else:
                mission = str(detail_equip)

            results.append(
                FacilityOut(
                    id=int(row["id"]),
                    name=name,
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    address=address,
                    mission=mission,
                    category=str(category),
                )
            )
        except Exception as e:
            logger.debug("Facility row parse failed (id=%s): %s", row.get("id", "Unknown"), e)
            continue

    return results


@app.post("/favorites/toggle")
def toggle_favorite(req: FavoriteToggleRequest) -> Dict[str, str]:
    table = "favorite_facilities"

    if req.is_favorite:
        payload = {"user_id": req.user_id, "facility_id": req.facility_id}
        r = requests.post(_sb_table_url(table), headers=_sb_json_headers(), json=payload, timeout=10)
    else:
        r = requests.delete(
            f"{_sb_table_url(table)}?user_id=eq.{req.user_id}&facility_id=eq.{req.facility_id}",
            headers=_sb_json_headers(),
            timeout=10,
        )

    if r.status_code not in (200, 201, 204, 409):  # 409: already exists -> treat as success
        raise HTTPException(r.status_code, f"Favorite operation failed: {r.text}")

    return {"status": "ok"}


@app.post("/mission/complete")
def complete_mission(req: MissionCompleteRequest) -> Dict[str, object]:
    r = requests.post(
        _sb_table_url("mission_logs"),
        headers=_sb_json_headers(prefer_return=True),
        json=req.model_dump(),
        timeout=10,
    )

    if r.status_code not in (200, 201):
        raise HTTPException(r.status_code, f"Mission log insert failed: {r.text}")

    return {"status": "ok", "data": r.json()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
