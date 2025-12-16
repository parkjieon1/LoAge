from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import math
import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

# --- 환경변수 및 기본 설정 ---
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# --- Supabase 헬퍼 ---
def _sb_json_headers(prefer_return: bool = False) -> Dict[str, str]:
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY가 설정되지 않았습니다.")
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer_return:
        headers["Prefer"] = "return=representation"
    return headers

def _sb_table_url(table: str) -> str:
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL이 설정되지 않았습니다.")
    return f"{SUPABASE_URL}/rest/v1/{table}"

# --- 모델 및 데이터 로딩 ---
_facilities_df: Optional[pd.DataFrame] = None
ENGINE_PATH = BASE_DIR / "models" / "model.pkl"
_engine_cache: Optional[Dict[str, pd.DataFrame]] = None

AGE_GRADES = [
    "19세 이하", "20대 초반", "20대 중반", "20대 후반", "30대 초반", "30대 중반",
    "30대 후반", "40대 초반", "40대 중반", "40대 후반", "50대 초반", "50대 중반",
    "50대 후반", "60대 초반", "60대 중반", "60대 후반", "70대 이상",
]

def load_facilities() -> pd.DataFrame:
    global _facilities_df
    # 캐시된 데이터가 있으면 반환 (개발 중에는 주석 처리하여 매번 로딩할 수도 있음)
    if _facilities_df is not None: return _facilities_df
    
    table_name = os.getenv("FACILITIES_TABLE", "facilities")
    base_url = _sb_table_url(table_name)
    headers = {"apikey": SUPABASE_SERVICE_ROLE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}", "Range-Unit": "items", "Prefer": "count=exact"}
    page_size = 1000; start = 0; frames: List[pd.DataFrame] = []
    
    while True:
        end = start + page_size - 1
        headers["Range"] = f"{start}-{end}"
        params={"select": "id,name,lat,lon,address,detail_equip,type,is_muscular_endurance,is_flexibility,is_cardio,quickness"}
        resp = requests.get(base_url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data: break
        frames.append(pd.DataFrame(data))
        if len(data) < page_size: break
        start += page_size
        
    if not frames: 
        _facilities_df = pd.DataFrame(columns=["id", "name", "lat", "lon", "address", "detail_equip", "is_cardio", "is_muscular_endurance", "is_flexibility", "quickness"])
        return _facilities_df
        
    df = pd.concat(frames, ignore_index=True).dropna(subset=["lat", "lon"])
    _facilities_df = df.astype({"lat": float, "lon": float})
    return _facilities_df

def load_engine() -> Dict[str, pd.DataFrame]:
    global _engine_cache
    if _engine_cache is not None: return _engine_cache
    if not ENGINE_PATH.exists(): raise FileNotFoundError(f"엔진 파일을 찾을 수 없습니다: {ENGINE_PATH}")
    obj = joblib.load(str(ENGINE_PATH))
    _engine_cache = obj
    return _engine_cache

# --- 유틸리티 함수 ---
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0; p = math.pi / 180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) * math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2
    return R * 2 * math.asin(math.sqrt(a))

def infer_category(row) -> str:
    categories = []
    # 1.0 (float) or 1 (int) 비교를 위해 안전하게 처리
    if row.get("is_cardio") == 1 or row.get("is_cardio") == 1.0: categories.append("심폐지구력")
    if row.get("is_muscular_endurance") == 1 or row.get("is_muscular_endurance") == 1.0: categories.append("근지구력")
    if row.get("is_flexibility") == 1 or row.get("is_flexibility") == 1.0: categories.append("유연성")
    if row.get("quickness") == 1 or row.get("quickness") == 1.0: categories.append("순발력")
    return ", ".join(categories) if categories else "기타"

def get_quantile_from_table(df: pd.DataFrame, sex_col: str, value: float) -> float:
    if sex_col not in df.columns: raise KeyError(f"엔진 테이블에 '{sex_col}' 컬럼이 없습니다.")
    q = df.index.to_numpy(dtype=float)
    v = df[sex_col].to_numpy(dtype=float)
    order = np.argsort(v)
    return float(np.interp(value, v[order], q[order], left=0.0, right=1.0))

def grade_index_to_lo_age_value(idx: int) -> int:
    idx = max(0, min(idx, len(AGE_GRADES) - 1))
    mapping = {
        0: 18, 1: 22, 2: 25, 3: 28, 4: 32, 5: 35, 6: 38, 7: 42, 8: 45, 9: 48,
        10: 52, 11: 55, 12: 58, 13: 62, 14: 65, 15: 68, 16: 72
    }
    return mapping.get(idx, 40)

# --- Pydantic 모델 ---
class PhysicalAgeRequest(BaseModel):
    user_id: Optional[str] = None; sex: str; flexibility: float; jump_power: float; cardio_endurance: float; sit_ups: float
    @field_validator("sex")
    def normalize_sex(cls, v: str) -> str:
        s = v.strip().lower()
        if s in ["f", "female", "여"]: return "Female"
        if s in ["m", "male", "남"]: return "Male"
        raise ValueError("sex는 남/여(M/F) 형태로 입력해야 합니다.")

class PhysicalAgeResponse(BaseModel):
    lo_age_value: int; lo_age_tier_label: str; percentile: float; weak_point: str; tier_index: int
    detail_quantiles: Dict[str, float]; avg_quantile: float; grade_index: int; grade_label: str
    assessment_id: Optional[int] = None

class PhysicalAgeRecord(BaseModel):
    id: int; user_id: str; measured_at: datetime
    grade_index: Optional[int] = None; grade_label: Optional[str] = None
    percentile: Optional[float] = None; weak_point: Optional[str] = None
    avg_quantile: Optional[float] = None; lo_age_value: Optional[int] = None
    lo_age_tier_label: Optional[str] = None; detail_quantiles: Optional[Dict[str, float]] = None

class PhysicalAgeHistoryResponse(BaseModel):
    user_id: str; records: List[PhysicalAgeRecord]

class FacilityOut(BaseModel):
    id: int; name: str; lat: float; lon: float; address: str; mission: str; category: str

class FavoriteToggleRequest(BaseModel):
    user_id: str; facility_id: int; is_favorite: bool

class MissionCompleteRequest(BaseModel):
    user_id: str; facility_id: int; status: str; is_favorite: bool

# --- FastAPI 앱 ---
app = FastAPI(title="Lo-Age API")

@app.on_event("startup")
def on_startup():
    try: load_engine(); print("[INFO] 신체나이 엔진 로딩 완료")
    except Exception as e: print(f"[ERROR] 신체나이 엔진 로딩 실패: {e}")
    try: load_facilities(); print("[INFO] 시설 데이터 로딩 완료")
    except Exception as e: print(f"[ERROR] 시설 데이터 로딩 실패: {e}")

@app.post("/predict/physical-age", response_model=PhysicalAgeResponse)
def predict_physical_age(req: PhysicalAgeRequest):
    # 디버그용
    print("[DEBUG] /predict-physical-age called, user_id =", repr(req.user_id))

    engine = load_engine()

    # 1) 각 항목별 분위수 계산
    q_dict = {
        "sit_ups": get_quantile_from_table(engine["sit_ups"], req.sex, req.sit_ups),
        "flexibility": get_quantile_from_table(engine["flexibility"], req.sex, req.flexibility),
        "jump_power": get_quantile_from_table(engine["jump_power"], req.sex, req.jump_power),
        "cardio_endurance": get_quantile_from_table(engine["cardio_endurance"], req.sex, req.cardio_endurance),
    }
    # 심폐지구력은 낮을수록 좋으니 반전
    q_dict["cardio_endurance"] = 1.0 - q_dict["cardio_endurance"]

    # 2) 평균 분위수 → grade index / tier 계산
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

    # 3) Supabase 저장 (있으면)
    assessment_id = None
    if req.user_id:
        row = {
            "user_id": req.user_id,
            "sex": req.sex,
            "sit_ups": req.sit_ups,
            "flexibility": req.flexibility,
            "jump_power": req.jump_power,
            "cardio_endurance": req.cardio_endurance,
            # "lo_age_value": lo_age_value,  # ← 컬럼 없으니 제외
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
                headers=_sb_json_headers(True),
                json=row,
                timeout=5,
            )
            print("[SUPABASE] /physical_age_assessments status:", r.status_code)
            print("[SUPABASE] /physical_age_assessments body:", r.text)

            if r.ok:
                data = r.json()
                if isinstance(data, list) and data:
                    assessment_id = data[0].get("id")
                elif isinstance(data, dict):
                    assessment_id = data.get("id")
            else:
                # 지금은 에러만 찍고, 예측 응답은 그대로 내려줌
                print("[WARN] Supabase insert 실패:", r.status_code, r.text)

        except Exception as e:
            print("[WARN] DB 저장 중 예외 발생:", e)

    # 4) 항상 최종 응답 반환 ✅ (여기가 중요)
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
def get_physical_age_history(user_id: str, limit: int = 20):
    url = _sb_table_url("physical_age_assessments")
    params = {"user_id": f"eq.{user_id}", "order": "measured_at.desc", "limit": str(limit)}
    headers = {"apikey": SUPABASE_SERVICE_ROLE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"}
    
    try:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"히스토리 조회 실패: {e}")

    records = [
        PhysicalAgeRecord(
            id=row["id"], user_id=row["user_id"], measured_at=row["measured_at"],
            grade_index=row.get("tier_index"), grade_label=row.get("lo_age_tier_label"),
            percentile=row.get("percentile"), weak_point=row.get("weak_point"),
            avg_quantile=(row.get("percentile") or 0)/100.0, 
            lo_age_value=row.get("lo_age_value"), lo_age_tier_label=row.get("lo_age_tier_label"),
            detail_quantiles=row.get("detail_quantiles")
        ) for row in rows
    ]
    return PhysicalAgeHistoryResponse(user_id=user_id, records=records)

@app.get("/facilities/near", response_model=List[FacilityOut])
def get_near_facilities(lat: float, lon: float, radius_km: float = 2.0):
    # [수정] 데이터 안전 처리 로직 강화
    # 1. 데이터가 로드되지 않았다면 시도
    if _facilities_df is None:
        load_facilities()
    
    # 2. 그래도 없으면 빈 리스트 반환 (500 에러 방지)
    if _facilities_df is None or _facilities_df.empty:
        return []

    df = _facilities_df
    results = []
    
    for _, row in df.iterrows():
        try:
            # 거리 필터
            if haversine_km(lat, lon, row["lat"], row["lon"]) > radius_km:
                continue

            # 3. 데이터 결측치(NaN/Null) 안전 처리
            # 이름 처리
            name_val = row["name"]
            if pd.isna(name_val): name_val = "이름 없는 시설"
            
            # 주소 처리 (Null이면 빈 문자열)
            addr_val = row["address"]
            if pd.isna(addr_val): addr_val = ""
            
            # 카테고리 추론
            category = infer_category(row)

            # 미션/운동설명 처리
            detail_equip = row.get("detail_equip")
            if pd.isna(detail_equip) or str(detail_equip).strip() == "":
                # 상세 장비 정보가 없으면 카테고리 기반 운동 이름 생성
                first_cat = category.split(',')[0].strip()
                mission = f"{first_cat} 운동"
            else:
                mission = str(detail_equip)
            
            # 4. Pydantic 모델로 변환 (타입 캐스팅 명시)
            results.append(FacilityOut(
                id=int(row["id"]),
                name=str(name_val),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                address=str(addr_val),
                mission=str(mission),
                category=str(category)
            ))
        except Exception as e:
            # 특정 행 처리 중 에러가 나도 전체 API가 죽지 않도록 로깅만 하고 넘어감
            print(f"[WARN] 시설(ID={row.get('id', 'Unknown')}) 데이터 변환 오류: {e}")
            continue
            
    return results

@app.post("/favorites/toggle")
def toggle_favorite(req: FavoriteToggleRequest):
    table = "favorite_facilities"
    if req.is_favorite:
        # [수정] Supabase insert 시 is_favorite 필드 제거 (user_id, facility_id만 전송)
        payload = {
            "user_id": req.user_id,
            "facility_id": req.facility_id
        }
        r = requests.post(_sb_table_url(table), headers=_sb_json_headers(), json=payload)
    else:
        r = requests.delete(f'{_sb_table_url(table)}?user_id=eq.{req.user_id}&facility_id=eq.{req.facility_id}', headers=_sb_json_headers())
    if r.status_code not in [200, 201, 204, 409]: # 409 Conflict (이미 존재)는 성공으로 간주
        raise HTTPException(r.status_code, f"즐겨찾기 처리 실패: {r.text}")
    return {"status": "ok"}

@app.post("/mission/complete")
def complete_mission(req: MissionCompleteRequest):
    r = requests.post(_sb_table_url("mission_logs"), headers=_sb_json_headers(True), json=req.dict())
    if r.status_code not in [200, 201]: raise HTTPException(r.status_code, f"미션 로그 저장 실패: {r.text}")
    return {"status": "ok", "data": r.json()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
