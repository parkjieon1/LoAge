# LoAge API (FastAPI)

LoAge is a fitness analytics service that computes a user's physical age and percentile scores from fitness metrics, and provides nearby facility recommendations via Supabase + location data.

## Tech Stack
- FastAPI (Python)
- Supabase (Postgres + REST)
- Naver Directions API (proxy endpoint)
- Joblib artifacts for quantile engine / regression model

## Project Structure
- `main.py` — FastAPI app entrypoint
- `routers/` — API routes (physical age, facilities, auth, naver directions)
- `models/` — serialized model artifacts (`.pkl`) loaded at runtime

## Environment Variables
Create `.env` locally (do not commit). See `.env.example`.

Required:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Optional:
- `FACILITIES_TABLE` (default: `facilities`)
- `NAVER_CLIENT_ID`
- `NAVER_CLIENT_SECRET`

## Run Locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
