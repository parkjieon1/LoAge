import os
import logging

import requests
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(
    prefix="/naver",
    tags=["naver-directions"],
)

logger = logging.getLogger(__name__)

# =========================
# Environment configuration
# =========================

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

BASE_URL = "https://maps.apigw.ntruss.com/map-direction/v1/driving"

# =========================
# Directions endpoint
# =========================

@router.get("/directions")
def get_directions(
    start: str = Query(..., description="Longitude,Latitude (e.g. 126.9780,37.5665)"),
    goal: str = Query(..., description="Longitude,Latitude (e.g. 126.9920,37.5700)"),
    option: str = Query(
        "traoptimal",
        description="Route option (traoptimal / trafast / tracomfort)",
    ),
):
    """
    Proxy endpoint for Naver Directions V1 Driving API.

    Architecture:
    Client (Flutter/Web) → Application Server → Naver Maps API

    This endpoint hides API credentials from the client and
    forwards routing responses transparently.
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        logger.error("Naver API credentials are not configured.")
        raise HTTPException(
            status_code=500,
            detail="Naver API credentials are not configured.",
        )

    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }

    params = {
        "start": start,
        "goal": goal,
        "option": option,
    }

    try:
        response = requests.get(
            BASE_URL,
            headers=headers,
            params=params,
            timeout=5,
        )
    except requests.RequestException as e:
        logger.warning("Failed to call Naver Directions API: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Failed to call Naver Directions API.",
        )

    if response.status_code != 200:
        logger.warning(
            "Naver API returned non-200 status: %s, body=%s",
            response.status_code,
            response.text,
        )
        raise HTTPException(
            status_code=response.status_code,
            detail="Naver Directions API returned an error response.",
        )

    data = response.json()

    # Naver API returns an internal 'code'; 0 indicates success
    if isinstance(data, dict) and data.get("code") != 0:
        logger.warning(
            "Naver API internal error: code=%s, message=%s",
            data.get("code"),
            data.get("message"),
        )
        raise HTTPException(
            status_code=400,
            detail={
                "naver_code": data.get("code"),
                "naver_message": data.get("message"),
            },
        )

    # Forward successful response payload as-is
    return data
