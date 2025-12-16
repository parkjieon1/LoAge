FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# requirements 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 소스 전체 복사 (main.py, models/, routers/ 등)
COPY . .

# Cloud Run 기본 포트는 환경변수 PORT 사용
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

