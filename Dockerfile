# 베이스 이미지 선택 (Python 버전 명시)
FROM python:3.11-slim

# 컨테이너 내부 작업 디렉터리 설정
WORKDIR /app

# requirements 먼저 복사 (캐시 효율 위해 순서 중요)
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 복사
COPY . .

# 기본 명령 (필요 시)
CMD ["python"]
