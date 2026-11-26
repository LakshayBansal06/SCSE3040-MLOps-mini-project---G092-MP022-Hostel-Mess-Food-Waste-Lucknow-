# Container image for the MP022 prediction service.
# Build:  docker build -t mp022-api:1.0 .
# Run:    docker run --rm -p 8000:8000 mp022-api:1.0

FROM python:3.12-slim

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv:/srv/src

# Dependencies first, so editing the code does not reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Then the code, the settings and the champion model the service loads.
COPY params.yaml ./
COPY src ./src
COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
