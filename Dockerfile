FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONPATH="${PYTHONPATH}:/app"

# Verify pipeline tests during build
RUN python -m pytest tests/test_prediction.py -v

EXPOSE 8005

ENTRYPOINT ["python"]
CMD ["main.py"]
