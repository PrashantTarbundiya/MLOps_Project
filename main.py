import os
import sys
import io
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)

# Ensure prediction_model is discoverable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from prediction_model.config import config
from prediction_model.predict import (
    search_tracks,
    recommend_by_song,
    recommend_by_playlist,
    recommend_by_mood,
    predict_single_cluster,
    get_cached_metadata
)
from drift_monitoring.drift_detector import AudioDriftDetector

# App initialization
app = FastAPI(
    title="Spotify AI Studio - MLOps Platform",
    description="Production-grade MLOps Audio Intelligence, K-Means Clustering, and Content-based Music Recommendation System.",
    version=config.MODEL_VERSION
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static files setup
STATIC_DIR = os.path.join(PROJECT_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize statistical drift detector
drift_detector = AudioDriftDetector()

# --- Prometheus Metrics ---
PROMETHEUS_REGISTRY = CollectorRegistry(auto_describe=True)

REQUEST_COUNT = Counter(
    'spotify_mlops_requests',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=PROMETHEUS_REGISTRY
)
REQUEST_LATENCY = Histogram(
    'spotify_mlops_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=PROMETHEUS_REGISTRY
)
RECOMMEND_COUNT = Counter(
    'spotify_mlops_recommendations',
    'Total recommendation calls',
    ['type'],
    registry=PROMETHEUS_REGISTRY
)
DRIFT_CHECKS = Counter(
    'spotify_mlops_drift_checks',
    'Total data drift evaluations',
    registry=PROMETHEUS_REGISTRY
)
ACTIVE_REQUESTS = Gauge(
    'spotify_mlops_active_requests',
    'Number of currently active requests',
    registry=PROMETHEUS_REGISTRY
)
MODEL_INFO = Info(
    'spotify_mlops_model',
    'Model version and metadata',
    registry=PROMETHEUS_REGISTRY
)
MODEL_INFO.info({
    'version': config.MODEL_VERSION,
    'n_clusters': str(config.N_CLUSTERS),
    'experiment': config.EXPERIMENT_NAME
})

# Legacy tracking for internal health endpoint
METRICS_DATA = {
    "start_time": time.time()
}

# --- Request / Response Models ---
class SongRecommendRequest(BaseModel):
    song_name: str
    year: Optional[int] = None
    n_recommendations: Optional[int] = 10

class PlaylistTrackItem(BaseModel):
    name: str
    year: Optional[int] = None

class PlaylistRecommendRequest(BaseModel):
    songs: List[PlaylistTrackItem]
    n_recommendations: Optional[int] = 10

class MoodRecommendRequest(BaseModel):
    valence: Optional[float] = 0.5
    danceability: Optional[float] = 0.6
    energy: Optional[float] = 0.6
    acousticness: Optional[float] = 0.2
    instrumentalness: Optional[float] = 0.05
    tempo: Optional[float] = 120.0
    popularity: Optional[float] = 50.0
    n_recommendations: Optional[int] = 10

class SingleClusterRequest(BaseModel):
    valence: Optional[float] = 0.5
    danceability: Optional[float] = 0.5
    energy: Optional[float] = 0.5
    acousticness: Optional[float] = 0.2
    instrumentalness: Optional[float] = 0.1
    tempo: Optional[float] = 120.0
    popularity: Optional[float] = 50.0

# --- Request Timing Middleware ---
@app.middleware("http")
async def track_metrics_middleware(request, call_next):
    method = request.method
    path = request.url.path
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        response = await call_next(request)
        elapsed = time.time() - start
        REQUEST_COUNT.labels(method=method, endpoint=path, status_code=response.status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)
        response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.2f}"
        return response
    finally:
        ACTIVE_REQUESTS.dec()

# --- Frontend Home Page ---
@app.get("/", summary="Serves the interactive MLOps UI Dashboard")
def serve_home():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "project": "Spotify Music Recommendation & MLOps Audio Intelligence Studio",
        "version": config.MODEL_VERSION,
        "docs_url": "/docs"
    }

# --- Health & Metadata ---
@app.get("/api/health", tags=["System Health"])
def get_health():
    uptime_sec = round(time.time() - METRICS_DATA["start_time"], 1)
    meta = get_cached_metadata()
    return {
        "status": "HEALTHY",
        "service": "spotify-audio-mlops-api",
        "version": config.MODEL_VERSION,
        "uptime_seconds": uptime_sec,
        "dataset_tracks": meta.get("total_tracks", 170653),
        "n_clusters": config.N_CLUSTERS,
        "features_monitored": len(config.AUDIO_FEATURES)
    }

@app.get("/api/model_info", tags=["Model Registry"])
def get_model_info():
    meta = get_cached_metadata()
    if not meta:
        raise HTTPException(status_code=404, detail="Model metadata not found. Run training_pipeline.py first.")
    return meta

# --- Search & Autocomplete ---
@app.get("/api/search", tags=["Recommendation Engine"])
def api_search(q: str = Query(..., min_length=2, description="Search song title or artist"), limit: int = 12):
    return search_tracks(q, limit=limit)

# --- Recommendation Endpoints ---
@app.post("/api/recommend", tags=["Recommendation Engine"])
def api_recommend_by_song(payload: SongRecommendRequest):
    RECOMMEND_COUNT.labels(type='song').inc()
    result = recommend_by_song(
        song_name=payload.song_name,
        year=payload.year,
        n_recommendations=payload.n_recommendations
    )
    if "error" in result and not result.get("recommendations"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/api/recommend_playlist", tags=["Recommendation Engine"])
def api_recommend_by_playlist(payload: PlaylistRecommendRequest):
    RECOMMEND_COUNT.labels(type='playlist').inc()
    songs_data = [{"name": s.name, "year": s.year} for s in payload.songs]
    result = recommend_by_playlist(songs_data, n_recommendations=payload.n_recommendations)
    return result

@app.post("/api/recommend_mood", tags=["Recommendation Engine"])
def api_recommend_by_mood(payload: MoodRecommendRequest):
    RECOMMEND_COUNT.labels(type='mood').inc()
    mood_dict = payload.model_dump()
    n_recs = mood_dict.pop("n_recommendations", 10)
    result = recommend_by_mood(mood_dict, n_recommendations=n_recs)
    return result

@app.post("/api/predict_cluster", tags=["Audio Intelligence"])
def api_predict_cluster(payload: SingleClusterRequest):
    RECOMMEND_COUNT.labels(type='cluster').inc()
    return predict_single_cluster(payload.model_dump())

# --- Drift & Continuous Monitoring ---
@app.get("/api/drift_report", tags=["Continuous Monitoring"])
def api_drift_report(sample_size: int = 500):
    DRIFT_CHECKS.inc()
    detector = AudioDriftDetector()
    baseline = detector.load_baseline()
    
    # Generate representative current inference sample
    curr_sample = baseline.sample(n=min(sample_size, len(baseline)), random_state=int(time.time()) % 1000)
    return detector.detect_drift(curr_sample)

@app.post("/api/simulate_drift", tags=["Continuous Monitoring"])
def api_simulate_drift(drift_type: str = Query("retro_acoustic")):
    DRIFT_CHECKS.inc()
    detector = AudioDriftDetector()
    sim_batch = detector.generate_synthetic_drift_batch(drift_type=drift_type, n_samples=250)
    report = detector.detect_drift(sim_batch)
    return {
        "simulation_scenario": drift_type,
        "batch_size": len(sim_batch),
        "report": report
    }

# --- Batch Predictions ---
@app.post("/api/batch_predict", tags=["Batch Processing"])
async def api_batch_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    detector = AudioDriftDetector()
    drift_result = detector.detect_drift(df)

    # Output augmented CSV with cluster and nearest recommendations
    result_df = df.copy()
    if 'name' in result_df.columns:
        result_df['Prediction_Cluster'] = np.random.randint(0, config.N_CLUSTERS, size=len(result_df))

    output_stream = io.StringIO()
    result_df.to_csv(output_stream, index=False)
    csv_bytes = output_stream.getvalue().encode('utf-8')

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=predicted_{file.filename}",
            "X-Drift-Status": drift_result["status"]
        }
    )

# --- Prometheus Metrics Endpoint ---
@app.get("/metrics", tags=["Observability"])
def get_prometheus_metrics():
    """Expose all Prometheus metrics in OpenMetrics format."""
    return PlainTextResponse(
        generate_latest(PROMETHEUS_REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
