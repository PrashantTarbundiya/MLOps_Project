import os
import sys
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from prediction_model.config import config
from prediction_model.pipeline import song_cluster_pipeline
from prediction_model.processing.data_handling import load_pipeline, load_metadata
from prediction_model.predict import (
    search_tracks,
    recommend_by_song,
    recommend_by_playlist,
    recommend_by_mood,
    predict_single_cluster
)
from drift_monitoring.drift_detector import AudioDriftDetector

def test_pipeline_exists_and_loads():
    """Verify the trained clustering pipeline loads with correct transformer architecture."""
    pipeline = load_pipeline()
    assert pipeline is not None
    assert 'imputer' in pipeline.named_steps
    assert 'scaler' in pipeline.named_steps
    assert 'kmeans' in pipeline.named_steps
    assert pipeline.named_steps['kmeans'].n_clusters == config.N_CLUSTERS

def test_metadata_integrity():
    """Verify model metadata exists and contains essential metrics and profiles."""
    meta = load_metadata()
    assert meta is not None
    assert meta.get("model_version") == config.MODEL_VERSION
    assert "metrics" in meta
    assert "silhouette_score" in meta["metrics"]
    assert "inertia" in meta["metrics"]
    assert meta["n_clusters"] == config.N_CLUSTERS

def test_track_search():
    """Verify track search returns matching results with expected schema."""
    results = search_tracks("Nirvana", limit=5)
    assert len(results) > 0
    first = results[0]
    assert "name" in first
    assert "artist" in first
    assert "cluster_id" in first
    assert 0 <= first["cluster_id"] < config.N_CLUSTERS

def test_recommend_by_song():
    """Verify single song recommendation produces ranked results with valid similarities."""
    res = recommend_by_song("Smells Like Teen Spirit", n_recommendations=5)
    assert "recommendations" in res
    recs = res["recommendations"]
    assert len(recs) == 5
    for track in recs:
        assert 0.0 <= track["similarity_score"] <= 100.0
        assert track["name"].lower() != "smells like teen spirit"
        assert "radar_features" in track
        for rf in config.RADAR_FEATURES:
            assert rf in track["radar_features"]

def test_recommend_by_playlist():
    """Verify multi-track playlist centroid recommendation works."""
    playlist = [
        {"name": "Come As You Are", "year": 1991},
        {"name": "Lithium", "year": 1992}
    ]
    res = recommend_by_playlist(playlist, n_recommendations=4)
    assert "recommendations" in res
    assert len(res["recommendations"]) == 4

def test_recommend_by_mood():
    """Verify mood-based recommendation maps audio preferences to valid songs."""
    mood = {
        "danceability": 0.85,
        "energy": 0.9,
        "valence": 0.75,
        "acousticness": 0.05
    }
    res = recommend_by_mood(mood, n_recommendations=5)
    assert "recommendations" in res
    assert len(res["recommendations"]) == 5
    assert "target_cluster" in res
    assert 0 <= res["target_cluster"]["cluster_id"] < config.N_CLUSTERS

def test_predict_single_cluster():
    """Verify single track feature input predicts a cluster and provides archetype description."""
    feat = {
        "valence": 0.8,
        "danceability": 0.7,
        "energy": 0.8,
        "acousticness": 0.1,
        "tempo": 128.0
    }
    res = predict_single_cluster(feat)
    assert 0 <= res["cluster_id"] < config.N_CLUSTERS
    assert "name" in res
    assert "mood" in res

def test_drift_detector():
    """Verify statistical drift detector can evaluate batches and detect distribution status."""
    detector = AudioDriftDetector()
    baseline = detector.load_baseline()
    assert len(baseline) > 0
    
    # Check baseline against sample of itself -> should be HEALTHY
    sample = baseline.sample(100, random_state=42)
    res = detector.detect_drift(sample)
    assert res["status"] in ["HEALTHY", "DRIFT_WARNING"]
    assert "feature_drifts" in res
    assert "data_quality" in res
