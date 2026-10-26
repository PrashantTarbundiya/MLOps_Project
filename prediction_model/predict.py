import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from prediction_model.config import config
from prediction_model.processing.data_handling import (
    load_pipeline,
    load_catalog_index,
    load_metadata,
    load_dataset
)
from prediction_model.processing.preprocessing import get_cluster_info

# In-memory singleton cache
_PIPELINE = None
_CATALOG = None
_DATASET = None
_METADATA = None

def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = load_pipeline()
    return _PIPELINE

def get_catalog():
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = load_catalog_index()
        if _CATALOG is None:
            # Fallback if catalog not cached yet: load from dataset and pipeline
            pipeline = get_pipeline()
            raw_data = get_raw_dataset()
            features = [c for c in config.AUDIO_FEATURES if c in raw_data.columns]
            imputed = pipeline.named_steps['imputer'].transform(raw_data[features])
            scaled = pipeline.named_steps['scaler'].transform(imputed)
            metadata_cols = [c for c in ['id', 'name', 'artists', 'artists_clean', 'year', 'popularity', 'release_date'] if c in raw_data.columns]
            meta = raw_data[metadata_cols].copy()
            meta['cluster_label'] = pipeline.predict(raw_data[features])
            _CATALOG = {
                "metadata": meta,
                "scaled_features": scaled.astype(np.float32),
                "feature_names": features
            }
    return _CATALOG

def get_raw_dataset():
    global _DATASET
    if _DATASET is None:
        _DATASET = load_dataset()
    return _DATASET

def get_cached_metadata():
    global _METADATA
    if _METADATA is None:
        _METADATA = load_metadata()
    return _METADATA

def search_tracks(query: str, limit: int = 15):
    """Fuzzy / substring search for tracks across catalog by title or artist."""
    if not query or len(query.strip()) < 2:
        return []
    
    q = query.strip().lower()
    cat = get_catalog()
    meta = cat["metadata"]

    mask = (
        meta['name'].str.lower().str.contains(q, na=False) |
        meta['artists_clean'].str.lower().str.contains(q, na=False)
    )
    
    matches = meta[mask].nlargest(limit, 'popularity')
    results = []
    for _, row in matches.iterrows():
        c_info = get_cluster_info(int(row.get('cluster_label', 0)))
        results.append({
            "id": str(row.get('id', '')),
            "name": str(row.get('name', 'Unknown')),
            "artist": str(row.get('artists_clean', 'Unknown')),
            "year": int(row.get('year', 0)),
            "popularity": int(row.get('popularity', 0)),
            "cluster_id": int(row.get('cluster_label', 0)),
            "cluster_name": c_info["name"],
            "cluster_icon": c_info["icon"]
        })
    return results

def recommend_by_song(song_name: str, year: int = None, n_recommendations: int = 10):
    """Recommends similar tracks given a seed song title and optional release year."""
    cat = get_catalog()
    meta = cat["metadata"]
    scaled_feats = cat["scaled_features"]
    raw_df = get_raw_dataset()

    # Find the seed song
    mask = meta['name'].str.lower() == song_name.strip().lower()
    if year is not None:
        mask = mask & (meta['year'] == int(year))
    
    candidates = meta[mask]
    if candidates.empty:
        # Partial match fallback
        mask = meta['name'].str.lower().str.contains(song_name.strip().lower(), na=False)
        candidates = meta[mask]
        if candidates.empty:
            return {"error": f"Track '{song_name}' not found in database.", "recommendations": []}

    # Pick the most popular candidate match
    seed_idx = candidates.index[0]
    seed_meta = meta.loc[seed_idx]
    seed_vector = scaled_feats[seed_idx].reshape(1, -1)

    # Compute Cosine Distance
    distances = cdist(seed_vector, scaled_feats, metric='cosine')[0]
    
    # Sort closest indices, excluding the seed song itself
    sorted_indices = np.argsort(distances)
    
    recommendations = []
    seen_names = {seed_meta['name'].lower()}
    
    for idx in sorted_indices:
        cand_name = str(meta.iloc[idx]['name'])
        if cand_name.lower() in seen_names:
            continue
        
        seen_names.add(cand_name.lower())
        c_label = int(meta.iloc[idx].get('cluster_label', 0))
        c_info = get_cluster_info(c_label)
        similarity = max(0.0, min(1.0, 1.0 - float(distances[idx])))

        # Extract radar audio attributes for UI
        raw_row = raw_df.iloc[idx]
        radar = {feat: round(float(raw_row.get(feat, 0.5)), 3) for feat in config.RADAR_FEATURES}

        recommendations.append({
            "id": str(meta.iloc[idx].get('id', '')),
            "name": cand_name,
            "artist": str(meta.iloc[idx].get('artists_clean', 'Unknown')),
            "year": int(meta.iloc[idx].get('year', 0)),
            "popularity": int(meta.iloc[idx].get('popularity', 0)),
            "similarity_score": round(similarity * 100, 1),
            "cluster_id": c_label,
            "cluster_name": c_info["name"],
            "cluster_icon": c_info["icon"],
            "radar_features": radar
        })
        if len(recommendations) >= n_recommendations:
            break

    seed_raw = raw_df.loc[seed_idx]
    seed_radar = {feat: round(float(seed_raw.get(feat, 0.5)), 3) for feat in config.RADAR_FEATURES}

    return {
        "seed_track": {
            "name": str(seed_meta['name']),
            "artist": str(seed_meta['artists_clean']),
            "year": int(seed_meta['year']),
            "popularity": int(seed_meta['popularity']),
            "cluster_id": int(seed_meta['cluster_label']),
            "cluster_name": get_cluster_info(int(seed_meta['cluster_label']))["name"],
            "cluster_icon": get_cluster_info(int(seed_meta['cluster_label']))["icon"],
            "radar_features": seed_radar
        },
        "recommendations": recommendations
    }

def recommend_by_playlist(songs_list: list, n_recommendations: int = 10):
    """
    Given a list of songs [{'name': '...', 'year': ...}], computes the mean centroid vector
    in scaled audio space and recommends closest matching tracks.
    """
    if not songs_list:
        return {"error": "Empty playlist provided.", "recommendations": []}

    cat = get_catalog()
    meta = cat["metadata"]
    scaled_feats = cat["scaled_features"]
    raw_df = get_raw_dataset()

    found_vectors = []
    playlist_names = set()

    for item in songs_list:
        s_name = item.get('name', '').strip().lower()
        s_year = item.get('year')
        mask = meta['name'].str.lower() == s_name
        if s_year is not None:
            mask = mask & (meta['year'] == int(s_year))
        matches = meta[mask]
        if not matches.empty:
            idx = matches.index[0]
            found_vectors.append(scaled_feats[idx])
            playlist_names.add(matches.iloc[0]['name'].lower())

    if not found_vectors:
        return {"error": "None of the specified songs were found in catalog.", "recommendations": []}

    centroid = np.mean(found_vectors, axis=0).reshape(1, -1)
    distances = cdist(centroid, scaled_feats, metric='cosine')[0]
    sorted_indices = np.argsort(distances)

    recommendations = []
    seen = set(playlist_names)
    for idx in sorted_indices:
        cand_name = str(meta.iloc[idx]['name'])
        if cand_name.lower() in seen:
            continue
        seen.add(cand_name.lower())

        c_label = int(meta.iloc[idx].get('cluster_label', 0))
        c_info = get_cluster_info(c_label)
        similarity = max(0.0, min(1.0, 1.0 - float(distances[idx])))
        raw_row = raw_df.iloc[idx]
        radar = {feat: round(float(raw_row.get(feat, 0.5)), 3) for feat in config.RADAR_FEATURES}

        recommendations.append({
            "id": str(meta.iloc[idx].get('id', '')),
            "name": cand_name,
            "artist": str(meta.iloc[idx].get('artists_clean', 'Unknown')),
            "year": int(meta.iloc[idx].get('year', 0)),
            "popularity": int(meta.iloc[idx].get('popularity', 0)),
            "similarity_score": round(similarity * 100, 1),
            "cluster_id": c_label,
            "cluster_name": c_info["name"],
            "cluster_icon": c_info["icon"],
            "radar_features": radar
        })
        if len(recommendations) >= n_recommendations:
            break

    return {
        "playlist_tracks_matched": len(found_vectors),
        "total_requested": len(songs_list),
        "recommendations": recommendations
    }

def recommend_by_mood(mood_params: dict, n_recommendations: int = 10):
    """
    Given targeted mood sliders (e.g. danceability, energy, valence, acousticness, tempo),
    projects input vector into scaled space and finds closest acoustic matches.
    """
    pipeline = get_pipeline()
    cat = get_catalog()
    meta = cat["metadata"]
    scaled_feats = cat["scaled_features"]
    raw_df = get_raw_dataset()

    # Build input dictionary with defaults
    default_row = {
        'valence': 0.5,
        'year': 2010,
        'acousticness': 0.2,
        'danceability': 0.6,
        'duration_ms': 200000,
        'energy': 0.6,
        'explicit': 0,
        'instrumentalness': 0.05,
        'key': 5,
        'liveness': 0.15,
        'loudness': -7.0,
        'mode': 1,
        'popularity': 50,
        'speechiness': 0.05,
        'tempo': 120.0
    }
    for k, v in mood_params.items():
        if k in default_row and v is not None:
            default_row[k] = float(v)

    input_df = pd.DataFrame([default_row])[cat["feature_names"]]
    imputed = pipeline.named_steps['imputer'].transform(input_df)
    target_vector = pipeline.named_steps['scaler'].transform(imputed)

    # Predict target cluster
    predicted_cluster = int(pipeline.named_steps['kmeans'].predict(target_vector)[0])
    c_info = get_cluster_info(predicted_cluster)

    distances = cdist(target_vector, scaled_feats, metric='cosine')[0]
    sorted_indices = np.argsort(distances)

    recommendations = []
    seen = set()
    for idx in sorted_indices:
        cand_name = str(meta.iloc[idx]['name'])
        if cand_name.lower() in seen:
            continue
        seen.add(cand_name.lower())

        c_label = int(meta.iloc[idx].get('cluster_label', 0))
        c_item_info = get_cluster_info(c_label)
        similarity = max(0.0, min(1.0, 1.0 - float(distances[idx])))
        raw_row = raw_df.iloc[idx]
        radar = {feat: round(float(raw_row.get(feat, 0.5)), 3) for feat in config.RADAR_FEATURES}

        recommendations.append({
            "id": str(meta.iloc[idx].get('id', '')),
            "name": cand_name,
            "artist": str(meta.iloc[idx].get('artists_clean', 'Unknown')),
            "year": int(meta.iloc[idx].get('year', 0)),
            "popularity": int(meta.iloc[idx].get('popularity', 0)),
            "similarity_score": round(similarity * 100, 1),
            "cluster_id": c_label,
            "cluster_name": c_item_info["name"],
            "cluster_icon": c_item_info["icon"],
            "radar_features": radar
        })
        if len(recommendations) >= n_recommendations:
            break

    return {
        "target_cluster": {
            "cluster_id": predicted_cluster,
            "name": c_info["name"],
            "mood": c_info["mood"],
            "icon": c_info["icon"]
        },
        "target_mood": mood_params,
        "recommendations": recommendations
    }

def predict_single_cluster(features_dict: dict):
    """Predicts the sonic cluster and provides audio archetype analysis for a single song."""
    pipeline = get_pipeline()
    cat = get_catalog()
    
    # Fill defaults for missing features
    row = {
        'valence': 0.5, 'year': 2015, 'acousticness': 0.3, 'danceability': 0.5,
        'duration_ms': 210000, 'energy': 0.5, 'explicit': 0, 'instrumentalness': 0.1,
        'key': 5, 'liveness': 0.2, 'loudness': -8.0, 'mode': 1, 'popularity': 50,
        'speechiness': 0.05, 'tempo': 115.0
    }
    for k, v in features_dict.items():
        if k in row and v is not None:
            row[k] = float(v)

    input_df = pd.DataFrame([row])[cat["feature_names"]]
    imputed = pipeline.named_steps['imputer'].transform(input_df)
    cluster_id = int(pipeline.named_steps['kmeans'].predict(pipeline.named_steps['scaler'].transform(imputed))[0])
    c_info = get_cluster_info(cluster_id)

    return {
        "cluster_id": cluster_id,
        "name": c_info["name"],
        "mood": c_info["mood"],
        "icon": c_info["icon"],
        "radar_features": {feat: round(float(row.get(feat, 0.5)), 3) for feat in config.RADAR_FEATURES}
    }