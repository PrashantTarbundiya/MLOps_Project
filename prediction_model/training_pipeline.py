import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from prediction_model.config import config
from prediction_model.pipeline import create_clustering_pipeline
from prediction_model.processing.data_handling import (
    load_dataset,
    save_pipeline,
    save_metadata,
    save_catalog_index
)
from prediction_model.processing.preprocessing import CLUSTER_ARCHETYPES, clean_artist_str

def run_training():
    print("=" * 60)
    print("[INFO] Spotify Music Recommendation & Feature Clustering Pipeline")
    print(f"[INFO] Model Version: {config.MODEL_VERSION}")
    print("=" * 60)
    start_time = time.time()

    # Step 1: Load Spotify Dataset
    print(f"[INFO] Loading dataset from: {config.DATA_FILE}")
    data = load_dataset(config.DATA_FILE)
    total_tracks = len(data)
    print(f"[SUCCESS] Loaded {total_tracks:,} tracks successfully.")

    # Step 2: Feature Selection & Imputation
    features_present = [col for col in config.AUDIO_FEATURES if col in data.columns]
    print(f"[INFO] Training on {len(features_present)} audio features:")
    print("      ", features_present)
    
    X = data[features_present].copy()

    # Step 3: Save baseline sample for Drift Monitoring
    os.makedirs(config.DATASETS_DIR, exist_ok=True)
    baseline_df = data.sample(n=min(10000, len(data)), random_state=config.RANDOM_STATE)[features_present]
    baseline_df.to_csv(config.BASELINE_DATA_PATH, index=False)
    print(f"[INFO] Saved baseline reference dataset ({len(baseline_df):,} rows) to: {config.BASELINE_DATA_PATH}")

    # Step 4: Initialize and Train Clustering Pipeline
    print(f"[INFO] Training K-Means Pipeline with K={config.N_CLUSTERS} clusters...")
    pipeline = create_clustering_pipeline(
        n_clusters=config.N_CLUSTERS,
        random_state=config.RANDOM_STATE,
        n_init=config.N_INIT
    )
    pipeline.fit(X)
    print("[SUCCESS] Pipeline fitting completed.")

    # Step 5: Inference & Cluster Assignment
    cluster_labels = pipeline.predict(X)
    data['cluster_label'] = cluster_labels

    # Step 6: Clustering Performance Metrics
    kmeans_model = pipeline.named_steps['kmeans']
    inertia = float(kmeans_model.inertia_)
    
    # Calculate sample silhouette score for computational efficiency (1000 samples to prevent OOM)
    sample_indices = np.random.RandomState(42).choice(len(X), size=min(1000, len(X)), replace=False)
    scaled_sample = pipeline.named_steps['scaler'].transform(
        pipeline.named_steps['imputer'].transform(X.iloc[sample_indices])
    )
    sample_labels = cluster_labels[sample_indices]

    try:
        silhouette = float(silhouette_score(scaled_sample, sample_labels))
    except Exception as e:
        print(f'[WARN] Silhouette score calculation skipped: {e}')
        silhouette = 0.0

    try:
        davies_bouldin = float(davies_bouldin_score(scaled_sample, sample_labels))
    except Exception as e:
        davies_bouldin = 0.0

    try:
        calinski = float(calinski_harabasz_score(scaled_sample, sample_labels))
    except Exception as e:
        calinski = 0.0

    print("-" * 60)
    print("[METRICS] Model Evaluation Metrics:")
    print(f"   • Inertia (Sum of Squared Distances): {inertia:,.2f}")
    print(f"   • Silhouette Score:                  {silhouette:.4f}")
    print(f"   • Davies-Bouldin Index:              {davies_bouldin:.4f}")
    print(f"   • Calinski-Harabasz Score:           {calinski:,.2f}")
    print("-" * 60)

    # Step 7: Cluster Distribution & Profiles
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index().to_dict()
    cluster_profiles = {}
    for c_id in range(config.N_CLUSTERS):
        c_mask = cluster_labels == c_id
        c_df = data[c_mask]
        mean_features = c_df[config.RADAR_FEATURES].mean().round(3).to_dict()
        archetype = CLUSTER_ARCHETYPES.get(c_id, {"name": f"Cluster {c_id}", "mood": "Mixed", "icon": "🎵"})
        
        # Sample popular tracks in cluster
        top_tracks = c_df.nlargest(3, 'popularity')[['name', 'artists_clean', 'year']].to_dict(orient='records')
        
        cluster_profiles[str(c_id)] = {
            "name": archetype["name"],
            "mood": archetype["mood"],
            "icon": archetype["icon"],
            "track_count": int(c_mask.sum()),
            "mean_features": mean_features,
            "sample_tracks": top_tracks
        }

    # Step 8: Build Fast Catalog Index for Real-time Cosine Similarity
    print("[INFO] Building fast similarity lookup catalog index...")
    # Scale all tracks once for instant Cosine Distance lookups
    imputed_X = pipeline.named_steps['imputer'].transform(X)
    scaled_feature_matrix = pipeline.named_steps['scaler'].transform(imputed_X)

    metadata_cols = [c for c in ['id', 'name', 'artists', 'artists_clean', 'year', 'popularity', 'release_date'] if c in data.columns]
    catalog_metadata = data[metadata_cols].copy()
    catalog_metadata['cluster_label'] = cluster_labels

    catalog_index = {
        "metadata": catalog_metadata,
        "scaled_features": scaled_feature_matrix.astype(np.float32),
        "feature_names": features_present
    }

    # Step 9: Serialize Artifacts
    saved_model_path = save_pipeline(pipeline)
    print(f"[SUCCESS] Saved Pipeline Model to: {saved_model_path}")

    saved_catalog_path = save_catalog_index(catalog_index)
    print(f"[SUCCESS] Saved Fast Catalog Index to: {saved_catalog_path}")

    training_duration = round(time.time() - start_time, 2)

    metadata = {
        "model_version": config.MODEL_VERSION,
        "experiment_name": config.EXPERIMENT_NAME,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_tracks": total_tracks,
        "n_clusters": config.N_CLUSTERS,
        "features": features_present,
        "radar_features": config.RADAR_FEATURES,
        "metrics": {
            "inertia": inertia,
            "silhouette_score": silhouette,
            "davies_bouldin_index": davies_bouldin,
            "calinski_harabasz_score": calinski,
            "training_time_seconds": training_duration
        },
        "cluster_distribution": {str(k): int(v) for k, v in cluster_counts.items()},
        "cluster_profiles": cluster_profiles
    }
    saved_metadata_path = save_metadata(metadata)
    print(f"[SUCCESS] Saved Model Metadata to: {saved_metadata_path}")

    print("=" * 60)
    print(f"[DONE] Training Pipeline completed successfully in {training_duration}s!")
    print("=" * 60)
    return metadata

if __name__ == "__main__":
    run_training()
