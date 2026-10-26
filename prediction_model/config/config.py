import os
import pathlib

# Directory resolution
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Data directory and file paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_FILE = os.path.join(DATA_DIR, "data.csv")
GENRE_FILE = os.path.join(DATA_DIR, "data_by_genres.csv")
YEAR_FILE = os.path.join(DATA_DIR, "data_by_year.csv")
ARTIST_FILE = os.path.join(DATA_DIR, "data_by_artist.csv")
GENRES_COMBINED_FILE = os.path.join(DATA_DIR, "data_w_genres.csv")

# Model artifacts directory
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, "trained_models")
DATASETS_DIR = os.path.join(PACKAGE_ROOT, "datasets")
MODEL_NAME = "song_cluster_pipeline.joblib"
MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_NAME)
METADATA_PATH = os.path.join(TRAINED_MODEL_DIR, "model_metadata.json")
CATALOG_INDEX_PATH = os.path.join(TRAINED_MODEL_DIR, "catalog_index.joblib")
BASELINE_DATA_PATH = os.path.join(DATASETS_DIR, "baseline_features.csv")

# Feature definitions
# Core audio features used for similarity, mood vectors, and clustering
AUDIO_FEATURES = [
    'valence',
    'year',
    'acousticness',
    'danceability',
    'duration_ms',
    'energy',
    'explicit',
    'instrumentalness',
    'key',
    'liveness',
    'loudness',
    'mode',
    'popularity',
    'speechiness',
    'tempo'
]

# Primary normalized features for cosine similarity & sonic radar visualization
RADAR_FEATURES = [
    'acousticness',
    'danceability',
    'energy',
    'instrumentalness',
    'liveness',
    'valence'
]

METADATA_COLS = ['id', 'name', 'artists', 'year', 'popularity', 'release_date']

# Model hyperparameters
N_CLUSTERS = 20
RANDOM_STATE = 42
N_INIT = 10
SAMPLE_TRAIN_SIZE = 170000  # Full or sampled dataset size for training

# MLOps & Experiment Tracking
EXPERIMENT_NAME = "spotify_audio_clustering_recommendation"
MODEL_VERSION = "1.0.0"
DRIFT_P_VALUE_THRESHOLD = 0.05
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
