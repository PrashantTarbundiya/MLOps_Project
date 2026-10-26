import os
import json
import joblib
import pandas as pd
from prediction_model.config import config
from prediction_model.processing.preprocessing import clean_artist_str

def load_dataset(file_path=None, nrows=None):
    """Loads track dataset, applies basic cleaning on string artist names."""
    path = file_path if file_path else config.DATA_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path, nrows=nrows)
    if 'artists' in df.columns:
        df['artists_clean'] = df['artists'].apply(clean_artist_str)
    else:
        df['artists_clean'] = "Unknown Artist"
        
    if 'year' in df.columns:
        df['decade'] = (df['year'] // 10 * 10).astype(str) + 's'
        
    return df

def load_genre_data():
    """Loads aggregated genre dataset."""
    if os.path.exists(config.GENRE_FILE):
        return pd.read_csv(config.GENRE_FILE)
    return pd.DataFrame()

def load_year_data():
    """Loads aggregated year dataset."""
    if os.path.exists(config.YEAR_FILE):
        return pd.read_csv(config.YEAR_FILE)
    return pd.DataFrame()

def save_pipeline(pipeline_to_save, filename=None):
    """Serializes the trained pipeline artifact with joblib."""
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    target_path = os.path.join(config.TRAINED_MODEL_DIR, filename if filename else config.MODEL_NAME)
    joblib.dump(pipeline_to_save, target_path)
    return target_path

def load_pipeline(filename=None):
    """Loads serialized pipeline artifact."""
    target_path = os.path.join(config.TRAINED_MODEL_DIR, filename if filename else config.MODEL_NAME)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Trained model not found at {target_path}. Please run training_pipeline.py first.")
    return joblib.load(target_path)

def save_metadata(metadata_dict, filename="model_metadata.json"):
    """Saves model metrics, parameters, and metadata as JSON."""
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    target_path = os.path.join(config.TRAINED_MODEL_DIR, filename)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=4)
    return target_path

def load_metadata(filename="model_metadata.json"):
    """Loads model metadata from JSON."""
    target_path = os.path.join(config.TRAINED_MODEL_DIR, filename)
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_catalog_index(catalog_data, filename="catalog_index.joblib"):
    """Saves pre-computed normalized audio matrices and fast search index."""
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    target_path = os.path.join(config.TRAINED_MODEL_DIR, filename)
    joblib.dump(catalog_data, target_path, compress=3)
    return target_path

def load_catalog_index(filename="catalog_index.joblib"):
    """Loads catalog index for fast cosine similarity lookups."""
    target_path = os.path.join(config.TRAINED_MODEL_DIR, filename)
    if os.path.exists(target_path):
        return joblib.load(target_path)
    return None
