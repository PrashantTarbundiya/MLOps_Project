import re
import ast
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def clean_artist_str(val):
    """Parses artist string representation like \"['Nirvana']\" into clean text \"Nirvana\"."""
    if not isinstance(val, str):
        return str(val)
    val = val.strip()
    if val.startswith('[') and val.endswith(']'):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return ", ".join([str(x).strip() for x in parsed if x])
        except Exception:
            # Fallback regex strip
            cleaned = re.sub(r"[\[\]\'\"]", "", val).strip()
            return cleaned
    return val

class AudioFeatureImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values in numerical audio feature columns with column median."""
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols
        self.median_dict = {}

    def fit(self, X, y=None):
        cols = self.feature_cols if self.feature_cols else X.select_dtypes(include=[np.number]).columns.tolist()
        self.median_dict = {col: float(X[col].median()) for col in cols if col in X.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for col, median_val in self.median_dict.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_val)
        return X

class DecadeExtractor(BaseEstimator, TransformerMixin):
    """Extracts decade categorical descriptor from year (e.g. 1994 -> '1990s')."""
    def __init__(self, year_col='year'):
        self.year_col = year_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.year_col in X.columns:
            X['decade'] = (X[self.year_col] // 10 * 10).astype(str) + 's'
        return X

class ArtistCleaner(BaseEstimator, TransformerMixin):
    """Normalizes string-formatted artist lists into clean readable names."""
    def __init__(self, artist_col='artists'):
        self.artist_col = artist_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.artist_col in X.columns:
            X[self.artist_col] = X[self.artist_col].apply(clean_artist_str)
        return X

CLUSTER_ARCHETYPES = {
    0: {"name": "Acoustic Serenity & Ballads", "mood": "Calm, Soft & Introspective", "icon": "Acoustic"},
    1: {"name": "High-Voltage Rock & Grunge", "mood": "Energetic, Raw & Loud", "icon": "Rock"},
    2: {"name": "Golden Age Pop & Dance", "mood": "Upbeat, Rhythmic & Catchy", "icon": "Pop"},
    3: {"name": "Vintage Classical & Nostalgia", "mood": "Historical, Orchestral & Pure", "icon": "Classical"},
    4: {"name": "Midnight Jazz & Blues", "mood": "Melancholic, Warm & Smooth", "icon": "Jazz"},
    5: {"name": "Modern Bass & Hip-Hop", "mood": "Punchy, Dynamic & Contemporary", "icon": "Hip-Hop"},
    6: {"name": "Ambient Chillwave & Lo-Fi", "mood": "Spacious, Dreamy & Meditative", "icon": "Chillwave"},
    7: {"name": "Retro 80s Synth & Disco", "mood": "Groovy, Bright & Nostalgic", "icon": "Synth"},
    8: {"name": "Acoustic Folk & Indie Roots", "mood": "Organic, Earthy & Soulful", "icon": "Folk"},
    9: {"name": "High-Tempo Club & Electronic", "mood": "Fast-Paced, Driving & Euphoric", "icon": "Electronic"},
    10: {"name": "Vocal Harmony & Classic Choral", "mood": "Lyrical, Reverberant & Gentle", "icon": "Vocal"},
    11: {"name": "Atmospheric Soundtrack & Cinematic", "mood": "Dramatic, Panoramic & Slow", "icon": "Soundtrack"},
    12: {"name": "R&B Grooves & Neo-Soul", "mood": "Sensual, Flowing & Harmonic", "icon": "R&B"},
    13: {"name": "Alternative Wave & Punk", "mood": "Rebellious, Punchy & Edgy", "icon": "Punk"},
    14: {"name": "Tropical Latin & Reggaeton", "mood": "Danceable, Warm & Percussive", "icon": "Latin"},
    15: {"name": "Dark Ambient & Industrial", "mood": "Heavy, Textured & Intense", "icon": "Industrial"},
    16: {"name": "Upbeat Country & Americana", "mood": "Storytelling, Twangy & Cheerful", "icon": "Country"},
    17: {"name": "Modern Minimalist Pop", "mood": "Clean, Stripped-Down & Trendy", "icon": "Minimalist"},
    18: {"name": "Psychedelic & Experimental Soundscapes", "mood": "Mind-Bending, Spatial & Free", "icon": "Psychedelic"},
    19: {"name": "Epic Orchestral Anthems", "mood": "Triumphant, Grand & Symphonic", "icon": "Orchestral"},
}

def get_cluster_info(cluster_id: int):
    """Returns human-readable archetype information for a cluster ID."""
    return CLUSTER_ARCHETYPES.get(cluster_id, {
        "name": f"Sonic Cluster #{cluster_id}",
        "mood": "Dynamic Audio Profile",
        "icon": "Track"
    })