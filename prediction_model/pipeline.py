from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp

def create_clustering_pipeline(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE, n_init=config.N_INIT):
    """
    Builds the production clustering pipeline:
    1. AudioFeatureImputer: guarantees clean input without NaNs.
    2. StandardScaler: scales multi-dimensional audio metrics into zero-mean, unit-variance space.
    3. KMeans: groups tracks into sonic archetypes.
    """
    pipeline = Pipeline([
        ('imputer', pp.AudioFeatureImputer(feature_cols=config.AUDIO_FEATURES)),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=300
        ))
    ])
    return pipeline

# Default initialized pipeline instance
song_cluster_pipeline = create_clustering_pipeline()
