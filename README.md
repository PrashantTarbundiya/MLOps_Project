# 🎵 Spotify AI Studio: Production MLOps Audio Intelligence & Recommender

> **MLOps Maturity Level 4**: Continuous Integration, Automated Continuous Training (CT), Continuous Deployment (CD), Statistical Data Drift Monitoring, Prometheus Telemetry, and Interactive Glassmorphism UI Showcase.

---

## 🌟 Overview

**Spotify AI Studio** is an end-to-end MLOps platform engineered to analyze, cluster, and recommend music across a massive catalog of **170,653 tracks** spanning 1921 to 2020. 

The system leverages unsupervised high-dimensional audio feature clustering (K-Means, $k=20$), cosine similarity vector projections, statistical data drift detection (Kolmogorov-Smirnov & Wasserstein Distance), and a real-time **FastAPI** microservice coupled with a luxury **dark-mode glassmorphism Web Studio**.

---

## 🏗️ Architecture & Pipeline Lifecycle

```
┌─────────────────┐       ┌────────────────────────┐       ┌───────────────────────┐
│   Spotify Data  │ ────> │  Feature Engineering   │ ────> │  K-Means Clustering   │
│  (170k+ Tracks) │ (DVC) │ (StandardScaler + Imp) │       │    (k=20 Archetypes)  │
└─────────────────┘       └────────────────────────┘       └───────────────────────┘
                                                                       │
                                                                       ▼
┌─────────────────┐       ┌────────────────────────┐       ┌───────────────────────┐
│  Prometheus &   │ <──── │   FastAPI Microservice │ <──── │   Catalog Index &     │
│  Drift Detector │       │   (REST API + Web UI)  │       │   Model Artifacts     │
└─────────────────┘       └────────────────────────┘       └───────────────────────┘
```

### 1. Data Versioning (DVC)
* Raw Spotify data (`data/data.csv`, `data_by_genres.csv`, `data_by_year.csv`, `data_by_artist.csv`) is tracked using **DVC** pointer files.
* Guaranteed immutability and data lineage across retraining cycles.

### 2. Preprocessing & Feature Engineering
* 15 continuous acoustic dimensions: `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `liveness`, `loudness`, `tempo`, `speechiness`, `popularity`, `key`, `mode`, `year`, `explicit`, `duration_ms`.
* Median Imputation (`AudioFeatureImputer`) handles nulls cleanly.
* `StandardScaler` maps all dimensions into zero-mean, unit-variance space.
* Artist name normalization and decade binning.

### 3. Model Architecture & Recommendation Engine
* **K-Means Clustering ($k=20$)**: Partitions songs into 20 sonic archetypes (e.g., *High-Voltage Rock*, *Midnight Jazz*, *Tropical Latin*, *Ambient Lo-Fi*).
* **Multi-Dimensional Cosine Similarity**: Real-time Euclidean/Cosine vector projections (`cdist`) across the scaled feature matrix.
* **Capabilities**:
  * **Seed Song Recommendation**: Input any track to discover closest sonic matches with match percentages.
  * **Playlist Centroid Recommender**: Multi-track centroid calculation for combined preference vectors.
  * **Sonic Mood Studio**: Real-time acoustic sliders (Danceability, Energy, Valence, Tempo) mapping to nearest clusters.

### 4. Continuous Monitoring (CM) & Drift Detection
* **Statistical Divergence**: Evaluates live inference batches against a 10,000-sample baseline using **Two-Sample Kolmogorov-Smirnov (KS) Hypothesis Tests** ($\alpha = 0.05$) and **Wasserstein Earth Mover's Distance**.
* **Data Quality Verification**: Automatic checks for missing cells, invalid ranges, and duplicated rows.
* **Live Drift Simulator**: Built-in simulator to test system alerts for sudden shifts (e.g. 1930s Acoustic shift or High-BPM EDM waves).

### 5. Telemetry & Observability
* Exposes `/metrics` endpoint with Prometheus-compatible counters and gauges:
  * `spotify_mlops_requests_total`
  * `spotify_mlops_recommendations_total`
  * `spotify_mlops_drift_checks_total`
  * `spotify_mlops_latency_avg_ms`
  * `spotify_mlops_uptime_seconds`

---

## 🚀 Quickstart & Local Execution

### 1. Clone & Set Up Environment
```bash
cd MLOps_Project
python -m pip install -r requirements.txt
```

### 2. Run Model Training Pipeline
Executes data ingestion, pipeline fitting, evaluation metrics, and fast catalog compilation:
```bash
python prediction_model/training_pipeline.py
```
*Output:*
```
[INFO] Loading dataset from: data/data.csv
[SUCCESS] Loaded 170,653 tracks successfully.
[INFO] Training K-Means Pipeline with K=20 clusters...
[SUCCESS] Pipeline fitting completed in ~11s.
[SUCCESS] Saved Pipeline Model to: trained_models/song_cluster_pipeline.joblib
[SUCCESS] Saved Fast Catalog Index to: trained_models/catalog_index.joblib
```

### 3. Run Automated Tests
```bash
python -m pytest tests/test_prediction.py -v
```
*All 8 tests validate pipeline loading, search, recommendations, cluster assignment, and drift detection.*

### 4. Launch Production Microservice & UI
```bash
python main.py
```
Open **[http://localhost:8005](http://localhost:8005)** in your browser to experience the **Spotify AI Studio**!

---

## 🌐 Interactive Web UI Showcase

The web application is located at `http://localhost:8005/`:
1. 🎵 **Recommender Tab**: Search any song across 170k+ tracks with live autocomplete, view its audio radar footprint, and explore top matches with similarity scores.
2. 🎛️ **Sonic Mood Studio**: Interactive audio sliders (Danceability, Energy, Valence, Tempo) and quick presets to discover songs matching custom moods.
3. 🔬 **MLOps Architecture**: Interactive interactive flowchart detailing every pipeline stage with live specifications.
4. 📊 **Cluster Archetypes**: Comprehensive gallery of the 20 sonic archetypes with mean features and sample tracks.
5. 🛡️ **Drift & Quality Center**: Live KS-test monitoring, p-values, Wasserstein divergence, and simulated drift triggers.
6. ⚡ **API Playground**: Live interactive REST runner for testing all endpoints with JSON response inspector.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive Spotify AI Studio Web UI |
| `GET` | `/api/health` | Healthcheck, uptime, track count, and model version |
| `GET` | `/api/model_info` | Model registry metadata, silhouette score, and cluster profiles |
| `GET` | `/api/search?q={query}` | Autocomplete search for track titles and artists |
| `POST` | `/api/recommend` | Single seed song similarity recommendations |
| `POST` | `/api/recommend_playlist` | Multi-track playlist centroid recommendations |
| `POST` | `/api/recommend_mood` | Acoustic feature slider-based recommendations |
| `POST` | `/api/predict_cluster` | Predicts cluster archetype for custom audio features |
| `GET` | `/api/drift_report` | Runs two-sample KS statistical drift tests |
| `POST` | `/api/simulate_drift` | Generates synthetic drift events for testing |
| `POST` | `/api/batch_predict` | Upload CSV for batch prediction and drift auditing |
| `GET` | `/metrics` | Prometheus telemetry metrics export |

---

## 🐳 Docker & Kubernetes Deployment

### Docker Build & Run
```bash
docker build -t spotify-mlops:latest .
docker run -p 8005:8005 spotify-mlops:latest
```

### Kubernetes Deployment
```bash
kubectl apply -f deployment.yml
kubectl apply -f service.yml
```

---

## 🧪 Technology Stack
* **Machine Learning**: `Scikit-Learn`, `KMeans`, `StandardScaler`, `SciPy` (Cosine & KS-Test)
* **API Microservice**: `FastAPI`, `Uvicorn`, `Pydantic v2`
* **Data Management**: `Pandas`, `NumPy`, `Joblib`, `DVC`
* **Continuous Integration**: `Pytest`, `GitHub Actions`
* **Observability**: `Prometheus`, `AudioDriftDetector`
* **Frontend Design**: Vanilla CSS (Glassmorphism, Dark Mode), Vanilla JS, Google Fonts Outfit
