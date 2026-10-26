/**
 * Spotify AI Studio - Frontend Application Engine
 * Handles live search, recommendation rendering, mood sculpting,
 * MLOps pipeline interactive exploration, drift monitoring & API testing.
 */

// Global State
let currentSeedTrack = null;
let debounceTimer = null;
let selectedEndpoint = 'health';
let modelMetadata = null;

// Audio feature definitions for radar rendering
const RADAR_CONFIG = [
    { key: 'danceability', label: 'Danceability', color: '#1db954' },
    { key: 'energy', label: 'Energy', color: '#f59e0b' },
    { key: 'valence', label: 'Valence (Mood)', color: '#38bdf8' },
    { key: 'acousticness', label: 'Acousticness', color: '#10b981' },
    { key: 'instrumentalness', label: 'Instrumental', color: '#818cf8' },
    { key: 'liveness', label: 'Liveness', color: '#f43f5e' }
];

// Pipeline Node Specifications
const PIPELINE_SPECS = {
    data: {
        title: "1. Data Versioning & S3/DVC Storage",
        desc: "The Spotify dataset contains 170,653 audio tracks from 1921 to 2020. Data files are tracked via DVC (.dvc) pointers and versioned across pipeline iterations to guarantee 100% reproducibility.",
        specs: [
            { label: "Dataset Size", val: "170,653 Tracks" },
            { label: "Data Versioning", val: "DVC 3.x" },
            { label: "Remote Storage", val: "AWS S3 / DVC Remote" },
            { label: "Raw Features", val: "19 Acoustic Columns" }
        ]
    },
    prep: {
        title: "2. Audio Preprocessing & Feature Scaling",
        desc: "Raw audio features are cleaned, imputed (AudioFeatureImputer), and standardized into zero-mean, unit-variance space using StandardScaler. Decades and artists are normalized.",
        specs: [
            { label: "Imputation", val: "Column-wise Median" },
            { label: "Scaler", val: "StandardScaler" },
            { label: "Features Scaled", val: "15 Numerical Dimensions" },
            { label: "Artist Cleaning", val: "Literal Eval / Regex" }
        ]
    },
    model: {
        title: "3. Unsupervised K-Means Cluster Engine",
        desc: "Partitions the 15-dimensional audio manifold into 20 sonic archetypes. Computes centroid distances and fits the cosine similarity recommendation index.",
        specs: [
            { label: "Algorithm", val: "K-Means (k=20)" },
            { label: "Metric Space", val: "Cosine Distance (cdist)" },
            { label: "Optimization", val: "Inertia & Silhouette" },
            { label: "Random State", val: "42 (Deterministic)" }
        ]
    },
    registry: {
        title: "4. Model Artifact Registry & Versioning",
        desc: "Fitted pipelines and fast catalog similarity indexes are serialized with joblib and stored with full metadata JSON tracking hyperparameters and evaluation scores.",
        specs: [
            { label: "Model Artifact", val: "song_cluster_pipeline.joblib" },
            { label: "Catalog Index", val: "catalog_index.joblib" },
            { label: "Model Version", val: "v1.0.0" },
            { label: "Registry Storage", val: "Local / MLflow Tracking" }
        ]
    },
    api: {
        title: "5. Production FastAPI Microservice",
        desc: "High-performance asynchronous REST API serving real-time song recommendations, mood projection, and cluster inferences with sub-25ms response latency.",
        specs: [
            { label: "Framework", val: "FastAPI + Uvicorn" },
            { label: "Containerization", val: "Docker / Alpine Python" },
            { label: "Deployment", val: "Kubernetes (EKS / Minikube)" },
            { label: "Concurrency", val: "Asynchronous ASGI" }
        ]
    },
    monitoring: {
        title: "6. Continuous Drift & Data Quality Monitoring",
        desc: "Evaluates production inference traffic against 10,000 baseline training records using two-sample Kolmogorov-Smirnov (KS) tests and Wasserstein Earth Mover distances.",
        specs: [
            { label: "Drift Test", val: "Two-sample KS-Test" },
            { label: "Significance α", val: "p < 0.05 Threshold" },
            { label: "Metrics Export", val: "/metrics (Prometheus)" },
            { label: "Quality Checks", val: "Missing, Range, Duplicates" }
        ]
    }
};

// Mood Presets Data
const MOOD_PRESETS = {
    workout: { danceability: 0.85, energy: 0.92, valence: 0.75, acousticness: 0.04, instrumentalness: 0.02, tempo: 135, popularity: 75 },
    focus: { danceability: 0.35, energy: 0.30, valence: 0.35, acousticness: 0.82, instrumentalness: 0.85, tempo: 90, popularity: 45 },
    party: { danceability: 0.90, energy: 0.88, valence: 0.90, acousticness: 0.05, instrumentalness: 0.01, tempo: 125, popularity: 80 },
    chill: { danceability: 0.55, energy: 0.40, valence: 0.45, acousticness: 0.65, instrumentalness: 0.25, tempo: 100, popularity: 60 },
    rock: { danceability: 0.45, energy: 0.95, valence: 0.50, acousticness: 0.02, instrumentalness: 0.20, tempo: 140, popularity: 70 }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSearch();
    initMoodSliders();
    initAPIExplorer();
    loadModelMetadata();
    loadRealDriftReport();

    // Default seed recommendation on load
    quickSelect('Smells Like Teen Spirit', 1991);
});

// --- Tab Navigation ---
function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const targetTab = tab.getAttribute('data-tab');
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            const targetContent = document.getElementById(`tab-${targetTab}`);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });
}

// --- Live Search & Autocomplete ---
function initSearch() {
    const searchInput = document.getElementById('track-search-input');
    const dropdown = document.getElementById('search-dropdown');
    const spinner = document.getElementById('search-spinner');

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        clearTimeout(debounceTimer);

        if (query.length < 2) {
            dropdown.classList.add('hidden');
            dropdown.innerHTML = '';
            return;
        }

        spinner.classList.remove('hidden');
        debounceTimer = setTimeout(async () => {
            try {
                const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=8`);
                const data = await res.json();
                renderSearchDropdown(data);
            } catch (err) {
                console.error("Search error:", err);
            } finally {
                spinner.classList.add('hidden');
            }
        }, 250);
    });

    // Close dropdown on click outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.classList.add('hidden');
        }
    });
}

function renderSearchDropdown(items) {
    const dropdown = document.getElementById('search-dropdown');
    if (!items || items.length === 0) {
        dropdown.innerHTML = '<div class="dropdown-item"><span class="item-title">No matching tracks found</span></div>';
        dropdown.classList.remove('hidden');
        return;
    }

    dropdown.innerHTML = items.map(item => `
        <div class="dropdown-item" onclick="selectFromDropdown('${escapeHtml(item.name)}', ${item.year})">
            <div>
                <div class="item-title">${escapeHtml(item.name)}</div>
                <div class="item-artist">${escapeHtml(item.artist)} (${item.year})</div>
            </div>
            <span class="cluster-pill">${item.cluster_name}</span>
        </div>
    `).join('');
    dropdown.classList.remove('hidden');
}

function selectFromDropdown(name, year) {
    document.getElementById('track-search-input').value = `${name} (${year})`;
    document.getElementById('search-dropdown').classList.add('hidden');
    fetchRecommendations(name, year);
}

function quickSelect(name, year) {
    document.getElementById('track-search-input').value = `${name} (${year})`;
    fetchRecommendations(name, year);
}

// --- Fetch & Render Recommendations ---
async function fetchRecommendations(name, year) {
    const nRecs = parseInt(document.getElementById('n-recs-select').value) || 10;
    const grid = document.getElementById('recommendations-grid');
    grid.innerHTML = '<div class="empty-state"><div class="spinner" style="margin: 0 auto 16px;"></div><h4>Computing high-dimensional acoustic cosine similarities...</h4></div>';

    try {
        const res = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                song_name: name,
                year: year,
                n_recommendations: nRecs
            })
        });

        if (!res.ok) {
            throw new Error(await res.text());
        }

        const data = await res.json();
        renderSeedSpotlight(data.seed_track);
        renderRecommendations(data.recommendations);
    } catch (err) {
        grid.innerHTML = `<div class="empty-state"><h4>Track not found</h4><p>${err.message}</p></div>`;
    }
}

function renderSeedSpotlight(seed) {
    const spotlight = document.getElementById('seed-spotlight');
    if (!seed) {
        spotlight.classList.add('hidden');
        return;
    }

    spotlight.classList.remove('hidden');
    document.getElementById('seed-title').textContent = seed.name;
    document.getElementById('seed-artist').textContent = seed.artist;
    document.getElementById('seed-year').textContent = seed.year;
    document.getElementById('seed-pop').textContent = `Popularity: ${seed.popularity}/100`;
    document.getElementById('seed-cluster-badge').textContent = seed.cluster_name;

    // Render Audio Radar Bars
    const radarContainer = document.getElementById('seed-radar-bars');
    const radar = seed.radar_features || {};
    radarContainer.innerHTML = RADAR_CONFIG.map(cfg => {
        const val = radar[cfg.key] !== undefined ? radar[cfg.key] : 0.5;
        const pct = Math.round(val * 100);
        return `
            <div class="radar-row">
                <div class="radar-label">
                    <span>${cfg.label}</span>
                    <span>${pct}%</span>
                </div>
                <div class="radar-bar-bg">
                    <div class="radar-bar-fill" style="width: ${pct}%; background: ${cfg.color};"></div>
                </div>
            </div>
        `;
    }).join('');
}

function renderRecommendations(recs) {
    const grid = document.getElementById('recommendations-grid');
    if (!recs || recs.length === 0) {
        grid.innerHTML = '<div class="empty-state"><div class="empty-icon">🔍</div><h4>No close recommendations found</h4></div>';
        return;
    }

    grid.innerHTML = recs.map(track => {
        const radar = track.radar_features || {};
        const miniBars = RADAR_CONFIG.slice(0, 4).map(cfg => {
            const val = radar[cfg.key] !== undefined ? radar[cfg.key] : 0.5;
            return `
                <div class="radar-row" style="margin-bottom: 2px;">
                    <div class="radar-label" style="font-size: 10px;">
                        <span>${cfg.label}</span>
                        <span>${Math.round(val * 100)}%</span>
                    </div>
                    <div class="radar-bar-bg" style="height: 4px;">
                        <div class="radar-bar-fill" style="width: ${Math.round(val * 100)}%; background: ${cfg.color};"></div>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div class="track-card">
                <div>
                    <div class="card-top">
                        <span class="cluster-pill">${escapeHtml(track.cluster_name)}</span>
                        <span class="match-badge">${track.similarity_score}% MATCH</span>
                    </div>
                    <h4 class="track-title">${escapeHtml(track.name)}</h4>
                    <p class="track-artist">${escapeHtml(track.artist)} &bull; ${track.year}</p>
                    <div style="margin-top: 10px;">
                        ${miniBars}
                    </div>
                </div>
                <div class="track-card-footer">
                    <span style="color: var(--text-dim);">Popularity ${track.popularity}/100</span>
                    <button class="btn btn-secondary btn-sm" onclick="quickSelect('${escapeHtml(track.name)}', ${track.year})">
                        Explore Sonic Match ➜
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

// --- Sonic Mood Studio ---
function initMoodSliders() {
    const sliderKeys = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'tempo', 'popularity'];
    sliderKeys.forEach(key => {
        const slider = document.getElementById(`slider-${key}`);
        const display = document.getElementById(`val-${key}`);
        if (slider && display) {
            slider.addEventListener('input', (e) => {
                const val = e.target.value;
                display.textContent = key === 'tempo' ? `${val} BPM` : parseFloat(val).toFixed(2);
            });
        }
    });

    document.getElementById('btn-reset-mood').addEventListener('click', () => {
        applyMoodPreset('chill');
    });

    document.getElementById('btn-generate-mood').addEventListener('click', () => {
        generateMoodRecommendations();
    });

    // Run initial mood recommendation
    generateMoodRecommendations();
}

function applyMoodPreset(presetName) {
    const preset = MOOD_PRESETS[presetName];
    if (!preset) return;

    for (const [key, val] of Object.entries(preset)) {
        const slider = document.getElementById(`slider-${key}`);
        const display = document.getElementById(`val-${key}`);
        if (slider && display) {
            slider.value = val;
            display.textContent = key === 'tempo' ? `${val} BPM` : parseFloat(val).toFixed(2);
        }
    }
    generateMoodRecommendations();
}

async function generateMoodRecommendations() {
    const payload = {
        danceability: parseFloat(document.getElementById('slider-danceability').value),
        energy: parseFloat(document.getElementById('slider-energy').value),
        valence: parseFloat(document.getElementById('slider-valence').value),
        acousticness: parseFloat(document.getElementById('slider-acousticness').value),
        instrumentalness: parseFloat(document.getElementById('slider-instrumentalness').value),
        tempo: parseFloat(document.getElementById('slider-tempo').value),
        popularity: parseFloat(document.getElementById('slider-popularity').value),
        n_recommendations: 8
    };

    const grid = document.getElementById('mood-results-grid');
    grid.innerHTML = '<div class="empty-state"><div class="spinner" style="margin: 0 auto 12px;"></div><p>Mapping mood vector onto audio manifold...</p></div>';

    try {
        const res = await fetch('/api/recommend_mood', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.target_cluster) {
            document.getElementById('mood-cluster-name').textContent = data.target_cluster.name;
            document.getElementById('mood-cluster-desc').textContent = `${data.target_cluster.mood} (Cluster #${data.target_cluster.cluster_id})`;
        }

        renderMoodRecommendations(data.recommendations);
    } catch (err) {
        grid.innerHTML = `<div class="empty-state"><p>Error: ${err.message}</p></div>`;
    }
}

function renderMoodRecommendations(recs) {
    const grid = document.getElementById('mood-results-grid');
    if (!recs || recs.length === 0) {
        grid.innerHTML = '<div class="empty-state"><p>No matching tracks for this mood profile.</p></div>';
        return;
    }

    grid.innerHTML = recs.map(track => `
        <div class="track-card">
            <div>
                <div class="card-top">
                    <span class="cluster-pill">${escapeHtml(track.cluster_name)}</span>
                    <span class="match-badge">${track.similarity_score}% MATCH</span>
                </div>
                <h4 class="track-title">${escapeHtml(track.name)}</h4>
                <p class="track-artist">${escapeHtml(track.artist)} &bull; ${track.year}</p>
            </div>
            <div class="track-card-footer">
                <span style="color: var(--text-dim);">Popularity ${track.popularity}/100</span>
                <button class="btn btn-secondary btn-sm" onclick="quickSelect('${escapeHtml(track.name)}', ${track.year})">
                    Explore ➜
                </button>
            </div>
        </div>
    `).join('');
}

// --- Pipeline Node Detail Inspector ---
function showNodeDetail(nodeKey) {
    document.querySelectorAll('.pipe-node').forEach(n => n.classList.remove('active'));
    event.currentTarget.classList.add('active');

    const spec = PIPELINE_SPECS[nodeKey];
    if (!spec) return;

    document.getElementById('detail-title').textContent = spec.title;
    document.getElementById('detail-desc').textContent = spec.desc;

    const specsGrid = document.getElementById('detail-specs');
    specsGrid.innerHTML = spec.specs.map(s => `
        <div class="spec-item">
            <div class="spec-label">${s.label}</div>
            <div class="spec-val">${s.val}</div>
        </div>
    `).join('');
}

// --- Load Model Metadata & Archetypes ---
async function loadModelMetadata() {
    try {
        const res = await fetch('/api/model_info');
        if (!res.ok) return;
        modelMetadata = await res.json();

        // Update Nav Metric
        if (modelMetadata.total_tracks) {
            document.getElementById('metric-tracks').textContent = `${modelMetadata.total_tracks.toLocaleString()} Tracks`;
        }

        renderClusterArchetypes(modelMetadata.cluster_profiles);
    } catch (err) {
        console.error("Metadata load error:", err);
    }
}

function renderClusterArchetypes(profiles) {
    const container = document.getElementById('clusters-grid');
    if (!profiles) return;

    container.innerHTML = Object.entries(profiles).map(([cid, prof]) => {
        const radar = prof.mean_features || {};
        const bars = Object.entries(radar).map(([k, v]) => `
            <div class="radar-row">
                <div class="radar-label" style="font-size: 11px;">
                    <span style="text-transform: capitalize;">${k}</span>
                    <span>${Math.round(v * 100)}%</span>
                </div>
                <div class="radar-bar-bg" style="height: 4px;">
                    <div class="radar-bar-fill" style="width: ${Math.round(v * 100)}%;"></div>
                </div>
            </div>
        `).join('');

        const sampleTracks = (prof.sample_tracks || []).map(s => `
            <div class="sample-song">🎵 ${escapeHtml(s.name)} - ${escapeHtml(s.artists_clean)} (${s.year})</div>
        `).join('');

        return `
            <div class="cluster-card">
                <div class="cluster-card-header">
                    <span class="cluster-card-title">${escapeHtml(prof.name)}</span>
                    <span class="node-badge">#${cid}</span>
                </div>
                <div class="cluster-card-mood">${escapeHtml(prof.mood)} &bull; ${prof.track_count.toLocaleString()} tracks</div>
                <div class="cluster-radar-bars">
                    ${bars}
                </div>
                <div class="cluster-samples">
                    <div class="cluster-samples-label">Popular Examples</div>
                    ${sampleTracks}
                </div>
            </div>
        `;
    }).join('');
}

// --- Data Drift & Quality Center ---
async function loadRealDriftReport() {
    const banner = document.getElementById('drift-status-banner');
    banner.className = 'drift-status-banner';
    document.getElementById('drift-status-title').textContent = "Analyzing Real-Time Data Drift...";

    try {
        const res = await fetch('/api/drift_report');
        const report = await res.json();
        renderDriftReport(report);
    } catch (err) {
        document.getElementById('drift-status-title').textContent = "Drift Check Offline";
    }
}

async function runDriftSimulation(driftType) {
    const banner = document.getElementById('drift-status-banner');
    document.getElementById('drift-status-title').textContent = `Simulating Drift: ${driftType}...`;

    try {
        const res = await fetch(`/api/simulate_drift?drift_type=${driftType}`, { method: 'POST' });
        const data = await res.json();
        renderDriftReport(data.report, data.simulation_scenario);
    } catch (err) {
        console.error("Drift simulation error:", err);
    }
}

function renderDriftReport(report, scenario = null) {
    const banner = document.getElementById('drift-status-banner');
    const title = document.getElementById('drift-status-title');
    const desc = document.getElementById('drift-status-desc');

    banner.className = 'drift-status-banner';
    if (report.status === 'DRIFT_CRITICAL') {
        banner.classList.add('critical');
    } else if (report.status === 'DRIFT_WARNING') {
        banner.classList.add('warning');
    }

    title.textContent = `Status: ${report.status}${scenario ? ` (Simulation: ${scenario})` : ''}`;
    desc.textContent = report.status_description;

    document.getElementById('stat-drift-share').textContent = `${report.drift_share_percent}%`;
    document.getElementById('stat-drifted-count').textContent = `${report.drifted_features_count} / ${report.total_features_tested}`;
    document.getElementById('stat-quality').textContent = `${100 - report.data_quality.missing_ratio_percent}%`;

    // Render table rows
    const tbody = document.getElementById('drift-table-body');
    tbody.innerHTML = Object.entries(report.feature_drifts).map(([feat, d]) => `
        <tr>
            <td style="color: #fff; font-weight: 600;">${feat}</td>
            <td>
                <span class="badge ${d.has_drifted ? 'badge-danger' : 'badge-success'}">
                    ${d.has_drifted ? 'DRIFT' : 'HEALTHY'}
                </span>
            </td>
            <td>${d.ks_statistic}</td>
            <td style="color: ${d.has_drifted ? '#fb7185' : '#1ed760'};">${d.p_value}</td>
            <td>${d.wasserstein_distance}</td>
            <td>${d.baseline_mean}</td>
            <td>${d.current_mean}</td>
        </tr>
    `).join('');
}

// --- API Explorer & Playground ---
const API_ENDPOINTS = {
    health: {
        method: 'GET',
        url: '/api/health',
        payload: ''
    },
    recommend: {
        method: 'POST',
        url: '/api/recommend',
        payload: JSON.stringify({
            song_name: "Smells Like Teen Spirit",
            year: 1991,
            n_recommendations: 3
        }, null, 2)
    },
    mood: {
        method: 'POST',
        url: '/api/recommend_mood',
        payload: JSON.stringify({
            danceability: 0.85,
            energy: 0.90,
            valence: 0.75,
            acousticness: 0.05,
            n_recommendations: 3
        }, null, 2)
    },
    cluster: {
        method: 'POST',
        url: '/api/predict_cluster',
        payload: JSON.stringify({
            danceability: 0.7,
            energy: 0.8,
            valence: 0.8,
            acousticness: 0.1,
            tempo: 128.0
        }, null, 2)
    },
    drift: {
        method: 'GET',
        url: '/api/drift_report',
        payload: ''
    },
    metrics: {
        method: 'GET',
        url: '/metrics',
        payload: ''
    }
};

function initAPIExplorer() {
    selectEndpoint('health');

    document.getElementById('btn-send-api').addEventListener('click', async () => {
        const ep = API_ENDPOINTS[selectedEndpoint];
        const statusBadge = document.getElementById('resp-status-badge');
        const latencyDisplay = document.getElementById('resp-latency');
        const responseBody = document.getElementById('api-response-body');

        responseBody.textContent = "Executing API request...";
        const startTime = performance.now();

        try {
            const options = { method: ep.method };
            if (ep.method === 'POST') {
                options.headers = { 'Content-Type': 'application/json' };
                options.body = document.getElementById('api-payload-input').value;
            }

            const res = await fetch(ep.url, options);
            const elapsed = Math.round(performance.now() - startTime);
            latencyDisplay.textContent = `Latency: ${elapsed} ms`;

            statusBadge.className = `badge ${res.ok ? 'badge-success' : 'badge-danger'}`;
            statusBadge.textContent = `${res.status} ${res.statusText}`;

            const contentType = res.headers.get("content-type") || "";
            if (contentType.includes("application/json")) {
                const data = await res.json();
                responseBody.textContent = JSON.stringify(data, null, 2);
            } else {
                const text = await res.text();
                responseBody.textContent = text;
            }
        } catch (err) {
            statusBadge.className = 'badge badge-danger';
            statusBadge.textContent = 'FAILED';
            responseBody.textContent = `Error: ${err.message}`;
        }
    });
}

function selectEndpoint(epKey) {
    selectedEndpoint = epKey;
    const ep = API_ENDPOINTS[epKey];

    document.querySelectorAll('.endpoint-btn').forEach(b => b.classList.remove('active'));
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }

    const methodTag = document.getElementById('req-method-tag');
    methodTag.textContent = ep.method;
    methodTag.className = `http-badge ${ep.method.toLowerCase()}`;

    const payloadInput = document.getElementById('api-payload-input');
    payloadInput.value = ep.payload;
    payloadInput.disabled = ep.method === 'GET';
}

// Utility: HTML Sanitizer
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
