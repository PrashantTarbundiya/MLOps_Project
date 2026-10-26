import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from prediction_model.config import config

class AudioDriftDetector:
    """
    Statistical Data Drift and Data Quality engine for Spotify Audio Feature Pipelines.
    Utilizes two-sample Kolmogorov-Smirnov (KS) tests and Wasserstein Distance
    to detect distribution divergence between baseline training data and live inference traffic.
    """
    def __init__(self, baseline_path=None, p_value_threshold=0.05):
        self.baseline_path = baseline_path if baseline_path else config.BASELINE_DATA_PATH
        self.p_value_threshold = p_value_threshold
        self._baseline_df = None

    def load_baseline(self):
        if self._baseline_df is None:
            if not os.path.exists(self.baseline_path):
                # Fallback: sample from main dataset if baseline file doesn't exist yet
                raw_df = pd.read_csv(config.DATA_FILE, nrows=10000)
                feats = [c for c in config.AUDIO_FEATURES if c in raw_df.columns]
                self._baseline_df = raw_df[feats].copy()
                os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
                self._baseline_df.to_csv(self.baseline_path, index=False)
            else:
                self._baseline_df = pd.read_csv(self.baseline_path)
        return self._baseline_df

    def detect_drift(self, current_df: pd.DataFrame):
        """
        Runs two-sample KS test and Wasserstein distance for each audio feature.
        Returns detailed drift diagnostics, p-values, and overall status.
        """
        baseline = self.load_baseline()
        features = [col for col in config.AUDIO_FEATURES if col in baseline.columns and col in current_df.columns]

        drift_results = {}
        drifted_features_count = 0

        for feat in features:
            base_col = baseline[feat].dropna()
            curr_col = current_df[feat].dropna()

            if len(curr_col) < 5:
                continue

            # Two-sample Kolmogorov-Smirnov Test
            ks_stat, p_val = ks_2samp(base_col, curr_col)
            w_dist = wasserstein_distance(base_col, curr_col)

            has_drifted = bool(p_val < self.p_value_threshold)
            if has_drifted:
                drifted_features_count += 1

            drift_results[feat] = {
                "feature": feat,
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": float(f"{p_val:.4e}"),
                "wasserstein_distance": round(float(w_dist), 4),
                "has_drifted": has_drifted,
                "baseline_mean": round(float(base_col.mean()), 3),
                "current_mean": round(float(curr_col.mean()), 3),
                "baseline_std": round(float(base_col.std()), 3),
                "current_std": round(float(curr_col.std()), 3),
            }

        total_tested = max(1, len(drift_results))
        drift_share = round((drifted_features_count / total_tested) * 100, 1)

        if drift_share >= 50:
            status = "DRIFT_CRITICAL"
            status_desc = "Significant data drift detected across multiple audio features. Model retraining recommended."
        elif drift_share > 20:
            status = "DRIFT_WARNING"
            status_desc = "Moderate feature drift detected. Monitor pipeline closely."
        else:
            status = "HEALTHY"
            status_desc = "Audio feature distributions conform to the baseline training distribution."

        # Quality checks
        missing_vals = int(current_df[features].isna().sum().sum())
        total_cells = max(1, current_df[features].size)
        missing_ratio = round((missing_vals / total_cells) * 100, 2)

        return {
            "status": status,
            "status_description": status_desc,
            "drift_share_percent": drift_share,
            "drifted_features_count": drifted_features_count,
            "total_features_tested": total_tested,
            "p_value_threshold": self.p_value_threshold,
            "current_batch_size": len(current_df),
            "data_quality": {
                "missing_values_count": missing_vals,
                "missing_ratio_percent": missing_ratio,
                "integrity_status": "PASSED" if missing_ratio < 1.0 else "WARNING"
            },
            "feature_drifts": drift_results
        }

    def generate_synthetic_drift_batch(self, drift_type="retro_acoustic", n_samples=200):
        """Generates a synthetic inference batch to demonstrate drift detection in the UI."""
        baseline = self.load_baseline().copy()
        sample = baseline.sample(n=min(n_samples, len(baseline)), replace=True).copy()

        if drift_type == "retro_acoustic":
            # Simulate shift to 1930s classical acoustic music
            sample['acousticness'] = np.clip(sample['acousticness'] * 1.8 + 0.3, 0.0, 1.0)
            sample['energy'] = np.clip(sample['energy'] * 0.4, 0.0, 1.0)
            sample['loudness'] = sample['loudness'] - 12.0
            sample['year'] = np.random.randint(1925, 1945, size=len(sample))
        elif drift_type == "high_bpm_edm":
            # Simulate high-energy, synthetic EDM wave
            sample['energy'] = np.clip(sample['energy'] * 1.5 + 0.2, 0.0, 1.0)
            sample['danceability'] = np.clip(sample['danceability'] * 1.4 + 0.1, 0.0, 1.0)
            sample['tempo'] = sample['tempo'] + 40.0
            sample['acousticness'] = np.clip(sample['acousticness'] * 0.1, 0.0, 1.0)
        return sample
