#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
major_revision_iot_anomaly_pipeline_v4.py

Pipeline completo para major revision de artigo sobre anomaly detection
em séries temporais IoT de temperatura.

Inclui:
1) geração de datasets sintéticos S1-S7
2) geração de dataset field-like
3) integração de dataset real ambiental
4) weak labels multivariadas para o dataset real
5) avaliação de filtros estatísticos + baselines ML
6) métricas de deteção e reconstrução
7) custo computacional (tempo + memória)
8) agente com função de custo e sensibilidade de pesos
9) figuras científicas legíveis e em alta resolução
10) eixo X uniforme em minutos para cenários sintéticos
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FIG_DPI = 300
FIG_W = 14
FIG_H = 6

DEFAULT_COST_WEIGHTS = {
    "w_rmse": 0.30,
    "w_fpr": 0.20,
    "w_fnr": 0.20,
    "w_var": 0.10,
    "w_slope": 0.10,
    "w_spike": 0.10,
}

# ============================================================
# CONFIGURAÇÃO DE FIGURAS
# ============================================================

USE_UNIFORM_X_AXIS = True
X_AXIS_MODE = "minutes"     # "minutes" ou "sample"

SYNTHETIC_XLIM_SAMPLE = (0, 1000)
SYNTHETIC_XLIM_MINUTES = (0, 5000)

REAL_PLOT_MAX_POINTS = 1000

METHOD_ORDER = ["Hampel", "IQR", "ZScore", "IsolationForest", "OneClassSVM"]

METHOD_STYLES = {
    "Hampel": {"color": "orange", "marker": "o", "label": "Hampel anomalies"},
    "IQR": {"color": "green", "marker": "o", "label": "IQR anomalies"},
    "ZScore": {"color": "purple", "marker": "o", "label": "Z-score anomalies"},
    "IsolationForest": {"color": "red", "marker": "x", "label": "Isolation Forest anomalies"},
    "OneClassSVM": {"color": "brown", "marker": "x", "label": "One-Class SVM anomalies"},
}


# ============================================================
# UTILIDADES
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < 1e-12:
        sigma = float(np.nanstd(x))
    if not np.isfinite(sigma) or sigma < 1e-12:
        sigma = 1e-12
    return sigma


def rolling_median(series: pd.Series, window: int) -> pd.Series:
    half = max(window // 2, 1)
    med = series.rolling(window=window, center=True, min_periods=half).median()
    return med.bfill().ffill()


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    half = max(window // 2, 1)
    mu = series.rolling(window=window, center=True, min_periods=half).mean()
    return mu.bfill().ffill()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    half = max(window // 2, 1)
    s = series.rolling(window=window, center=True, min_periods=half).std(ddof=0)
    s = s.bfill().ffill()
    s = s.replace(0, np.nan)
    fillv = s[s > 0].median() if (s > 0).any() else 1e-12
    return s.fillna(fillv)


def compute_slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return 0.0
    xm = np.mean(x)
    ym = np.mean(y)
    denom = np.sum((x - xm) ** 2)
    if denom < 1e-12:
        return 0.0
    slope = np.sum((x - xm) * (y - ym)) / denom
    return float(slope)


def detect_residual_spikes(original: np.ndarray, cleaned: np.ndarray, threshold_sigma: float = 3.5) -> float:
    resid = np.asarray(original) - np.asarray(cleaned)
    scale = robust_scale(resid)
    score = np.abs(resid - np.median(resid)) / scale
    return float(np.mean(score > threshold_sigma))


def normalize_metric(value: float, ref: float, eps: float = 1e-12) -> float:
    ref = abs(ref)
    if ref < eps:
        ref = eps
    return float(value / ref)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "Accuracy": acc,
    }


def reconstruction_metrics(y_true_clean: np.ndarray, y_pred_clean: np.ndarray) -> Dict[str, float]:
    a = np.asarray(y_true_clean, dtype=float)
    b = np.asarray(y_pred_clean, dtype=float)
    diff = a - b
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return {"RMSE": rmse, "MAE": mae}


def profile_function(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, (t1 - t0) * 1000.0, peak / 1024.0


def mark_segment(gt: np.ndarray, atype: np.ndarray, start: int, end: int, label: str):
    start = max(0, start)
    end = min(len(gt), end)
    if end > start:
        gt[start:end] = 1
        atype[start:end] = label


def mark_point_window(gt: np.ndarray, atype: np.ndarray, center: int, half_window: int, label: str):
    start = max(0, center - half_window)
    end = min(len(gt), center + half_window + 1)
    gt[start:end] = 1
    atype[start:end] = label


def mark_threshold_deviation(gt: np.ndarray,
                             atype: np.ndarray,
                             observed: np.ndarray,
                             clean: np.ndarray,
                             threshold: float,
                             label: str,
                             start: Optional[int] = None,
                             end: Optional[int] = None):
    n = len(gt)
    s = 0 if start is None else max(0, start)
    e = n if end is None else min(n, end)
    diff = np.abs(np.asarray(observed[s:e]) - np.asarray(clean[s:e]))
    mask = diff >= threshold
    idx = np.where(mask)[0] + s
    gt[idx] = 1
    atype[idx] = label


def expand_binary_mask(mask: np.ndarray, left: int = 0, right: int = 0) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    out = mask.copy()
    idx = np.where(mask)[0]
    for i in idx:
        s = max(0, i - left)
        e = min(len(mask), i + right + 1)
        out[s:e] = True
    return out


def build_plot_x(df: pd.DataFrame):
    if X_AXIS_MODE == "sample":
        x = np.arange(len(df))
        xlabel = "Sample index"
    else:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        t0 = ts.iloc[0]
        x = (ts - t0).dt.total_seconds() / 60.0
        xlabel = "Time (min)"
    return x, xlabel


# ============================================================
# FEATURE MATRIX PARA BASELINES ML
# ============================================================

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    temp = df["temp_observed"].astype(float).to_numpy()
    temp_diff1 = np.diff(temp, prepend=temp[0])
    temp_diff2 = np.diff(temp_diff1, prepend=temp_diff1[0])
    temp_roll_std = pd.Series(temp).rolling(5, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()

    feats = [temp, temp_diff1, temp_diff2, temp_roll_std]

    if "humidity_avg" in df.columns:
        hum = pd.to_numeric(df["humidity_avg"], errors="coerce").ffill().bfill().to_numpy()
        hum_diff = np.diff(hum, prepend=hum[0])
        feats.extend([hum, hum_diff])

    if "light_avg" in df.columns:
        light = pd.to_numeric(df["light_avg"], errors="coerce").ffill().bfill().to_numpy()
        light_diff = np.diff(light, prepend=light[0])
        feats.extend([light, light_diff])

    X = np.column_stack(feats)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ============================================================
# GERAÇÃO DE DATASETS SINTÉTICOS
# ============================================================

def build_timestamp_series(n: int, freq: str = "5min", start: str = "2025-01-01 00:00:00") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq)


def base_temperature_signal(n: int, base_temp: float = 21.0, daily_amp: float = 0.6,
                            long_amp: float = 0.4, noise_sigma: float = 0.08,
                            trend: float = 0.0) -> np.ndarray:
    t = np.arange(n)
    daily = daily_amp * np.sin(2 * np.pi * t / 288.0)
    slow = long_amp * np.sin(2 * np.pi * t / 1200.0)
    linear = trend * t
    noise = np.random.normal(0, noise_sigma, n)
    return base_temp + daily + slow + linear + noise


def inject_spikes(x: np.ndarray, idx: List[int], magnitudes: List[float]) -> np.ndarray:
    y = x.copy()
    for i, m in zip(idx, magnitudes):
        if 0 <= i < len(y):
            y[i] += m
    return y


def inject_impulsive_segments(x: np.ndarray, starts: List[int], lengths: List[int], amplitudes: List[float]) -> np.ndarray:
    y = x.copy()
    for s, L, a in zip(starts, lengths, amplitudes):
        e = min(len(y), s + L)
        y[s:e] += a * np.sign(np.random.randn(e - s))
    return y


def inject_drift(x: np.ndarray, start: int, end: int, total_offset: float) -> np.ndarray:
    y = x.copy()
    start = max(0, start)
    end = min(len(y), end)
    if end <= start:
        return y
    drift = np.linspace(0, total_offset, end - start)
    y[start:end] += drift
    return y


def inject_flat_corruption(x: np.ndarray, start: int, end: int, hold_value: Optional[float] = None) -> np.ndarray:
    y = x.copy()
    start = max(0, start)
    end = min(len(y), end)
    if end <= start:
        return y
    if hold_value is None:
        hold_value = float(y[start])
    y[start:end] = hold_value
    return y


def inject_regime_switch(x: np.ndarray, switch_points: List[int], offsets: List[float]) -> np.ndarray:
    y = x.copy()
    prev = 0
    for sp, off in zip(switch_points, offsets):
        sp = min(max(sp, 0), len(y))
        y[prev:sp] += off
        prev = sp
    if offsets:
        y[prev:] += offsets[-1]
    return y


def assemble_df(sid: str, scenario_name: str, clean: np.ndarray, observed: np.ndarray,
                gt: np.ndarray, atype: np.ndarray,
                timestamp_start: str = "2025-01-01 00:00:00",
                base_name: Optional[str] = None) -> pd.DataFrame:
    n = len(clean)
    ts = build_timestamp_series(n, start=timestamp_start)
    df = pd.DataFrame({
        "timestamp": ts,
        "temp_clean": clean.astype(float),
        "temp_observed": observed.astype(float),
        "is_anomaly": gt.astype(int),
        "anomaly_type": atype.astype(object),
        "scenario_id": sid,
        "scenario_name": scenario_name,
        "seed": RANDOM_SEED,
    })
    if base_name is not None:
        df["dataset_name"] = base_name
    return df


def create_scenario_s1(n=1000) -> pd.DataFrame:
    clean = base_temperature_signal(n, base_temp=21.2, noise_sigma=0.05)
    obs = clean.copy()

    spike_idx = [80, 155, 311, 470, 650, 810, 920]
    spike_mag = [3.8, -4.2, 4.5, -3.9, 4.1, -4.3, 3.7]
    obs = inject_spikes(obs, spike_idx, spike_mag)

    gt = np.zeros(n, dtype=int)
    atype = np.array(["normal"] * n, dtype=object)
    for i in spike_idx:
        mark_point_window(gt, atype, i, half_window=0, label="spike")

    return assemble_df("S1", "Stable signal with sparse spikes", clean, obs, gt, atype)


def create_scenario_s2(n=1000) -> pd.DataFrame:
    clean = base_temperature_signal(n, base_temp=20.8, noise_sigma=0.08)
    obs = clean.copy()

    starts = [100, 230, 410, 700, 860]
    lengths = [8, 14, 10, 12, 9]
    amps = [2.8, 3.2, 2.5, 3.0, 2.7]
    obs = inject_impulsive_segments(obs, starts, lengths, amps)

    gt = np.zeros(n, dtype=int)
    atype = np.array(["normal"] * n, dtype=object)
    for s, L in zip(starts, lengths):
        e = min(n, s + L)
        mark_segment(gt, atype, s, e, "impulsive_noise")

    return assemble_df("S2", "Dense impulsive noise", clean, obs, gt, atype)


def create_scenario_s3(n=1000) -> pd.DataFrame:
    clean = base_temperature_signal(n, base_temp=22.0, noise_sigma=0.07, trend=0.0008)
    obs = clean.copy()

    drift_start = 250
    drift_end = 760
    drift_offset = 2.6
    obs = inject_drift(obs, drift_start, drift_end, total_offset=drift_offset)

    spike_idx = [310, 560, 730]
    spike_mag = [1.8, -2.1, 2.0]
    obs = inject_spikes(obs, spike_idx, spike_mag)

    gt = np.zeros(n, dtype=int)
    atype = np.array(["normal"] * n, dtype=object)

    mark_point_window(gt, atype, drift_start, half_window=8, label="drift_transition_start")
    mark_point_window(gt, atype, drift_end - 1, half_window=8, label="drift_transition_end")

    mark_threshold_deviation(
        gt=gt,
        atype=atype,
        observed=obs,
        clean=clean,
        threshold=1.85,
        label="drift_deviation",
        start=drift_start,
        end=drift_end
    )

    for i in spike_idx:
        mark_point_window(gt, atype, i, half_window=1, label="spike_on_drift")

    return assemble_df("S3", "Gradual drift with embedded spikes", clean, obs, gt, atype)


def create_scenario_s4(n=1000) -> pd.DataFrame:
    clean = base_temperature_signal(n, base_temp=19.9, noise_sigma=0.06)
    obs = clean.copy()
    obs = inject_flat_corruption(obs, 180, 245)
    obs = inject_flat_corruption(obs, 610, 690)
    obs = inject_spikes(obs, [120, 505, 870], [2.7, -3.1, 2.9])

    gt = np.zeros(n, dtype=int)
    atype = np.array(["normal"] * n, dtype=object)
    mark_segment(gt, atype, 180, 245, "flat_corrupted")
    mark_segment(gt, atype, 610, 690, "flat_corrupted")
    for i in [120, 505, 870]:
        mark_point_window(gt, atype, i, half_window=0, label="spike")

    return assemble_df("S4", "Corrupted flat segments", clean, obs, gt, atype)


def create_scenario_s5(n=1000) -> pd.DataFrame:
    clean = base_temperature_signal(n, base_temp=21.5, noise_sigma=0.08)
    obs = clean.copy()

    switch_points = [260, 520, 760]
    offsets = [0.0, 1.2, -0.8]
    obs = inject_regime_switch(obs, switch_points, offsets)

    spike_idx = [140, 398, 540, 801, 930]
    spike_mag = [2.4, -2.7, 2.2, -2.9, 2.5]
    obs = inject_spikes(obs, spike_idx, spike_mag)

    gt = np.zeros(n, dtype=int)
    atype = np.array(["normal"] * n, dtype=object)

    for sp in switch_points:
        mark_point_window(gt, atype, sp, half_window=8, label="regime_transition")

    mark_segment(gt, atype, 260, 290, "post_transition_instability")
    mark_segment(gt, atype, 520, 545, "post_transition_instability")
    mark_segment(gt, atype, 760, 785, "post_transition_instability")

    for i in spike_idx:
        mark_point_window(gt, atype, i, half_window=1, label="spike")

    return assemble_df("S5", "Mixed anomalies with non-stationary behaviour", clean, obs, gt, atype)


def create_scenario_s6(n=1000) -> pd.DataFrame:
    clean = base_temperature_signal(n, base_temp=20.6, noise_sigma=0.06, daily_amp=1.1)
    obs = clean.copy()

    anomaly_blocks = [(180, 205), (430, 455), (740, 765)]
    for s, e in anomaly_blocks:
        obs[s:e] += np.linspace(0.0, 2.2, e - s)
        obs[s:e] += np.random.normal(0, 0.2, e - s)

    gt = np.zeros(n, dtype=int)
    atype = np.array(["normal"] * n, dtype=object)
    for s, e in anomaly_blocks:
        mark_segment(gt, atype, s, e, "cyclic_embedded_anomaly")

    return assemble_df("S6", "Cyclic variations with embedded anomalies", clean, obs, gt, atype)


def create_scenario_s7(n=1000) -> pd.DataFrame:
    clean = base_temperature_signal(n, base_temp=21.0, noise_sigma=0.05)
    obs = clean.copy()

    heavy_idx = np.random.choice(np.arange(100, 950), size=35, replace=False)
    heavy_noise = np.random.standard_t(df=2.2, size=len(heavy_idx)) * 1.8
    obs[heavy_idx] += heavy_noise

    extreme_idx = [260, 611, 880]
    obs[extreme_idx] += [5.0, -5.5, 4.8]

    gt = np.zeros(n, dtype=int)
    gt[heavy_idx] = 1
    gt[extreme_idx] = 1
    atype = np.array(["normal"] * n, dtype=object)
    atype[heavy_idx] = "heavy_tailed_noise"
    atype[extreme_idx] = "extreme_event"

    return assemble_df("S7", "Heavy-tailed noise and extreme events", clean, obs, gt, atype)


def create_field_like_dataset(n=1440) -> pd.DataFrame:
    t = np.arange(n)
    clean = (
        4.2
        + 0.35 * np.sin(2 * np.pi * t / 288.0)
        + 0.15 * np.sin(2 * np.pi * t / 1440.0)
        + np.random.normal(0, 0.05, n)
    )
    obs = clean.copy()

    events = [(210, 225, 1.8), (480, 502, 2.4), (890, 905, 2.0), (1180, 1200, 2.6)]
    gt = np.zeros(n, dtype=int)
    atype = np.array(["normal"] * n, dtype=object)

    for s, e, amp in events:
        obs[s:e] += np.linspace(0.3, amp, e - s)
        mark_segment(gt, atype, s, e, "door_opening_excursion")

    obs[980:1100] += np.linspace(0.0, 3.1, 120)
    mark_segment(gt, atype, 980, 1100, "cooling_failure")

    for i in [140, 341, 700, 1320]:
        obs[i] += np.random.choice([2.2, -2.4])
        mark_point_window(gt, atype, i, half_window=0, label="sensor_spike")

    return assemble_df("SF", "Field-like cold-chain signal", clean, obs, gt, atype, base_name="field_like_temperature")


def save_generated_datasets(outdir: Path) -> List[Path]:
    dataset_dir = ensure_dir(outdir / "datasets")
    datasets = [
        create_scenario_s1(),
        create_scenario_s2(),
        create_scenario_s3(),
        create_scenario_s4(),
        create_scenario_s5(),
        create_scenario_s6(),
        create_scenario_s7(),
        create_field_like_dataset(),
    ]

    saved = []
    summary_rows = []
    for df in datasets:
        sid = df["scenario_id"].iloc[0]
        name = df["scenario_name"].iloc[0]
        filename = f"{sid.lower()}_{name.lower().replace(' ', '_').replace('-', '_')}.csv"
        path = dataset_dir / filename
        df.to_csv(path, index=False)
        saved.append(path)

        print(f"[CHECK] {sid}: anomalies={int(df['is_anomaly'].sum())} rate={100.0 * df['is_anomaly'].mean():.3f}%")

        summary_rows.append({
            "scenario_id": sid,
            "scenario_name": name,
            "samples": len(df),
            "anomalies": int(df["is_anomaly"].sum()),
            "anomaly_rate_pct": round(100.0 * df["is_anomaly"].mean(), 3),
            "file": path.name
        })

    pd.DataFrame(summary_rows).to_csv(dataset_dir / "dataset_summary.csv", index=False)
    return saved


# ============================================================
# DATASET REAL PRINCIPAL
# ============================================================

def infer_real_weak_labels_multivariate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    temp = out["temp_observed"].astype(float)
    baseline = rolling_median(temp, 31).ewm(span=9, adjust=False).mean()

    resid_t = temp - baseline
    z_temp = np.abs(resid_t - np.median(resid_t)) / robust_scale(resid_t.to_numpy())

    dtemp = temp.diff().fillna(0.0)
    z_dtemp = np.abs(dtemp - np.median(dtemp)) / robust_scale(dtemp.to_numpy())

    score = 0.65 * z_temp + 0.35 * z_dtemp

    if "humidity_avg" in out.columns:
        hum = pd.to_numeric(out["humidity_avg"], errors="coerce").ffill().bfill()
        hum_base = rolling_median(hum, 31).ewm(span=9, adjust=False).mean()
        hum_resid = hum - hum_base
        z_hum = np.abs(hum_resid - np.median(hum_resid)) / robust_scale(hum_resid.to_numpy())
        score = score + 0.20 * z_hum

    if "light_avg" in out.columns:
        light = pd.to_numeric(out["light_avg"], errors="coerce").ffill().bfill()
        light_diff = light.diff().fillna(0.0)
        z_light = np.abs(light_diff - np.median(light_diff)) / robust_scale(light_diff.to_numpy())
        score = score + 0.10 * z_light

    weak_label = (score > 4.75).astype(int)
    weak_label = expand_binary_mask(weak_label.to_numpy(), left=1, right=1).astype(int)

    out["temp_clean"] = baseline
    out["is_anomaly"] = weak_label
    out["anomaly_type"] = np.where(weak_label == 1, "weak_multivariate_excursion", "normal")
    return out


def load_real_sensor_benchmark(real_csv: Path,
                               min_points_per_series: int = 800,
                               max_series: int = 8) -> List[pd.DataFrame]:
    df = pd.read_csv(real_csv)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required_cols = ["timestamp", "boardid", "location", "temp_avg"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Falta a coluna obrigatória '{c}' no dataset real.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["temp_avg"] = pd.to_numeric(df["temp_avg"], errors="coerce")

    if "humidity_avg" in df.columns:
        df["humidity_avg"] = pd.to_numeric(df["humidity_avg"], errors="coerce")
    if "light_avg" in df.columns:
        df["light_avg"] = pd.to_numeric(df["light_avg"], errors="coerce")

    df = df.dropna(subset=["timestamp", "temp_avg"]).copy()
    df = df.sort_values(["boardid", "location", "timestamp"]).reset_index(drop=True)

    grouped = (
        df.groupby(["boardid", "location"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    selected = grouped[grouped["n"] >= min_points_per_series].head(max_series)

    real_series = []
    for _, row in selected.iterrows():
        boardid = row["boardid"]
        location = row["location"]

        sub = df[(df["boardid"] == boardid) & (df["location"] == location)].copy()
        sub = sub.sort_values("timestamp").reset_index(drop=True)

        keep_cols = ["timestamp", "temp_avg", "location", "boardid"]
        for c in ["humidity_avg", "light_avg", "latitude", "longitude", "model"]:
            if c in sub.columns:
                keep_cols.append(c)
        sub = sub[keep_cols].copy()

        try:
            boardid_int = int(float(boardid))
        except Exception:
            boardid_int = 0

        sub = sub.rename(columns={"temp_avg": "temp_observed"})
        sub["scenario_id"] = f"REAL_{boardid_int}"
        sub["scenario_name"] = f"Real sensor series - {location} - board {boardid_int}"
        sub["seed"] = RANDOM_SEED
        sub["dataset_name"] = f"real_sensor_board_{boardid_int}"

        sub = infer_real_weak_labels_multivariate(sub)
        real_series.append(sub)

    return real_series


def save_real_sensor_series(real_series: List[pd.DataFrame], outdir: Path) -> List[Path]:
    dataset_dir = ensure_dir(outdir / "datasets")
    paths = []
    rows = []
    for df in real_series:
        sid = df["scenario_id"].iloc[0]
        name = df["dataset_name"].iloc[0]
        filename = f"{name}.csv"
        path = dataset_dir / filename
        df.to_csv(path, index=False)
        paths.append(path)

        rows.append({
            "scenario_id": sid,
            "scenario_name": df["scenario_name"].iloc[0],
            "samples": len(df),
            "anomalies": int(df["is_anomaly"].sum()),
            "anomaly_rate_pct": round(100.0 * df["is_anomaly"].mean(), 3),
            "location": df["location"].iloc[0] if "location" in df.columns else "",
            "boardid": df["boardid"].iloc[0] if "boardid" in df.columns else "",
            "file": path.name
        })

    pd.DataFrame(rows).to_csv(dataset_dir / "real_dataset_summary.csv", index=False)
    return paths


# ============================================================
# MÉTODOS DE DETEÇÃO
# ============================================================

def hampel_filter(series: pd.Series, window: int = 21, n_sigmas: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    x = series.astype(float)
    med = rolling_median(x, window)
    mad = (x - med).abs().rolling(window=window, center=True, min_periods=max(window // 2, 1)).median()
    sigma = (1.4826 * mad).replace(0, np.nan)
    sigma = sigma.fillna(sigma[sigma > 0].median() if (sigma > 0).any() else 1e-12)
    flags = ((x - med).abs() > n_sigmas * sigma).fillna(False)
    clean = x.where(~flags, med).bfill().ffill()
    return clean, flags.astype(int)


def iqr_filter(series: pd.Series, window: int = 21, k_mult: float = 1.5) -> Tuple[pd.Series, pd.Series]:
    x = series.astype(float)
    half = max(window // 2, 1)
    q1 = x.rolling(window=window, center=True, min_periods=half).quantile(0.25)
    q3 = x.rolling(window=window, center=True, min_periods=half).quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k_mult * iqr
    upper = q3 + k_mult * iqr
    flags = ((x < lower) | (x > upper)).fillna(False)
    med = rolling_median(x, window)
    clean = x.where(~flags, med).bfill().ffill()
    return clean, flags.astype(int)


def zscore_filter(series: pd.Series, window: int = 21, zthr: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    x = series.astype(float)
    mu = rolling_mean(x, window)
    std = rolling_std(x, window)
    z = ((x - mu).abs() / std).fillna(0.0)
    flags = (z > zthr).astype(int)
    med = rolling_median(x, window)
    clean = x.where(flags == 0, med).bfill().ffill()
    return clean, flags


def isolation_forest_detector(feature_matrix: np.ndarray, series: pd.Series, contamination: float = 0.03) -> Tuple[pd.Series, pd.Series]:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_SEED
    )
    pred = model.fit_predict(feature_matrix)
    flags = (pred == -1).astype(int)
    med = rolling_median(series.astype(float), 21)
    clean = series.astype(float).where(flags == 0, med).bfill().ffill()
    return clean, pd.Series(flags, index=series.index)


def ocsvm_detector(feature_matrix: np.ndarray, series: pd.Series, nu: float = 0.03, gamma: str = "scale") -> Tuple[pd.Series, pd.Series]:
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    pred = model.fit_predict(feature_matrix)
    flags = (pred == -1).astype(int)
    med = rolling_median(series.astype(float), 21)
    clean = series.astype(float).where(flags == 0, med).bfill().ffill()
    return clean, pd.Series(flags, index=series.index)


# ============================================================
# FUNÇÃO DE CUSTO / AGENTE
# ============================================================

def compute_agent_cost(original: np.ndarray,
                       clean_target: np.ndarray,
                       cleaned: np.ndarray,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       weights: Dict[str, float]) -> Dict[str, float]:
    rec = reconstruction_metrics(clean_target, cleaned)
    det = binary_metrics(y_true, y_pred)

    original_var = np.var(original) + 1e-12
    cleaned_var = np.var(cleaned) + 1e-12

    rmse_n = normalize_metric(rec["RMSE"], np.sqrt(original_var))
    fpr = det["FPR"]
    fnr = det["FNR"]
    var_n = abs(cleaned_var - np.var(clean_target)) / (np.var(clean_target) + 1e-12)
    slope_diff = abs(compute_slope(original) - compute_slope(cleaned))
    slope_n = normalize_metric(slope_diff, abs(compute_slope(original)) + 1e-6)
    spike_ratio = detect_residual_spikes(original, cleaned)

    cost = (
        weights["w_rmse"] * rmse_n +
        weights["w_fpr"] * fpr +
        weights["w_fnr"] * fnr +
        weights["w_var"] * var_n +
        weights["w_slope"] * slope_n +
        weights["w_spike"] * spike_ratio
    )

    return {
        "RMSE_n": rmse_n,
        "FPR_component": fpr,
        "FNR_component": fnr,
        "Var_n": var_n,
        "Slope_n": slope_n,
        "ResidualSpikeRatio": spike_ratio,
        "AgentCost": cost,
    }


def sensitivity_weight_configs() -> List[Dict[str, float]]:
    configs = []
    values = [
        {"w_rmse": 0.40, "w_fpr": 0.15, "w_fnr": 0.15, "w_var": 0.10, "w_slope": 0.10, "w_spike": 0.10},
        {"w_rmse": 0.25, "w_fpr": 0.25, "w_fnr": 0.20, "w_var": 0.10, "w_slope": 0.10, "w_spike": 0.10},
        {"w_rmse": 0.20, "w_fpr": 0.20, "w_fnr": 0.25, "w_var": 0.10, "w_slope": 0.10, "w_spike": 0.15},
        {"w_rmse": 0.30, "w_fpr": 0.10, "w_fnr": 0.10, "w_var": 0.20, "w_slope": 0.15, "w_spike": 0.15},
        {"w_rmse": 0.30, "w_fpr": 0.20, "w_fnr": 0.20, "w_var": 0.10, "w_slope": 0.10, "w_spike": 0.10},
    ]
    for i, cfg in enumerate(values, start=1):
        cfg2 = dict(cfg)
        cfg2["config_id"] = i
        configs.append(cfg2)
    return configs


# ============================================================
# FIGURAS
# ============================================================

def style_axes(ax, title: str, xlabel: str = "Time (min)", ylabel: str = "Temperature (°C)"):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.25)


def plot_detection_comparison(df: pd.DataFrame,
                              method_outputs: Dict[str, Dict[str, pd.Series]],
                              save_path: Path):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    x, xlabel = build_plot_x(df)
    ax.plot(x, df["temp_observed"], label="Temperature", linewidth=1.2)

    for method_name in METHOD_ORDER:
        if method_name not in method_outputs:
            continue
        flags = method_outputs[method_name]["flags"].to_numpy().astype(int)
        idx = np.where(flags == 1)[0]
        if len(idx) > 0:
            st = METHOD_STYLES[method_name]
            ax.scatter(np.asarray(x)[idx], df["temp_observed"].iloc[idx], s=20, marker=st["marker"], label=st["label"])

    sid = str(df["scenario_id"].iloc[0])
    if USE_UNIFORM_X_AXIS and not sid.startswith("REAL_"):
        if X_AXIS_MODE == "sample":
            ax.set_xlim(*SYNTHETIC_XLIM_SAMPLE)
        else:
            ax.set_xlim(*SYNTHETIC_XLIM_MINUTES)

    style_axes(ax,
               f"Detection comparison: {df['scenario_id'].iloc[0]} - {df['scenario_name'].iloc[0]}",
               xlabel=xlabel,
               ylabel="Temperature (°C)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_comparison(df: pd.DataFrame,
                                   method_outputs: Dict[str, Dict[str, pd.Series]],
                                   save_path: Path,
                                   methods_to_show: List[str] = None):
    if methods_to_show is None:
        methods_to_show = ["Hampel", "IsolationForest", "OneClassSVM"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x, xlabel = build_plot_x(df)

    ax.plot(x, df["temp_observed"], label="Observed", linewidth=1.2, alpha=0.8)
    ax.plot(x, df["temp_clean"], label="Reference clean/baseline", linewidth=1.5)

    for method_name in methods_to_show:
        if method_name in method_outputs:
            ax.plot(x, method_outputs[method_name]["cleaned"], linewidth=1.0, label=f"{method_name} cleaned")

    sid = str(df["scenario_id"].iloc[0])
    if USE_UNIFORM_X_AXIS and not sid.startswith("REAL_"):
        if X_AXIS_MODE == "sample":
            ax.set_xlim(*SYNTHETIC_XLIM_SAMPLE)
        else:
            ax.set_xlim(*SYNTHETIC_XLIM_MINUTES)

    style_axes(ax,
               f"Reconstruction comparison: {df['scenario_id'].iloc[0]} - {df['scenario_name'].iloc[0]}",
               xlabel=xlabel,
               ylabel="Temperature (°C)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_real_multivariate_context(df: pd.DataFrame, save_path: Path):
    plot_df = df.copy()
    if len(plot_df) > REAL_PLOT_MAX_POINTS:
        plot_df = plot_df.iloc[:REAL_PLOT_MAX_POINTS].copy()

    x, xlabel = build_plot_x(plot_df)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(x, plot_df["temp_observed"], linewidth=1.2, label="Observed temperature")
    axes[0].plot(x, plot_df["temp_clean"], linewidth=1.4, label="Robust baseline")

    idx = plot_df["is_anomaly"] == 1
    if idx.any():
        axes[0].scatter(
            np.asarray(x)[np.where(idx.to_numpy())[0]],
            plot_df.loc[idx, "temp_observed"],
            s=16,
            marker="x",
            label="Weak anomaly label"
        )

    axes[0].set_ylabel("Temp (°C)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    if "humidity_avg" in plot_df.columns:
        axes[1].plot(x, plot_df["humidity_avg"], linewidth=1.0, label="Humidity")
        axes[1].set_ylabel("Humidity")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.25)
    else:
        axes[1].axis("off")

    if "light_avg" in plot_df.columns:
        axes[2].plot(x, plot_df["light_avg"], linewidth=1.0, label="Light")
        axes[2].set_ylabel("Light")
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.25)
    else:
        axes[2].axis("off")

    axes[2].set_xlabel(xlabel)
    fig.suptitle(f"Real sensor benchmark context: {plot_df['scenario_name'].iloc[0]}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bar(df: pd.DataFrame, metric: str, title: str, save_path: Path):
    ordered = df.sort_values(metric, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ordered["method"], ordered[metric])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_bar(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df["method"], df["Runtime_ms_mean"])

    ax.set_title("Mean execution time by method", fontsize=14)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Runtime (ms, log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.25)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_memory_bar(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df["method"], df["PeakMemory_KB_mean"])

    ax.set_title("Mean peak memory by method", fontsize=14)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Peak memory (KB)", fontsize=12)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.25)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("config_id")
        ax.plot(sub["config_id"], sub["AgentCost"], marker="o", linewidth=1.2, label=method)
    ax.set_title("Agent cost sensitivity analysis across weight configurations", fontsize=14)
    ax.set_xlabel("Weight configuration ID", fontsize=12)
    ax.set_ylabel("Agent cost", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# AVALIAÇÃO POR MÉTODO
# ============================================================

def evaluate_one_method(method_name: str,
                        df: pd.DataFrame,
                        weights: Dict[str, float]) -> Dict[str, object]:

    series = df["temp_observed"].astype(float)
    clean_target = df["temp_clean"].astype(float)
    y_true = df["is_anomaly"].astype(int)
    X = build_feature_matrix(df)

    if method_name == "Hampel":
        (cleaned, flags), runtime_ms, peak_kb = profile_function(hampel_filter, series)
    elif method_name == "IQR":
        (cleaned, flags), runtime_ms, peak_kb = profile_function(iqr_filter, series)
    elif method_name == "ZScore":
        (cleaned, flags), runtime_ms, peak_kb = profile_function(zscore_filter, series)
    elif method_name == "IsolationForest":
        (cleaned, flags), runtime_ms, peak_kb = profile_function(isolation_forest_detector, X, series)
    elif method_name == "OneClassSVM":
        (cleaned, flags), runtime_ms, peak_kb = profile_function(ocsvm_detector, X, series)
    else:
        raise ValueError(f"Método não suportado: {method_name}")

    det = binary_metrics(y_true.to_numpy(), flags.to_numpy())
    rec = reconstruction_metrics(clean_target.to_numpy(), cleaned.to_numpy())
    cost = compute_agent_cost(
        original=series.to_numpy(),
        clean_target=clean_target.to_numpy(),
        cleaned=cleaned.to_numpy(),
        y_true=y_true.to_numpy(),
        y_pred=flags.to_numpy(),
        weights=weights,
    )

    row = {
        "method": method_name,
        "Runtime_ms": runtime_ms,
        "PeakMemory_KB": peak_kb,
        **det,
        **rec,
        **cost,
        "cleaned_series": cleaned,
        "flag_series": flags,
    }
    return row


def process_dataset(df: pd.DataFrame,
                    dataset_name: str,
                    outdir: Path,
                    default_weights: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    rows = []
    method_outputs = {}

    for method in METHOD_ORDER:
        res = evaluate_one_method(method, df, default_weights)
        cleaned = res.pop("cleaned_series")
        flags = res.pop("flag_series")

        res["dataset_name"] = dataset_name
        res["scenario_id"] = df["scenario_id"].iloc[0]
        res["scenario_name"] = df["scenario_name"].iloc[0]
        rows.append(res)

        method_outputs[method] = {"cleaned": cleaned, "flags": flags}

    result_df = pd.DataFrame(rows)

    best_method = result_df.sort_values("AgentCost").iloc[0]["method"]
    result_df["AgentSelected"] = (result_df["method"] == best_method).astype(int)

    export_df = df.copy()
    for method, payload in method_outputs.items():
        export_df[f"{method}_cleaned"] = payload["cleaned"].to_numpy()
        export_df[f"{method}_flag"] = payload["flags"].to_numpy()

    dataset_result_dir = ensure_dir(outdir / "per_dataset" / dataset_name)
    export_df.to_csv(dataset_result_dir / f"{dataset_name}_method_outputs.csv", index=False)
    result_df.to_csv(dataset_result_dir / f"{dataset_name}_metrics.csv", index=False)

    plot_detection_comparison(
        df=df,
        method_outputs=method_outputs,
        save_path=dataset_result_dir / f"{dataset_name}_detection_comparison.png"
    )

    plot_reconstruction_comparison(
        df=df,
        method_outputs=method_outputs,
        save_path=dataset_result_dir / f"{dataset_name}_reconstruction_comparison.png"
    )

    if str(df["scenario_id"].iloc[0]).startswith("REAL_"):
        plot_real_multivariate_context(
            df=df,
            save_path=dataset_result_dir / f"{dataset_name}_real_context.png"
        )

    sens_rows = []
    for cfg in sensitivity_weight_configs():
        cfg_id = cfg["config_id"]
        w = {k: v for k, v in cfg.items() if k != "config_id"}
        for method in METHOD_ORDER:
            cleaned = export_df[f"{method}_cleaned"].to_numpy()
            flags = export_df[f"{method}_flag"].to_numpy()
            cost = compute_agent_cost(
                original=df["temp_observed"].astype(float).to_numpy(),
                clean_target=df["temp_clean"].astype(float).to_numpy(),
                cleaned=cleaned,
                y_true=df["is_anomaly"].astype(int).to_numpy(),
                y_pred=flags,
                weights=w
            )
            sens_rows.append({
                "dataset_name": dataset_name,
                "scenario_id": df["scenario_id"].iloc[0],
                "method": method,
                "config_id": cfg_id,
                **w,
                **cost
            })

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(dataset_result_dir / f"{dataset_name}_sensitivity.csv", index=False)
    plot_sensitivity(sens_df, dataset_result_dir / f"{dataset_name}_sensitivity.png")

    return result_df, sens_df


# ============================================================
# SAÍDAS GLOBAIS
# ============================================================

def build_global_outputs(all_metrics: pd.DataFrame, all_sens: pd.DataFrame, outdir: Path):
    ensure_dir(outdir / "tables")
    ensure_dir(outdir / "figures")

    all_metrics.to_csv(outdir / "tables" / "all_datasets_metrics.csv", index=False)
    all_sens.to_csv(outdir / "tables" / "all_datasets_sensitivity.csv", index=False)

    summary = all_metrics.groupby("method", as_index=False).agg({
        "Precision": "mean",
        "Recall": "mean",
        "F1": "mean",
        "FPR": "mean",
        "Accuracy": "mean",
        "RMSE": "mean",
        "MAE": "mean",
        "Runtime_ms": "mean",
        "PeakMemory_KB": "mean",
        "AgentCost": "mean",
        "AgentSelected": "sum",
    }).rename(columns={
        "Runtime_ms": "Runtime_ms_mean",
        "PeakMemory_KB": "PeakMemory_KB_mean",
        "AgentSelected": "AgentSelectionCount",
    })

    summary.to_csv(outdir / "tables" / "method_summary_mean.csv", index=False)

    scenario_summary = all_metrics.groupby(["scenario_id", "scenario_name", "method"], as_index=False).agg({
        "Precision": "mean",
        "Recall": "mean",
        "F1": "mean",
        "FPR": "mean",
        "Accuracy": "mean",
        "RMSE": "mean",
        "MAE": "mean",
        "Runtime_ms": "mean",
        "PeakMemory_KB": "mean",
        "AgentCost": "mean",
    })
    scenario_summary.to_csv(outdir / "tables" / "scenario_method_summary.csv", index=False)

    ds_rows = []
    dataset_dir = outdir / "datasets"
    required_cols = {"scenario_id", "scenario_name", "temp_observed", "is_anomaly"}

    for csv_path in sorted(dataset_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)

            if not required_cols.issubset(set(df.columns)):
                print(f"[INFO] Ignorado em dataset_characteristics_table: {csv_path.name} (faltam colunas benchmark obrigatórias)")
                continue

            ds_rows.append({
                "file": csv_path.name,
                "scenario_id": df["scenario_id"].iloc[0],
                "scenario_name": df["scenario_name"].iloc[0],
                "samples": len(df),
                "anomalies": int(pd.to_numeric(df["is_anomaly"], errors="coerce").fillna(0).sum()),
                "anomaly_rate_pct": round(100.0 * pd.to_numeric(df["is_anomaly"], errors="coerce").fillna(0).mean(), 3),
                "temp_mean": round(pd.to_numeric(df["temp_observed"], errors="coerce").mean(), 4),
                "temp_std": round(pd.to_numeric(df["temp_observed"], errors="coerce").std(), 4),
            })

        except Exception as e:
            print(f"[WARNING] Erro a processar {csv_path.name} em dataset_characteristics_table: {e}")

    if ds_rows:
        pd.DataFrame(ds_rows).to_csv(outdir / "tables" / "dataset_characteristics_table.csv", index=False)
    else:
        print("[WARNING] Nenhum dataset benchmark válido encontrado para dataset_characteristics_table.csv")

    real_metrics = all_metrics[all_metrics["scenario_id"].astype(str).str.startswith("REAL_")].copy()
    if not real_metrics.empty:
        real_summary = real_metrics.groupby("method", as_index=False).agg({
            "Precision": "mean",
            "Recall": "mean",
            "F1": "mean",
            "FPR": "mean",
            "Accuracy": "mean",
            "RMSE": "mean",
            "MAE": "mean",
            "Runtime_ms": "mean",
            "PeakMemory_KB": "mean",
            "AgentCost": "mean",
        })
        real_summary.to_csv(outdir / "tables" / "real_benchmark_method_summary.csv", index=False)

    plot_metric_bar(summary, "F1", "Mean F1-score by method", outdir / "figures" / "mean_f1_by_method.png")
    plot_metric_bar(summary, "Recall", "Mean recall by method", outdir / "figures" / "mean_recall_by_method.png")
    plot_metric_bar(summary, "Precision", "Mean precision by method", outdir / "figures" / "mean_precision_by_method.png")
    plot_runtime_bar(summary, outdir / "figures" / "runtime_by_method.png")
    plot_memory_bar(summary, outdir / "figures" / "memory_by_method.png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./revision_outputs")
    parser.add_argument(
        "--real_csv",
        type=str,
        default=None,
        help="CSV real principal: sensor-readings-with-temperature-light-humidity-every-5-minutes-at-8-locations-t.csv"
    )
    parser.add_argument("--min_real_points", type=int, default=800)
    parser.add_argument("--max_real_series", type=int, default=8)
    args = parser.parse_args()

    outdir = ensure_dir(Path(args.outdir))
    ensure_dir(outdir / "datasets")
    ensure_dir(outdir / "per_dataset")
    ensure_dir(outdir / "tables")
    ensure_dir(outdir / "figures")

    print("\n[1/6] A gerar datasets sintéticos e field-like...")
    generated_paths = save_generated_datasets(outdir)

    print("[2/6] A processar dataset real principal...")
    if args.real_csv:
        real_series = load_real_sensor_benchmark(
            Path(args.real_csv),
            min_points_per_series=args.min_real_points,
            max_series=args.max_real_series
        )
        real_paths = save_real_sensor_series(real_series, outdir)
        generated_paths.extend(real_paths)
    else:
        print("Sem dataset real fornecido. O pipeline continuará apenas com datasets sintéticos e field-like.")

    print("[3/6] A avaliar métodos por dataset...")
    all_metric_frames = []
    all_sens_frames = []

    for csv_path in sorted(generated_paths):
        df = pd.read_csv(csv_path)
        dataset_name = Path(csv_path).stem

        metrics_df, sens_df = process_dataset(
            df=df,
            dataset_name=dataset_name,
            outdir=outdir,
            default_weights=DEFAULT_COST_WEIGHTS,
        )
        all_metric_frames.append(metrics_df)
        all_sens_frames.append(sens_df)

    print("[4/6] A consolidar outputs globais...")
    all_metrics = pd.concat(all_metric_frames, ignore_index=True)
    all_sens = pd.concat(all_sens_frames, ignore_index=True)
    build_global_outputs(all_metrics, all_sens, outdir)

    print("[5/6] A exportar relatório resumo...")
    report_path = outdir / "revision_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Major Revision IoT Temperature Anomaly Detection Pipeline - V4\n")
        f.write("=============================================================\n\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Default cost weights: {DEFAULT_COST_WEIGHTS}\n")
        f.write(f"X-axis mode: {X_AXIS_MODE}\n")
        f.write(f"Synthetic uniform x-axis: {USE_UNIFORM_X_AXIS}\n\n")

        f.write("Generated outputs:\n")
        f.write("- datasets/\n")
        f.write("- per_dataset/\n")
        f.write("- tables/\n")
        f.write("- figures/\n\n")

        f.write("Main comparison methods:\n")
        for m in METHOD_ORDER:
            f.write(f"- {m}\n")
        f.write("\n")

        f.write("Main metrics:\n")
        f.write("- Precision\n")
        f.write("- Recall\n")
        f.write("- F1\n")
        f.write("- FPR\n")
        f.write("- Accuracy\n")
        f.write("- RMSE\n")
        f.write("- MAE\n")
        f.write("- Runtime_ms\n")
        f.write("- PeakMemory_KB\n")
        f.write("- AgentCost\n\n")

        if args.real_csv:
            f.write(f"Real dataset ingested: {args.real_csv}\n")
            f.write("Important note: the real dataset was used with weak multivariate labels, not with manually verified fault ground truth.\n")
            f.write("It should be described in the manuscript as an external real-world IoT environmental benchmark.\n")
        else:
            f.write("No real dataset was supplied.\n")

    print("[6/6] Concluído.")
    print(f"Outputs guardados em: {outdir.resolve()}\n")


if __name__ == "__main__":
    main()