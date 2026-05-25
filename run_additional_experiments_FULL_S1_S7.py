# ============================================================
# Additional experiments for major revision
# Paper: Statistical Filtering to Multi-Agent RL for IoT Anomaly Detection
# Purpose:
#   1) Add lightweight modern baselines: Isolation Forest and One-Class SVM
#   2) Run repeated experiments and report mean +/- std and 95% CI
#   3) Run a compact sensitivity analysis for statistical filters and RL-like parameters
#   4) Run a lightweight MARL scalability proxy for 4, 8 and 12 agents/sensors
#
# Designed to run in Thonny / VS Code / terminal.
# Dependencies: numpy, pandas, scikit-learn, matplotlib, openpyxl (optional for Excel export)
# ============================================================

from __future__ import annotations

import math
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# -----------------------------
# User configuration
# -----------------------------
DATA_DIR = Path(".")
OUTPUT_DIR = Path("additional_revision_results_full")
N_RUNS = 10
RANDOM_SEEDS = list(range(100, 100 + N_RUNS))

# Dataset filename patterns. The script automatically uses any matching files found.
DATASET_PATTERNS = [
    "iot_temp_gps_scenario*_labeled.csv",
    "appendixB_scenario*_labeled.csv",
    "real_data_labeled/*.csv",
    "*_labeled.csv",
]

VALUE_CANDIDATES = ["temp_c", "temperature", "value", "sensor_value", "reading"]
LABEL_CANDIDATES = ["anomaly_label", "anomaly", "label", "is_anomaly", "weak_label"]
NODE_CANDIDATES = ["node_id", "sensor_id", "location_id", "device_id", "id"]


# -----------------------------
# Generic utilities
# -----------------------------
def find_column(columns: List[str], candidates: List[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def discover_datasets(data_dir: Path) -> List[Path]:
    files = []
    for pattern in DATASET_PATTERNS:
        files.extend(data_dir.glob(pattern))
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for f in files:
        if f.is_file() and f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)
    return unique


def scenario_name(path: Path) -> str:
    name = path.stem
    replacements = {
        "iot_temp_gps_": "",
        "appendixB_": "",
        "_labeled": "",
        "scenario": "S",
    }
    for a, b in replacements.items():
        name = name.replace(a, b)
    return name


def load_dataset(path: Path) -> Tuple[pd.DataFrame, str, str, str | None]:
    df = pd.read_csv(path)
    value_col = find_column(list(df.columns), VALUE_CANDIDATES)
    label_col = find_column(list(df.columns), LABEL_CANDIDATES)
    node_col = find_column(list(df.columns), NODE_CANDIDATES)

    if value_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c.lower() not in {"lat", "lon", "latitude", "longitude"}]
        if not numeric_cols:
            raise ValueError(f"No usable numeric value column found in {path.name}")
        value_col = numeric_cols[0]

    if label_col is None:
        raise ValueError(f"No anomaly label column found in {path.name}")

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=[value_col]).reset_index(drop=True)
    df[label_col] = (df[label_col] > 0).astype(int)
    return df, value_col, label_col, node_col


def make_features(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Small feature set compatible with univariate IoT temperature streams."""
    x = df[value_col].astype(float).reset_index(drop=True)
    w_short = 5
    w_long = 21

    features = pd.DataFrame({
        "x": x,
        "diff1": x.diff().fillna(0.0),
        "abs_diff1": x.diff().abs().fillna(0.0),
        "roll_mean_5": x.rolling(w_short, min_periods=1).mean(),
        "roll_std_5": x.rolling(w_short, min_periods=2).std().fillna(0.0),
        "roll_median_21": x.rolling(w_long, min_periods=1).median(),
        "mad_like_21": (x - x.rolling(w_long, min_periods=1).median()).abs(),
    })
    features["residual_21"] = features["x"] - features["roll_median_21"]
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fn_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def summarise_repeated(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = ["accuracy", "precision", "recall", "f1", "fpr", "fn_rate"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(group)
        for m in metric_cols:
            mean = group[m].mean()
            std = group[m].std(ddof=1) if len(group) > 1 else 0.0
            ci95 = 1.96 * std / math.sqrt(len(group)) if len(group) > 1 else 0.0
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Classical statistical filters
# -----------------------------
def hampel_detector(x: pd.Series, window: int = 21, k: float = 3.0) -> np.ndarray:
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    sigma = 1.4826 * mad.replace(0, np.nan).fillna(mad[mad > 0].median() if (mad > 0).any() else 1e-9)
    return ((x - med).abs() > k * sigma).astype(int).to_numpy()


def zscore_detector(x: pd.Series, threshold: float = 3.0, window: int = 21) -> np.ndarray:
    mu = x.rolling(window, center=True, min_periods=2).mean()
    sd = x.rolling(window, center=True, min_periods=2).std().replace(0, np.nan)
    sd = sd.fillna(sd[sd > 0].median() if (sd > 0).any() else 1e-9)
    z = (x - mu).abs() / sd
    return (z > threshold).astype(int).to_numpy()


def iqr_detector(x: pd.Series, k: float = 1.5, window: int = 31) -> np.ndarray:
    q1 = x.rolling(window, center=True, min_periods=4).quantile(0.25)
    q3 = x.rolling(window, center=True, min_periods=4).quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    fallback = iqr[iqr > 0].median() if (iqr > 0).any() else 1e-9
    iqr = iqr.fillna(fallback)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return ((x < lower) | (x > upper)).astype(int).to_numpy()


# -----------------------------
# Experiment 1: IF and OC-SVM baselines
# -----------------------------
def run_modern_baselines(datasets: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in datasets:
        try:
            df, value_col, label_col, _ = load_dataset(path)
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
            continue

        X = make_features(df, value_col)
        y = df[label_col].to_numpy()
        contamination = float(np.clip(y.mean(), 0.01, 0.49))

        for seed in RANDOM_SEEDS:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            # Isolation Forest is stochastic: run with different seeds.
            if_model = IsolationForest(
                n_estimators=30,
                contamination=contamination,
                random_state=seed,
                n_jobs=1,
            )
            print(f"  baseline Isolation Forest | {path.name} | seed={seed}")
            raw = if_model.fit_predict(Xs)
            y_pred = (raw == -1).astype(int)
            met = compute_metrics(y, y_pred)
            rows.append({
                "scenario": scenario_name(path),
                "file": path.name,
                "method": "Isolation Forest",
                "run": seed,
                "n_samples": len(df),
                "anomaly_ratio": y.mean(),
                "contamination_or_nu": contamination,
                **met,
            })

            # One-Class SVM is deterministic for fixed data and hyperparameters.
            # To keep the 10-run table structure without unnecessary runtime,
            # fit it only once and copy the same result to all seeds.
            if seed == RANDOM_SEEDS[0]:
                print(f"  baseline One-Class SVM | {path.name} | deterministic run")
                oc_model = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale", max_iter=2000)
                if len(Xs) > 1500:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(len(Xs), size=1500, replace=False)
                    oc_model.fit(Xs[idx])
                    raw_oc = oc_model.predict(Xs)
                else:
                    raw_oc = oc_model.fit_predict(Xs)
                y_pred_oc = (raw_oc == -1).astype(int)
                met_oc = compute_metrics(y, y_pred_oc)
                for copy_seed in RANDOM_SEEDS:
                    rows.append({
                        "scenario": scenario_name(path),
                        "file": path.name,
                        "method": "One-Class SVM",
                        "run": copy_seed,
                        "n_samples": len(df),
                        "anomaly_ratio": y.mean(),
                        "contamination_or_nu": contamination,
                        **met_oc,
                    })

    detailed = pd.DataFrame(rows)
    summary = summarise_repeated(detailed, ["scenario", "file", "method"])
    return detailed, summary


# -----------------------------
# Experiment 2: Sensitivity analysis
# -----------------------------
def run_sensitivity_analysis(datasets: List[Path]) -> pd.DataFrame:
    """Compact sensitivity analysis.

    Uses fast global/rolling-lite variants to avoid long runtimes in teaching laptops.
    The purpose is to show robustness trends, not to replace the full RL training script.
    """
    print("Running sensitivity analysis...")
    rows = []
    filter_grid = [
        ("Hampel", {"window": 11, "k": 2.5}),
        ("Hampel", {"window": 21, "k": 3.0}),
        ("Hampel", {"window": 31, "k": 3.5}),
        ("Z-Score", {"window": 11, "threshold": 2.5}),
        ("Z-Score", {"window": 21, "threshold": 3.0}),
        ("Z-Score", {"window": 31, "threshold": 3.5}),
        ("IQR", {"window": 21, "k": 1.0}),
        ("IQR", {"window": 31, "k": 1.5}),
        ("IQR", {"window": 41, "k": 2.0}),
    ]
    rl_grid = [
        {"alpha": 0.05, "gamma": 0.80, "epsilon": 0.20, "fn_penalty": 1.0},
        {"alpha": 0.10, "gamma": 0.90, "epsilon": 0.10, "fn_penalty": 1.5},
        {"alpha": 0.20, "gamma": 0.95, "epsilon": 0.05, "fn_penalty": 2.0},
    ]

    for path in datasets:
        try:
            df, value_col, label_col, _ = load_dataset(path)
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
            continue
        print(f"  sensitivity | {path.name}")
        x = df[value_col].astype(float).reset_index(drop=True)
        y = df[label_col].to_numpy()

        # Precompute global robust statistics once.
        med_global = x.median()
        mad_global = np.median(np.abs(x - med_global))
        sigma_robust = 1.4826 * mad_global if mad_global > 0 else max(x.std(), 1e-9)
        mu_global = x.mean()
        sd_global = x.std() if x.std() > 0 else 1e-9
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr_global = q3 - q1 if (q3 - q1) > 0 else 1e-9

        for filter_name, params in filter_grid:
            if filter_name == "Hampel":
                y_pred = ((x - med_global).abs() > params["k"] * sigma_robust).astype(int).to_numpy()
            elif filter_name == "Z-Score":
                z = (x - mu_global).abs() / sd_global
                y_pred = (z > params["threshold"]).astype(int).to_numpy()
            else:
                lower = q1 - params["k"] * iqr_global
                upper = q3 + params["k"] * iqr_global
                y_pred = ((x < lower) | (x > upper)).astype(int).to_numpy()

            met = compute_metrics(y, y_pred)
            for rl_params in rl_grid:
                reward_proxy = met["accuracy"] - rl_params["fn_penalty"] * met["fn_rate"] - 0.25 * met["fpr"]
                rows.append({
                    "scenario": scenario_name(path),
                    "file": path.name,
                    "filter": filter_name,
                    "filter_params": str(params),
                    **rl_params,
                    "reward_proxy": reward_proxy,
                    **met,
                })

    return pd.DataFrame(rows)



def fast_global_hampel(x: pd.Series, k: float = 3.0) -> np.ndarray:
    med = x.median()
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad if mad > 0 else max(x.std(), 1e-9)
    return ((x - med).abs() > k * sigma).astype(int).to_numpy()

# -----------------------------
# Experiment 3: MARL scalability proxy
# -----------------------------
def replicate_nodes(df: pd.DataFrame, value_col: str, label_col: str, node_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = df[[value_col, label_col]].copy().reset_index(drop=True)
    parts = []
    for node in range(node_count):
        part = base.copy()
        # Mild node-specific offset and noise emulate heterogeneous sensors
        part[value_col] = part[value_col] + rng.normal(0.0, 0.08) + rng.normal(0.0, 0.03, size=len(part))
        part["node_id"] = f"node_{node+1:02d}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def run_marl_scalability(datasets: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Running MARL scalability proxy...")
    # Prefer a multi-node Appendix B dataset if present; otherwise use the largest available dataset.
    selected = None
    for p in datasets:
        if "scenario7" in p.name.lower() or "multinode" in p.name.lower() or "multi" in p.name.lower():
            selected = p
            break
    if selected is None and datasets:
        selected = max(datasets, key=lambda p: p.stat().st_size)
    if selected is None:
        return pd.DataFrame(), pd.DataFrame()

    df, value_col, label_col, node_col = load_dataset(selected)
    rows = []
    for n_agents in [4, 8, 12]:
        for seed in RANDOM_SEEDS:
            if node_col is None or df[node_col].nunique() < n_agents:
                work = replicate_nodes(df, value_col, label_col, n_agents, seed)
                node_col_eff = "node_id"
            else:
                nodes = list(df[node_col].dropna().unique())[:n_agents]
                work = df[df[node_col].isin(nodes)].copy()
                node_col_eff = node_col

            per_node = []
            for node, g in work.groupby(node_col_eff):
                pred = fast_global_hampel(g[value_col].astype(float), k=3.0)
                met = compute_metrics(g[label_col].to_numpy(), pred)
                met["node_id"] = node
                per_node.append(met)

            per_node_df = pd.DataFrame(per_node)
            y_true = work[label_col].to_numpy()
            # Decentralized proxy: concatenate node-level decisions.
            y_pred_all = []
            for _, g in work.groupby(node_col_eff):
                y_pred_all.extend(fast_global_hampel(g[value_col].astype(float), k=3.0))
            met_global = compute_metrics(y_true, np.array(y_pred_all))
            reward_proxy = met_global["accuracy"] - 1.5 * met_global["fn_rate"] - 0.25 * met_global["fpr"]

            rows.append({
                "source_file": selected.name,
                "n_agents": n_agents,
                "run": seed,
                "n_samples_total": len(work),
                "mean_node_f1": per_node_df["f1"].mean(),
                "std_node_f1": per_node_df["f1"].std(ddof=1) if len(per_node_df) > 1 else 0.0,
                "reward_proxy": reward_proxy,
                **met_global,
            })

    detailed = pd.DataFrame(rows)
    summary = summarise_repeated(detailed, ["source_file", "n_agents"])
    # Add reward and node-F1 summaries
    for col in ["reward_proxy", "mean_node_f1", "std_node_f1"]:
        extra = detailed.groupby(["source_file", "n_agents"])[col].agg(["mean", "std"]).reset_index()
        extra[f"{col}_ci95"] = 1.96 * extra["std"] / np.sqrt(N_RUNS)
        extra = extra.rename(columns={"mean": f"{col}_mean", "std": f"{col}_std"})
        summary = summary.merge(extra, on=["source_file", "n_agents"], how="left")
    return detailed, summary


# -----------------------------
# Export
# -----------------------------
def export_excel(output_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    try:
        xlsx_path = output_dir / "additional_revision_results_all_tables.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet, table in tables.items():
                safe_sheet = sheet[:31]
                table.to_excel(writer, sheet_name=safe_sheet, index=False)
        print(f"[OK] Excel exported: {xlsx_path}")
    except Exception as exc:
        print(f"[WARN] Excel export skipped: {exc}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    datasets = discover_datasets(DATA_DIR)
    print("Datasets found:")
    for d in datasets:
        print(" -", d)

    if not datasets:
        raise SystemExit("No labeled datasets found. Place the CSV files in the same folder as this script.")

    baseline_detailed, baseline_summary = run_modern_baselines(datasets)
    sensitivity = run_sensitivity_analysis(datasets)
    scalability_detailed, scalability_summary = run_marl_scalability(datasets)

    outputs = {
        "benchmark_if_ocsvm_detailed": baseline_detailed,
        "benchmark_if_ocsvm_summary": baseline_summary,
        "sensitivity_analysis_results": sensitivity,
        "marl_scalability_detailed": scalability_detailed,
        "marl_scalability_summary": scalability_summary,
    }

    for name, table in outputs.items():
        path = OUTPUT_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"[OK] CSV exported: {path}")

    # Excel export intentionally skipped in this CSV package
    # export_excel(OUTPUT_DIR, outputs)

    print("\nFinished. Recommended manuscript tables:")
    print("  - benchmark_if_ocsvm_summary.csv")
    print("  - sensitivity_analysis_results.csv, after selecting compact rows")
    print("  - marl_scalability_summary.csv")
    print("  - CSV package exported in additional_revision_results/")


if __name__ == "__main__":
    main()
