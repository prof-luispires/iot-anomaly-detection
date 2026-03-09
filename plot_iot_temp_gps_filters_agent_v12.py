#!/usr/bin/env python3
"""
plot_iot_temp_gps_filters_agent_v12.py
--------------------------------------
Versão 12 – simples e determinística:

- Aplica Hampel, IQR e Z-Score a um CSV (timestamp, temp_c).
- Calcula métricas básicas para cada filtro.
- Usa uma função de custo fixa para obter:
    best_filter_agent_raw_v12  (decisão "pura" do agente)

- Usa um mapa de aprendizagem supervisionada para 4 cenários:
    scenario1_stable_spikes      -> IQR
    scenario2_impulsive_noise    -> IQR
    scenario3_drift_periodic     -> Hampel
    scenario4_flat_corrupted     -> Hampel

- Define:
    best_filter_supervised_v12  (ground truth, ou "N/A")
    best_filter_agent_v12       (agent final, corrige pelo supervised se existir)

Outputs (na pasta --out, default: ./results_v12):

    <stem>_with_filters_v12.csv
    <stem>_outlier_summary_v12.txt
    <stem>_filters_results_v12_agent.xlsx
    <stem>_filters_results_v12_supervised.xlsx
    hampel_filter_v12.png
    iqr_filter_v12.png
    zscore_filter_v12.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Mapa supervisionado – nomes exatos dos ficheiros
# ---------------------------------------------------------------------

SUPERVISED_BY_FILENAME = {
    "iot_temp_gps_scenario1_stable_spikes.csv": "iqr",
    "iot_temp_gps_scenario2_impulsive_noise.csv": "iqr",
    "iot_temp_gps_scenario3_drift_periodic.csv": "hampel",
    "iot_temp_gps_scenario4_flat_corrupted.csv": "hampel",
}


def get_supervised_label(csv_name: str) -> str:
    """Devolve filtro supervisionado ('hampel'/'iqr') ou 'N/A'."""
    return SUPERVISED_BY_FILENAME.get(csv_name, "N/A")


# ---------------------------------------------------------------------
# Utilitários de paths e leitura
# ---------------------------------------------------------------------

def infer_csv_path(csv_arg: str | None) -> Path:
    if csv_arg:
        return Path(csv_arg).expanduser().resolve()

    here = Path(__file__).parent
    # fallback: procurar um dos cenários
    for name in SUPERVISED_BY_FILENAME.keys():
        cand = here / name
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        "Não foi possível encontrar o CSV. "
        "Use --csv para indicar explicitamente o ficheiro."
    )


def resolve_out_dir(out_arg: str | None, script_path: Path) -> Path:
    if out_arg:
        p = Path(out_arg).expanduser().resolve()
    else:
        p = script_path.parent / "results_v12"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" not in df.columns or "temp_c" not in df.columns:
        raise ValueError("CSV deve conter colunas 'timestamp' e 'temp_c'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Filtros estatísticos
# ---------------------------------------------------------------------

def hampel_filter(series: pd.Series, window: int = 21, n_sigmas: float = 3.0):
    x = series.astype(float).copy()
    k = window // 2
    med = x.rolling(window=window, center=True, min_periods=k).median()
    mad = (x - med).abs().rolling(window=window, center=True, min_periods=k).median()
    sigma = 1.4826 * mad
    thresh = n_sigmas * sigma
    flags = (x - med).abs() > thresh
    clean = x.where(~flags, med)
    return clean, flags.fillna(False)


def iqr_filter(series: pd.Series, window: int = 21, k: float = 1.5):
    x = series.astype(float).copy()
    half = window // 2
    q1 = x.rolling(window=window, center=True, min_periods=half).quantile(0.25)
    q3 = x.rolling(window=window, center=True, min_periods=half).quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    flags = (x < lower) | (x > upper)

    clean_nan = x.where(~flags, np.nan)
    med_roll = x.rolling(window=window, center=True, min_periods=half).median()
    repaired = clean_nan.fillna(med_roll)
    return clean_nan, repaired, flags.fillna(False)


def zscore_filter(series: pd.Series, window: int = 21, zthr: float = 3.0):
    x = series.astype(float).copy()
    half = window // 2
    mean = x.rolling(window=window, center=True, min_periods=half).mean()
    std = x.rolling(window=window, center=True, min_periods=half).std(ddof=0)
    z = (x - mean).abs() / std.replace(0, np.nan)
    flags = z > zthr
    clean_nan = x.where(~flags, np.nan)
    med_roll = x.rolling(window=window, center=True, min_periods=half).median()
    repaired = clean_nan.fillna(med_roll)
    return clean_nan, repaired, flags.fillna(False), z


# ---------------------------------------------------------------------
# Plot simples
# ---------------------------------------------------------------------

def plot_compare(time_min, orig, cleaned, flags, out_dir: Path, name: str, show: int):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_min, orig, lw=1, label="original")
    ax.plot(time_min, cleaned, lw=1, label="cleaned")
    try:
        ax.scatter(time_min[flags], orig[flags], s=20, marker="x", label="outlier")
    except Exception:
        pass

    ax.set_title(name.replace("_", " ").title())
    ax.set_xlabel("Time [minutes]")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_v12.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# Métricas + custo
# ---------------------------------------------------------------------

def base_metrics(orig: pd.Series, repaired: pd.Series, flags: pd.Series) -> dict:
    orig = orig.astype(float)
    repaired = repaired.astype(float)
    diff = repaired - orig
    rmse = float(np.sqrt(np.nanmean(diff**2)))
    frac_outliers = float(flags.mean())
    return {"rmse": rmse, "frac_outliers": frac_outliers}


def extra_metrics(orig: pd.Series, repaired: pd.Series) -> dict:
    x_o = orig.astype(float).dropna()
    x_r = repaired.astype(float).dropna()
    n = min(len(x_o), len(x_r))
    if n < 4:
        return {"var_repaired": 0.0, "slope_diff": 0.0, "spikes_resid_rate": 0.0}
    x_o = x_o.iloc[:n]
    x_r = x_r.iloc[:n]
    idx = np.arange(n)
    slope_o, _ = np.polyfit(idx, x_o.values, 1)
    slope_r, _ = np.polyfit(idx, x_r.values, 1)
    slope_diff = abs(slope_r - slope_o)
    var_repaired = float(np.var(x_r))
    med_r = float(x_r.median())
    mad_r = float((x_r - med_r).abs().median())
    if mad_r == 0:
        spikes_resid_rate = 0.0
    else:
        thresh = 3.0 * 1.4826 * mad_r
        spikes_resid_rate = float(((x_r - med_r).abs() > thresh).mean())
    return {
        "var_repaired": var_repaired,
        "slope_diff": slope_diff,
        "spikes_resid_rate": spikes_resid_rate,
    }


def cost_function(m: dict, std_orig: float) -> float:
    """
    Custo simples com pesos fixos:
        - penaliza RMSE
        - penaliza muitos outliers
        - penaliza var_repaired demasiado grande
        - penaliza slope_diff e spikes_resid
    """
    w_rmse = 1.0
    w_frac = 1.0
    w_var = 0.5
    w_slope = 1.0
    w_spikes = 1.0

    rmse_norm = m["rmse"] / (std_orig + 1e-6)
    var_norm = m["var_repaired"] / (std_orig**2 + 1e-6)
    slope_norm = m["slope_diff"] / (abs(m["slope_diff"]) + 1e-6)

    return float(
        w_rmse * rmse_norm
        + w_frac * m["frac_outliers"]
        + w_var * var_norm
        + w_slope * slope_norm
        + w_spikes * m["spikes_resid_rate"]
    )


def select_best_filter(metrics: dict) -> str:
    best_name = None
    best_score = float("inf")
    for name, vals in metrics.items():
        if vals["score"] < best_score:
            best_score = vals["score"]
            best_name = name
    return best_name


# ---------------------------------------------------------------------
# Resumo TXT + Excels
# ---------------------------------------------------------------------

def build_summary_df(metrics: dict, flags_dict: dict, total: int) -> pd.DataFrame:
    rows = []
    for m in ["hampel", "iqr", "zscore"]:
        n_out = int(flags_dict[m].sum())
        frac = n_out / total if total > 0 else 0.0
        rows.append({
            "method": m,
            "n_outliers": n_out,
            "frac_outliers": frac,
            "rmse": metrics[m]["rmse"],
            "var_repaired": metrics[m]["var_repaired"],
            "slope_diff": metrics[m]["slope_diff"],
            "spikes_resid_rate": metrics[m]["spikes_resid_rate"],
            "score_agent": metrics[m]["score"],
        })
    return pd.DataFrame(rows)


def save_summary_v12(stem: str,
                     df,
                     flags_dict,
                     metrics,
                     best_agent_raw,
                     best_agent_final,
                     best_supervised,
                     out_dir: Path) -> Path:
    total = len(df)
    lines = []
    lines.append(f"# Outlier summary for {stem}")
    for key, mask in flags_dict.items():
        n = int(mask.sum())
        lines.append(f"{key}: {n} outliers ({(n/total*100):.2f}%)")
    lines.append("")
    lines.append("# Filter metrics (v12, fixed-cost agent)")
    for name, vals in metrics.items():
        lines.append(
            f"{name}: rmse={vals['rmse']:.4f}, "
            f"frac_outliers={vals['frac_outliers']:.4f}, "
            f"var_repaired={vals['var_repaired']:.4f}, "
            f"slope_diff={vals['slope_diff']:.4f}, "
            f"spikes_resid_rate={vals['spikes_resid_rate']:.4f}, "
            f"score_agent={vals['score']:.4f}"
        )
    lines.append("")
    lines.append(f"# Agent raw decision (v12): {best_agent_raw}")
    lines.append(f"# Agent final decision (v12): {best_agent_final}")
    lines.append(f"# Supervised decision (v12): {best_supervised}")

    summary_path = out_dir / f"{stem}_outlier_summary_v12.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary_path


def save_excel_agent_v12(stem: str,
                         out_dir: Path,
                         out_df: pd.DataFrame,
                         metrics: dict,
                         flags_dict: dict,
                         best_agent_raw: str,
                         best_agent_final: str,
                         best_supervised: str) -> Path:
    out_xlsx = out_dir / f"{stem}_filters_results_v12_agent.xlsx"
    total = len(out_df)
    summary_df = build_summary_df(metrics, flags_dict, total)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter",
                        datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
        out_df.to_excel(writer, sheet_name="Data", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        workbook = writer.book
        ws = writer.sheets["Summary"]

        base_row = len(summary_df) + 2
        ws.write(base_row + 0, 0, "Best filter (agent raw v12):")
        ws.write(base_row + 0, 1, best_agent_raw)
        ws.write(base_row + 1, 0, "Best filter (agent final v12):")
        ws.write(base_row + 1, 1, best_agent_final)
        ws.write(base_row + 2, 0, "Best filter (supervised v12):")
        ws.write(base_row + 2, 1, best_supervised)

        # Gráfico temp vs time_min
        data_ws = writer.sheets["Data"]
        n_rows = len(out_df)
        chart = workbook.add_chart({"type": "line"})
        chart.add_series({
            "name":       ["Data", 0, 2],               # temp_c header
            "categories": ["Data", 1, 1, n_rows, 1],   # time_min
            "values":     ["Data", 1, 2, n_rows, 2],   # temp_c
        })
        chart.add_series({
            "name":       ["Data", 0, 3],              # temp_hampel
            "categories": ["Data", 1, 1, n_rows, 1],
            "values":     ["Data", 1, 3, n_rows, 3],
        })
        chart.add_series({
            "name":       ["Data", 0, 4],              # temp_iqr
            "categories": ["Data", 1, 1, n_rows, 1],
            "values":     ["Data", 1, 4, n_rows, 4],
        })
        chart.add_series({
            "name":       ["Data", 0, 5],              # temp_zscore
            "categories": ["Data", 1, 1, n_rows, 1],
            "values":     ["Data", 1, 5, n_rows, 5],
        })
        chart.set_title({"name": "Temperature: Original vs Filters (v12 - agent)"})
        chart.set_x_axis({"name": "Time [minutes]"})
        chart.set_y_axis({"name": "Temperature (°C)"})
        chart.set_legend({"position": "bottom"})
        ws.insert_chart("G2", chart)

    return out_xlsx


def save_excel_supervised_v12(stem: str,
                              out_dir: Path,
                              out_df: pd.DataFrame,
                              metrics: dict,
                              flags_dict: dict,
                              best_supervised: str) -> Path:
    out_xlsx = out_dir / f"{stem}_filters_results_v12_supervised.xlsx"
    total = len(out_df)
    summary_df = build_summary_df(metrics, flags_dict, total)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter",
                        datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
        out_df.to_excel(writer, sheet_name="Data", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        workbook = writer.book
        ws = writer.sheets["Summary"]

        base_row = len(summary_df) + 2
        ws.write(base_row, 0, "Best filter (supervised v12):")
        ws.write(base_row, 1, best_supervised)

        data_ws = writer.sheets["Data"]
        n_rows = len(out_df)
        chart = workbook.add_chart({"type": "line"})
        chart.add_series({
            "name":       ["Data", 0, 2],  # temp_c
            "categories": ["Data", 1, 1, n_rows, 1],
            "values":     ["Data", 1, 2, n_rows, 2],
        })
        chart.add_series({
            "name":       ["Data", 0, 3],  # hampel
            "categories": ["Data", 1, 1, n_rows, 1],
            "values":     ["Data", 1, 3, n_rows, 3],
        })
        chart.add_series({
            "name":       ["Data", 0, 4],  # iqr
            "categories": ["Data", 1, 1, n_rows, 1],
            "values":     ["Data", 1, 4, n_rows, 4],
        })
        chart.add_series({
            "name":       ["Data", 0, 5],  # zscore
            "categories": ["Data", 1, 1, n_rows, 1],
            "values":     ["Data", 1, 5, n_rows, 5],
        })
        chart.set_title({"name": "Temperature: Original vs Filters (v12 - supervised)"})
        chart.set_x_axis({"name": "Time [minutes]"})
        chart.set_y_axis({"name": "Temperature (°C)"})
        chart.set_legend({"position": "bottom"})
        ws.insert_chart("G2", chart)

    return out_xlsx


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="IoT temp+GPS filters with agent + supervised (v12)"
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Caminho para CSV (timestamp,temp_c)")
    parser.add_argument("--out", type=str, default=None,
                        help="Diretoria de saída (default ./results_v12)")
    parser.add_argument("--show", type=int, default=0,
                        help="1 para mostrar gráficos, 0 para só guardar")
    parser.add_argument("--win", type=int, default=21,
                        help="Janela para filtros (ímpar)")
    parser.add_argument("--hthr", type=float, default=3.0,
                        help="n_sigmas para Hampel")
    parser.add_argument("--zthr", type=float, default=3.0,
                        help="|z| threshold para Z-Score")
    parser.add_argument("--iqrk", type=float, default=1.5,
                        help="k do IQR (Q1-k*IQR, Q3+k*IQR)")
    args = parser.parse_args(cli_args)

    script_path = Path(__file__).resolve()
    csv_path = infer_csv_path(args.csv)
    out_dir = resolve_out_dir(args.out, script_path)

    print(f"[INFO v12] CSV: {csv_path}")
    print(f"[INFO v12] Output dir: {out_dir}")

    df = read_csv(csv_path)
    ts = df["timestamp"]
    temp = df["temp_c"]
    time_min = (ts - ts.iloc[0]).dt.total_seconds() / 60.0
    std_orig = float(np.nanstd(temp))

    # Aplicar filtros
    hampel_clean, hampel_flags = hampel_filter(temp, window=args.win, n_sigmas=args.hthr)
    _, iqr_repaired, iqr_flags = iqr_filter(temp, window=args.win, k=args.iqrk)
    _, z_repaired, z_flags, z_vals = zscore_filter(temp, window=args.win, zthr=args.zthr)

    # Plots simples
    stem = csv_path.stem

    plot_compare(time_min, temp, hampel_clean, hampel_flags, out_dir, f"{stem}_hampel", args.show)
    plot_compare(time_min, temp, iqr_repaired, iqr_flags, out_dir, f"{stem}_iqr", args.show)
    plot_compare(time_min, temp, z_repaired, z_flags, out_dir, f"{stem}_zscore", args.show)


    # Métricas + custos
    metrics = {}

    m_h = base_metrics(temp, hampel_clean, hampel_flags)
    m_h.update(extra_metrics(temp, hampel_clean))
    metrics["hampel"] = m_h

    m_i = base_metrics(temp, iqr_repaired, iqr_flags)
    m_i.update(extra_metrics(temp, iqr_repaired))
    metrics["iqr"] = m_i

    m_z = base_metrics(temp, z_repaired, z_flags)
    m_z.update(extra_metrics(temp, z_repaired))
    metrics["zscore"] = m_z

    for name, m in metrics.items():
        m["score"] = cost_function(m, std_orig)

    best_agent_raw = select_best_filter(metrics)
    print(f"[AGENT v12] Agent raw decision: {best_agent_raw}")

    csv_name = csv_path.name
    best_supervised = get_supervised_label(csv_name)
    print(f"[SUPERVISED v12] Supervised decision for {csv_name}: {best_supervised}")

    if best_supervised != "N/A":
        best_agent_final = best_supervised
        print("[AGENT v12] Agent final decision overridden by supervised label.")
    else:
        best_agent_final = best_agent_raw

    out_df = pd.DataFrame({
        "timestamp": ts,
        "time_min": time_min,
        "temp_c": temp,
        "temp_hampel": hampel_clean,
        "temp_iqr": iqr_repaired,
        "temp_zscore": z_repaired,
        "is_outlier_hampel": hampel_flags.astype(int),
        "is_outlier_iqr": iqr_flags.astype(int),
        "is_outlier_zscore": z_flags.astype(int),
        "zscore_abs": z_vals,
    })
    out_df["best_filter_agent_raw_v12"] = best_agent_raw
    out_df["best_filter_agent_v12"] = best_agent_final
    out_df["best_filter_supervised_v12"] = best_supervised

    out_csv = out_dir / (csv_path.stem + "_with_filters_v12.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO v12] CSV com filtros guardado em: {out_csv}")

    flags_dict = {
        "hampel": hampel_flags,
        "iqr": iqr_flags,
        "zscore": z_flags,
    }

    summary_path = save_summary_v12(
        stem=csv_path.stem,
        df=df,
        flags_dict=flags_dict,
        metrics=metrics,
        best_agent_raw=best_agent_raw,
        best_agent_final=best_agent_final,
        best_supervised=best_supervised,
        out_dir=out_dir,
    )
    print(f"[INFO v12] Resumo TXT em: {summary_path}")

    xlsx_agent = save_excel_agent_v12(
        stem=csv_path.stem,
        out_dir=out_dir,
        out_df=out_df,
        metrics=metrics,
        flags_dict=flags_dict,
        best_agent_raw=best_agent_raw,
        best_agent_final=best_agent_final,
        best_supervised=best_supervised,
    )
    print(f"[INFO v12] Excel (agent) em: {xlsx_agent}")

    xlsx_sup = save_excel_supervised_v12(
        stem=csv_path.stem,
        out_dir=out_dir,
        out_df=out_df,
        metrics=metrics,
        flags_dict=flags_dict,
        best_supervised=best_supervised,
    )
    print(f"[INFO v12] Excel (supervised) em: {xlsx_sup}")


if __name__ == "__main__":
    main()
