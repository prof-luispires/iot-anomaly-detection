#!/usr/bin/env python3
"""
run_all_scenarios_v12.py

Batch para correr a versão v12 do script de filtros+agente
em todos os 6 cenários de temperatura IoT.

Requisitos:
- plot_iot_temp_gps_filters_agent_v12.py na mesma pasta.
- CSVs dos cenários 1 a 6 na mesma pasta.
"""

import subprocess
from pathlib import Path

# Nomes exatos dos ficheiros CSV de entrada
SCENARIOS = [
    "iot_temp_gps_scenario1_stable_spikes.csv",
    "iot_temp_gps_scenario2_impulsive_noise.csv",
    "iot_temp_gps_scenario3_drift_periodic.csv",
    "iot_temp_gps_scenario4_flat_corrupted.csv",
    "iot_temp_gps_scenario5_complex.csv",
    "iot_temp_gps_scenario6_complex_plus.csv",
    "iot_temp_gps_scenario7_heavy_tail_drift.csv",
]

def main():
    base = Path(__file__).parent.resolve()
    script = base / "plot_iot_temp_gps_filters_agent_v12.py"

    if not script.exists():
        raise FileNotFoundError(f"Script não encontrado: {script}")

    out_dir = base / "results_vFinal"
    out_dir.mkdir(exist_ok=True)

    for csv_name in SCENARIOS:
        csv_path = base / csv_name
        if not csv_path.exists():
            print(f"[WARN] CSV não encontrado, a saltar: {csv_path}")
            continue

        print(f"\n[RUN] Processar cenário: {csv_path.name}")
        cmd = [
            "python",
            str(script),
            "--csv", str(csv_path),
            "--out", str(out_dir),
        ]
        # Se quiser ver os gráficos: adicionar "--show", "1"
        subprocess.run(cmd, check=True)
        print(f"[OK] Concluído: {csv_path.name}")

    print("\n[DONE] Todos os cenários disponíveis foram processados para results_v12/")

if __name__ == "__main__":
    main()
