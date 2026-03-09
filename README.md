Adaptive Statistical Filtering with Reinforcement Learning for IoT Anomaly Detection

This repository contains the source code and experimental assets associated with the report:

“Adaptive Statistical Filtering for IoT Time-Series Anomaly Detection using Reinforcement Learning”

The project investigates how classical statistical filters can be combined with reinforcement learning (RL) to enable adaptive and interpretable anomaly detection in IoT temperature time-series data.

Project Overview
The proposed framework evolves through three main stages:

Static statistical filtering Hampel, IQR, and Z-Score filters with fixed parameters.

Reinforcement Learning with filter and parameter adaptation A single-agent RL approach selects both the filter type and its parameter configuration.

Multi-Agent Reinforcement Learning (MARL) One agent per scenario learns independently, reducing policy interference and improving training stability.

The system is evaluated on four synthetic IoT scenarios representing common anomaly patterns (spikes, impulsive noise, drift, and corrupted signals).

Repository Structure . ├── data/ │ ├── iot_temp_gps_scenario1_stable_spikes_labeled.csv │ ├── iot_temp_gps_scenario2_impulsive_noise_labeled.csv │ ├── iot_temp_gps_scenario3_drift_periodic_labeled.csv │ └── iot_temp_gps_scenario4_flat_corrupted_labeled.csv │ ├── src/ │ ├── statistical_filters.py │ ├── rl_filter_selection.py │ ├── rl_training_single.py │ ├── rl_training_multiagent.py │ └── evaluation_metrics.py │ ├── results/ │ ├── rl_results_single.xlsx │ ├── rl_results_multiagent.xlsx │ └── Chapter9_Table9_1_Figure9_1_EDITABLE.xlsx │ ├── figures/ │ ├── Figure_8_1_TrainingReward_PerAgent.png │ └── Figure_9_1_Reward_Comparison.png │ ├── diagrams/ │ ├── UML_Class_Diagram.drawio │ └── Sequence_Training_Episode.drawio │ ├── requirements.txt └── README.md

Requirements

Python ≥ 3.9

NumPy

Pandas

Matplotlib

XlsxWriter

Install dependencies with:

pip install -r requirements.txt

How to Reproduce the Experiments Step 1 – Run static filter baselines python src/statistical_filters.py
Step 2 – Train RL (filter + parameter adaptation) python src/rl_training_single.py

Step 3 – Train Multi-Agent RL python src/rl_training_multiagent.py

Step 4 – Generate comparison tables and figures python src/evaluation_metrics.py

This step produces Table 9.1 and Figure 9.1 in Excel and image formats.

Mapping to the Report
Figure 8.1 – Training reward evolution per agent

Table 8.1 – Optimal filter–parameter configuration per scenario

Table 9.1 – Quantitative comparison across methods

Figure 9.1 – Reward comparison by scenario and method

Reproducibility Notes
Random seeds are fixed where applicable.

All results are derived directly from the provided datasets.

No external or proprietary data is required.

License
This project is provided for academic and educational purposes.
