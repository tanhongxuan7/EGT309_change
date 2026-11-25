# EGT309 Project – Predictive Maintenance (`maintenance_required`)

> **Status:** Task 1 (EDA) and Week 7 data preparation are complete. Model training and evaluation are pending.

## 1. Group information

| | |
|---|---|
| **Group Change** | TODO |
| **Tham Hon Shane** | TODO – name |
| **Khoo Jayho** | TODO – name |
| **https://github.com/tanhongxuan7/EGT309_change** | TODO – public repo link |

### Who wrote which file

Each member must write at least one `.py` file for the code-quality assessment.

| File | Author | Description |
|------|--------|-------------|
| `eda.ipynb` | TODO | Task 1 exploratory data analysis |
| `src/ingest.py` | Khoo Jayho (Week 7) | CSV ingestion and validation |
| `src/features.py` | Khoo Jayho (Week 7) | Data cleaning and feature engineering |
| `src/prepare_data.py` | Khoo Jayho (Week 7) | Training/test split and CSV export |
| `src/...py` | TODO (Task 2) | e.g. model training & evaluation |

## 2. Problem statement (summary)

Sensor readings from 50 machines are used to predict whether a machine is due for maintenance
(`maintenance_required`: 0 = No, 1 = Yes), so that the maintenance team can be alerted before an
unscheduled breakdown. At least three models must be built, evaluated and compared, and the features that
contribute most to the target must be identified.

## 3. Project structure

```
.
├── data/
│   └── proj_machine_readings.csv   # raw dataset (100,000 rows, 13 columns)
├── eda.ipynb                       # Task 1 – exploratory data analysis
├── README.md
├── requirements.txt                # Python dependencies
├── src/                            # Data ingestion, feature engineering and split export
├── saved_model/                    # TODO (Task 2) – trained model files
└── run.sh                          # TODO (Task 2) – runs the pipeline end to end
```

## 4. How to run the EDA notebook

Requires **Python 3.11 or newer** (tested with Python 3.12).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook (run from the project root so the relative path data/... resolves)
jupyter notebook eda.ipynb
```

The notebook reads `data/proj_machine_readings.csv` and runs top to bottom in under a minute.

## 5. How to run the pipeline

After installing the dependencies from section 4, run from the repository root in the VS Code terminal.

```powershell
python -m src.prepare_data
```

This reads `data/proj_machine_readings.csv` and writes `train.csv` and `test.csv` under
`data/prepared/sensor/`. The default split is stratified 80/20 with seed 42. Each output includes
`maintenance_required` as its final column. Use `--input`, `--output-dir`, `--test-size` or `--seed`
to override the defaults.

`--feature-set all` writes to `data/prepared/all/` and includes derived columns that leak the target.
Use it only for comparison. For future model cross-validation, fit `ReadingFeatures` inside the model's
scikit-learn pipeline on raw training rows so each fold learns its own imputation median.

## 6. Docker development environment

Build the image from the repository root. Mount the local `data` directory to retain generated CSVs.

```powershell
docker build -t egt309 .
docker run --rm --mount "type=bind,source=$($PWD.Path)\data,target=/notebook/data" egt309 python -m src.prepare_data
```

To start the EDA notebook, run `docker run --rm -p 127.0.0.1:8888:8888 egt309` and open the URL from the container logs.

## 7. Summary of key EDA findings

Full details, statistics and charts are in [eda.ipynb](eda.ipynb).

### Data quality

| Issue | Evidence | Handling |
|-------|----------|----------|
| **Timestamps are day-first** (`D/M/YYYY H:MM`) | First component reaches 31, second only 3; month-first parsing fails on 50,400 rows | Parse with explicit format `%d/%m/%Y %H:%M` |
| **6,703 (6.7%) negative `vibration` values** | Magnitudes match the positive values (KS test p ≈ 0.58); the anomaly label treats < -80 the same as > 80 | Sign error → use `abs(vibration)`; no rows dropped |
| **14,000 (14%) missing `temperature`** | Missingness unrelated to the target, machine or any column (all tests non-significant after Bonferroni correction) → MCAR | Median imputation + `temperature_missing` indicator, fitted on the training split only |
| **Outliers** in temperature / vibration | High-side outliers are 100% maintenance cases | Kept – they are real high-risk events |
| `downtime_risk` | Equals `anomaly_flag` (ρ = 1.00) except 12 rows with 0.88–0.99 | Redundant – dropped |
| `failure_type` | Non-Normal only when `machine_status = 2`; 1,956 failures are labelled "Normal" | Post-failure information – dropped |
| Duplicates | None (every timestamp is unique; one reading per minute) | No action |

### Relationship with the target

1. **Class imbalance:** 19.7% of readings need maintenance (≈ 4 : 1). Always predicting "no" scores 80.3%
   accuracy, so accuracy is not a useful metric.
2. **Target leakage:** the target is exactly
   `anomaly_flag == 1 OR machine_status == 2 OR predicted_remaining_life < 20` (0 exceptions in 100,000 rows).
3. **`anomaly_flag` is a threshold rule:** `temperature > 90 °C OR |vibration| > 80 mm/s`. Every anomaly also has a
   `predicted_remaining_life` below 50 hours.
4. **Sensors:** `temperature` is the strongest sensor (rank-biserial r ≈ 0.34), then `vibration` (r ≈ 0.12), both
   with sharp step effects at 90 °C and 80 mm/s. **`humidity`, `pressure` and `energy_consumption` show no
   relationship** with the target (|r| < 0.01, mutual information ≈ 0) and are uniformly distributed.
5. **No temporal or machine effects:** maintenance rate is flat across hour, weekday, date and machine (chi-square
   p > 0.05), and lag-1 autocorrelation within each machine is ≈ 0, so readings behave as independent observations.
6. **About 45% of positives come only from the failure state** (`machine_status = 2`), which has the same sensor
   averages as normal operation. A leakage-free model that uses only sensor readings therefore has a recall ceiling
   well below 100%.

## 8. Feature engineering (implemented in the pipeline)

| Feature | Definition | Justification |
|---------|------------|---------------|
| `vibration` | `abs(vibration)` | Fixes the sign error |
| `temperature` | Median-imputed (training-split median) | MCAR; on hidden values median was as accurate as mean / per-machine median and better than KNN (RMSE 10.04 vs 10.89 °C). Iterative imputation was only ~2% better (9.83 °C) and relied on the leaky `predicted_remaining_life` column |
| `temperature_missing` | 1 if temperature was missing | Tells the model a value was imputed; 922 missing values hid readings above 90 °C |
| `temp_above_90` | 1 if temperature > 90 | Maintenance rate jumps from ~12–17% to 100% above 90 °C |
| `vibration_above_80` | 1 if \|vibration\| > 80 | Maintenance rate jumps from ~16–19% to 100% above 80 mm/s |

**Excluded from the main feature set:** `anomaly_flag`, `downtime_risk`, `machine_status`, `failure_type`,
`predicted_remaining_life` (target leakage / post-failure information), and `timestamp`, `machine_id`
(no signal). The pipeline should allow a configurable `all` feature set that includes the derived columns,
for comparison only.

## 9. Choice of models

**TODO (Task 2).** Recommendations from the EDA:

- The relationships are threshold-shaped (step changes at 90 °C and 80 mm/s), which favours **tree-based models**
  (e.g. Random Forest, Gradient Boosting such as `HistGradientBoostingClassifier`, which also handles missing
  values natively).
- **Logistic Regression** (with standard scaling and the threshold features) as an interpretable baseline.
- Use a **stratified** train/test split and stratified cross-validation (19.7% positive class; no time
  dependence, so a random split is appropriate).
- Hyperparameter tuning method and justification: TODO.

## 10. Evaluation metrics

**TODO (Task 2)** – final choice and results. Recommendation from the EDA:

- **Recall** on the "maintenance required" class is the priority, because a missed case means unscheduled
  downtime.
- Report it together with **precision**, **F1**, **PR-AUC** and the **confusion matrix**, so the false-alert rate
  for the maintenance team is visible too.
- Do **not** rely on accuracy because of the class imbalance.
