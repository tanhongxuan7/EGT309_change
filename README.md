# EGT309 Project – Predictive Maintenance (`maintenance_required`)

> **Status:** Task 1 (EDA), data preparation and the model training/evaluation pipeline are complete.
> Model-choice and metric justifications (sections 9–10) are pending.

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
| `src/config.yaml` | TODO | Hyperparameters, search spaces, paths and evaluation settings |
| `src/config.py` | TODO | Loads and validates the YAML config; shared command-line options |
| `src/model_factory.py` | TODO | Builds model pipelines and grid/random searches from the config |
| `src/artifacts.py` | TODO | Loads the prepared splits; saves and loads trained models |
| `src/train.py` | TODO | Tunes, trains and saves each configured model |
| `src/evaluate.py` | TODO | Test-set metrics, confusion matrices and permutation feature importance |
| `src/plots.py` | TODO | Model-comparison charts |
| `run.sh` | TODO | Runs the full pipeline end to end |

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
├── src/
│   ├── ingest.py                   # Load and validate the raw CSV
│   ├── features.py                 # Cleaning and feature engineering
│   ├── prepare_data.py             # Stratified train/test split -> data/prepared/<feature_set>/
│   ├── config.yaml                 # Model, tuning, path and evaluation settings
│   ├── config.py                   # Config loading/validation and shared CLI options
│   ├── model_factory.py            # Builds pipelines and hyperparameter searches
│   ├── artifacts.py                # Prepared-data loading, model save/load
│   ├── train.py                    # Tune, train and save models
│   ├── evaluate.py                 # Test-set metrics and feature importance
│   └── plots.py                    # Comparison charts
├── saved_model/<feature_set>/      # Trained models (.joblib), training_summary.json, config_used.yaml
├── reports/<feature_set>/          # metrics.csv/json, feature_importance.csv, plots
└── run.sh                          # Runs the pipeline end to end
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
Use it only for comparison.

### Train and evaluate the models

```powershell
python -m src.train       # tune + train every enabled model, save to saved_model/sensor/
python -m src.evaluate    # score them on the test set, write reports/sensor/
```

Or run all three steps at once (Git Bash / Linux / Docker): `./run.sh` (or `./run.sh all`).

All settings live in [src/config.yaml](src/config.yaml): fixed hyperparameters (`params`), tuning
search spaces (`search_space`), tuning method/CV folds/scoring, the decision threshold and the paths.
Any value can also be changed for one run without editing the file:

```powershell
python -m src.train --models random_forest                       # train only some models
python -m src.train --set models.random_forest.params.n_estimators=500
python -m src.train --set tuning.method=grid --set tuning.cv_folds=3
python -m src.train --set tuning.enabled=false                   # fixed params only (fast)
python -m src.evaluate --set evaluation.threshold=0.3            # trade precision for recall
python -m src.train --config my_experiment.yaml                  # use another config file
```

To add a model, add an entry under `models:` with the import path of any scikit-learn classifier
(e.g. `estimator: sklearn.tree.DecisionTreeClassifier`); no code changes are needed. Both scripts
validate the config before running and stop with a short error message on unknown keys, invalid
values or missing input files. Run `python -m src.train --help` for every option.

Note: the prepared CSVs already contain imputed temperatures (median from the whole training split),
so the cross-validation folds share that median. The effect is negligible here (MCAR, 14% missing).

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
