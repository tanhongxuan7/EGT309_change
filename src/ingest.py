"""Load and validate the project's labeled machine readings."""

from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "proj_machine_readings.csv"
TARGET = "maintenance_required"
SENSORS = ("temperature", "vibration", "humidity", "pressure", "energy_consumption")
DERIVED = ("machine_status", "anomaly_flag", "predicted_remaining_life", "failure_type", "downtime_risk")


def load_readings(path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Read the source CSV and parse day-first timestamps."""
    readings = pd.read_csv(path)
    required = {"timestamp", "machine_id", TARGET, *SENSORS, *DERIVED}
    missing = required.difference(readings.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

    readings["timestamp"] = pd.to_datetime(
        readings["timestamp"], format="%d/%m/%Y %H:%M"
    )
    numeric_columns = (*SENSORS, "machine_id", TARGET, *DERIVED)
    for column in numeric_columns:
        if column != "failure_type":
            readings[column] = pd.to_numeric(readings[column])
    required_values = required - {"temperature"}
    if readings.loc[:, sorted(required_values)].isna().any().any():
        raise ValueError("Only temperature may contain missing values.")
    if set(readings[TARGET].unique()) != {0, 1}:
        raise ValueError(f"{TARGET} must contain both labels 0 and 1.")
    return readings
