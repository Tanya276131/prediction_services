"""
Data drift check: compares a batch of new/incoming data against the
training reference set and flags features that have drifted.

This is the piece that separates "I trained a model" from "I understand
models degrade in production." In a real deployment this would run on
a schedule (e.g. daily) against logged production requests.

Run: python -m src.monitoring.drift_check --current data/raw/telco_churn.csv
(defaults to comparing the reference set against itself as a smoke test)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.data.preprocess import TARGET_COL, clean, load_raw

REFERENCE_PATH = Path("data/raw/telco_churn.csv")
REPORT_PATH = Path("monitoring_report.html")
SUMMARY_PATH = Path("monitoring_summary.json")


def run_drift_check(current_path: Path) -> dict:
    reference = clean(load_raw(REFERENCE_PATH))
    current = clean(load_raw(current_path))

    column_mapping = ColumnMapping(target=TARGET_COL)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current,
               column_mapping=column_mapping)
    report.save_html(str(REPORT_PATH))

    result = report.as_dict()
    drift_metric = result["metrics"][0]["result"]
    summary = {
        "dataset_drift_detected": drift_metric["dataset_drift"],
        "share_of_drifted_columns": drift_metric["share_of_drifted_columns"],
        "number_of_drifted_columns": drift_metric["number_of_drifted_columns"],
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Full report written to {REPORT_PATH}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=REFERENCE_PATH,
                         help="CSV of new data to check for drift against the reference set")
    args = parser.parse_args()
    run_drift_check(args.current)
