"""Strict SINCON submission validation; never repairs or prunes input rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["example_id", "boardings"]


def _read(frame_or_path) -> pd.DataFrame:
    if isinstance(frame_or_path, pd.DataFrame):
        return frame_or_path.copy()
    return pd.read_csv(frame_or_path, dtype={"example_id": "string"})


def _ids(frame: pd.DataFrame, name: str) -> pd.Series:
    if "example_id" not in frame:
        raise ValueError(f"{name}: example_id column is missing.")
    ids = frame["example_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: example_id contains missing/empty values.")
    if not ids.map(lambda x: isinstance(x, str)).all():
        raise ValueError(f"{name}: example_id must be strings; preserve original identifiers.")
    if ids.duplicated().any():
        raise ValueError(f"{name}: duplicate example_id values.")
    return ids.astype(str).reset_index(drop=True)


def _distribution(values) -> dict:
    v = np.asarray(values, dtype=float).reshape(-1)
    if len(v) == 0 or not np.isfinite(v).all():
        raise ValueError("Distribution input must contain finite predictions.")
    return {
        "count": len(v), "min": float(v.min()), "mean": float(v.mean()),
        "p01": float(np.quantile(v, 0.01)), "p50": float(np.quantile(v, 0.5)),
        "p99": float(np.quantile(v, 0.99)), "max": float(v.max()),
        "zero_fraction": float((v == 0).mean()),
    }


def validate_submission(submission, test, sample, scope: str = "test", reference_predictions=None) -> dict:
    """Validate exact schema, scope-specific ID order, values, and scope mismatch.

    ``test`` is the officially prescribed 123-day scope and expects every test
    ID. ``template`` checks the stale 92-day sample only for diagnostics; passing
    that check does not make its file ready for the official full-test scope.
    The discrepancy is reported in either mode and never silently resolved.
    """
    if scope not in ("template", "test"):
        raise ValueError("scope must be 'template' or 'test'.")
    sub, tst, tmpl = (_read(x) for x in (submission, test, sample))
    if list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Submission columns/order must equal {EXPECTED_COLUMNS}, got {list(sub.columns)}.")
    if list(tmpl.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Sample columns/order must equal {EXPECTED_COLUMNS}.")
    sub_ids, test_ids, sample_ids = _ids(sub, "submission"), _ids(tst, "test"), _ids(tmpl, "sample")
    if len(sub) == 0:
        raise ValueError("Submission cannot be empty.")
    expected = sample_ids if scope == "template" else test_ids
    if len(sub_ids) != len(expected):
        raise ValueError(f"Submission has {len(sub_ids)} rows but {scope} scope requires {len(expected)}.")
    if not np.array_equal(sub_ids.to_numpy(), expected.to_numpy()):
        raise ValueError(f"Submission example_id values/order differ from {scope} scope.")
    pred = pd.to_numeric(sub["boardings"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Submission predictions contain NaN or infinite values.")
    if (pred < 0).any():
        raise ValueError("Submission predictions contain negative boardings.")
    test_set, sample_set = set(test_ids), set(sample_ids)
    missing_from_template = test_set - sample_set
    unknown_template_ids = sample_set - test_set
    if unknown_template_ids:
        raise ValueError(f"Sample contains {len(unknown_template_ids)} IDs absent from test.")
    warnings = []
    scope_agrees = np.array_equal(test_ids.to_numpy(), sample_ids.to_numpy())
    if not scope_agrees:
        warnings.append(
            f"Stale sample template: sample={len(sample_ids)} rows, test={len(test_ids)} rows; "
            f"{len(missing_from_template)} test IDs are absent from the sample. "
            "Official submission scope is the full test ID sequence. "
            + ("Validated that full test scope." if scope == "test" else "Template scope is diagnostic only and is not the official submission scope.")
        )
    mismatch = {"template_rows": len(sample_ids), "test_rows": len(test_ids), "test_ids_absent_from_template": len(missing_from_template)}
    if "date" in tst and missing_from_template:
        dates = tst.loc[tst["example_id"].astype(str).isin(missing_from_template), "date"].astype(str)
        mismatch.update({"missing_period_start": dates.min(), "missing_period_end": dates.max()})
    result = {
        "valid": True, "scope": scope, "rows": len(sub), "columns": list(sub.columns),
        "official_scope": "test", "matches_official_scope": scope == "test",
        "exact_id_order": True, "finite_nonnegative": True,
        "template_matches_test": bool(scope_agrees), "scope_discrepancy": mismatch,
        "warnings": warnings, "prediction_distribution": _distribution(pred),
    }
    if reference_predictions is not None:
        ref = _distribution(reference_predictions)
        result["reference_prediction_distribution"] = ref
        result["distribution_comparison"] = {
            "mean_ratio": result["prediction_distribution"]["mean"] / ref["mean"] if ref["mean"] != 0 else None,
            "median_ratio": result["prediction_distribution"]["p50"] / ref["p50"] if ref["p50"] != 0 else None,
            "note": "Diagnostic only: forecast dates/composition differ; ratios are not automatic rejection rules.",
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--scope", choices=("template", "test"), default="test")
    parser.add_argument("--reference", type=Path, help="Optional validation prediction CSV.")
    parser.add_argument("--reference-column", default="prediction")
    parser.add_argument("--report", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)
    try:
        ref = None
        if args.reference:
            reference = pd.read_csv(args.reference)
            ref = reference[args.reference_column].to_numpy()
        result = validate_submission(args.submission, args.test, args.sample, args.scope, ref)
    except (ValueError, KeyError, OSError) as error:
        print(f"Submission validation failed: {error}", file=sys.stderr)
        return 2
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    print(content)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(content + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
