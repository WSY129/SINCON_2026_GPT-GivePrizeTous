#!/usr/bin/env python3
"""Build a self-contained Kaggle/Colab notebook from the current project source.

Only source, tests, the default configuration, requirements, and public holiday
provenance are embedded. No competition CSV, model weights, token, or local
environment is included. Re-run after source changes to refresh the snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cell(kind: str, text: str, number: int) -> dict:
    content = text.strip() + "\n"
    result = {
        "cell_type": kind,
        "id": hashlib.sha256(f"{number}:{content}".encode()).hexdigest()[:12],
        "metadata": {},
        "source": content.splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(project_root: Path, output: Path) -> dict:
    project_root = project_root.resolve()
    paths = sorted((project_root / "src").glob("*.py"))
    paths += sorted((project_root / "tests").glob("test_*.py"))
    paths += [project_root / "configs/default.json", project_root / "configs/three_tree.json", project_root / "requirements.txt"]
    provenance = project_root / "reports/holiday_sources.md"
    if provenance.is_file():
        paths.append(provenance)
    embedded = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required notebook source missing: {path}")
        relative = path.relative_to(project_root).as_posix()
        source = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            compile(source, relative, "exec")
        embedded[relative] = source
    required = {"src/__init__.py", "src/eda.py", "src/pipeline.py", "src/validate_submission.py", "configs/default.json", "requirements.txt"}
    if not required.issubset(embedded):
        raise ValueError(f"Missing essential notebook files: {sorted(required - set(embedded))}")
    config = json.loads(embedded["configs/default.json"])
    if config.get("selection_metric") != "rmsle" or config.get("horizon_days") != 123:
        raise ValueError("Notebook snapshot must use the official RMSLE / 123-day setup.")
    manifest = {name: sha256(source) for name, source in sorted(embedded.items())}
    snapshot = sha256(json.dumps(manifest, sort_keys=True))
    payload_literal = "{\n" + "".join(f"    {name!r}: {content!r},\n" for name, content in embedded.items()) + "}"

    blocks = []
    def add(kind, source):
        blocks.append(cell(kind, source, len(blocks)))

    add("markdown", f"""
# SINCON 2026 재현 노트북

인천 도시철도 역×날짜×시간대별 **승차 인원(boardings)**을 예측합니다.
공식 평가지표는 **RMSLE**이며, `log1p(boardings)`의 MSE를 학습합니다.
2026-07-01~2026-10-31의 123일, **전체 test ID 157,194개**가 제출 대상입니다.
로컬 `sample_submission.csv`는 92일만 포함한 오래된 양식이므로 제출 범위를 결정하는 데 쓰지 않습니다.

이 파일 하나에 프로젝트 코드·테스트·설정·의존성 목록을 포함했습니다.
경쟁 데이터, 사전학습 가중치, 계정 인증정보는 포함하지 않았습니다.
공식 과제·규정은 사용자가 제공한 [대회 안내](https://www.kaggle.com/competitions/sincon-2026/overview)를 기준으로 반영했습니다.

- Kaggle: 대회 데이터 4개 CSV를 Add Input으로 연결한 뒤 순서대로 실행합니다.
- Colab: 데이터 폴더를 준비하고 아래 `DATA_DIR_OVERRIDE`를 지정합니다.
- 모델은 기본 CPU에서 실행합니다. LightGBM 16개 설정을 최대 1,000회·early stopping 100회로 탐색합니다. 마지막 fold는 재사용 검증 구간입니다.
- 자동 제출은 하지 않습니다. 마지막 셀에서 검수한 `submission.csv`를 제공합니다.

소스 스냅샷 SHA-256: `{snapshot}`
""")
    add("markdown", """
## 1. 코드 복원

기본 작업 경로는 Kaggle의 `/kaggle/working/SINCON`, Colab의 `/content/SINCON`입니다.
아래 셀을 다시 실행하면 포함된 소스·테스트·설정 파일을 현재 스냅샷으로 복원합니다.
데이터 파일은 복사하거나 변경하지 않습니다. 경로를 바꾸려면 `PROJECT_ROOT_OVERRIDE`만 수정합니다.
""")
    bootstrap = '''from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT_OVERRIDE = ""  # Optional absolute path for a different runtime.
IS_KAGGLE = Path("/kaggle/working").is_dir()
PROJECT_ROOT = Path(PROJECT_ROOT_OVERRIDE) if PROJECT_ROOT_OVERRIDE else Path("/kaggle/working/SINCON" if IS_KAGGLE else "/content/SINCON")
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

EMBEDDED_FILES = __PAYLOAD__
SOURCE_SHA256 = __MANIFEST__
for relative, content in EMBEDDED_FILES.items():
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe embedded relative path: {relative}")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != SOURCE_SHA256[relative]:
        raise ValueError(f"Embedded source checksum mismatch: {relative}")
    destination = PROJECT_ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
(PROJECT_ROOT / "source_manifest.json").write_text(json.dumps(SOURCE_SHA256, indent=2), encoding="utf-8")
print(f"Restored {len(EMBEDDED_FILES)} files to {PROJECT_ROOT}")
print("Python:", sys.version)
'''.replace("__PAYLOAD__", payload_literal).replace("__MANIFEST__", repr(manifest))
    add("code", bootstrap)

    add("markdown", """
## 2. 데이터 경로 확인

Kaggle에서는 연결된 입력 중 네 파일이 모두 있고 헤더가 일치하는 폴더를 찾습니다.
여러 후보가 있으면 임의로 선택하지 않으므로 `DATA_DIR_OVERRIDE`에 사용할 경로를 지정합니다.
Colab에서는 업로드·마운트한 CSV 폴더의 절대 경로를 입력합니다.
""")
    add("code", '''DATA_DIR_OVERRIDE = ""  # Colab example: "/content/sincon-2026"
EXPECTED_HEADERS = {
    "train.csv": ["date", "station_id", "hour", "boardings", "alightings"],
    "test.csv": ["example_id", "date", "station_id", "hour"],
    "metaData.csv": ["station_id", "station_name", "canonical_line", "first_date", "last_date"],
    "sample_submission.csv": ["example_id", "boardings"],
}

def matches_input(directory):
    directory = Path(directory)
    for filename, expected_header in EXPECTED_HEADERS.items():
        path = directory / filename
        if not path.is_file():
            return False
        with path.open(encoding="utf-8-sig", newline="") as handle:
            if next(csv.reader(handle), None) != expected_header:
                return False
    return True

if DATA_DIR_OVERRIDE:
    DATA_DIR = Path(DATA_DIR_OVERRIDE).expanduser().resolve()
    if not matches_input(DATA_DIR):
        raise ValueError(f"Missing CSV or unexpected header in {DATA_DIR}")
elif IS_KAGGLE:
    candidates = sorted({p.parent.resolve() for p in Path("/kaggle/input").rglob("train.csv") if matches_input(p.parent)})
    if len(candidates) != 1:
        raise ValueError(f"Expected one matching dataset; found {candidates}. Set DATA_DIR_OVERRIDE explicitly.")
    DATA_DIR = candidates[0]
else:
    DATA_DIR = Path("/content/sincon-2026")
    if not matches_input(DATA_DIR):
        raise ValueError("Set DATA_DIR_OVERRIDE to the Colab folder containing all four original CSV files.")

OUT_DIR = PROJECT_ROOT / "outputs" / "lightgbm_tuned"
print("Data:", DATA_DIR)
print("Output:", OUT_DIR)
print("Input byte sizes:", {name: (DATA_DIR / name).stat().st_size for name in EXPECTED_HEADERS})
''')

    add("markdown", """
## 3. 의존성 설치

참조 실행의 버전을 고정한 `requirements.txt`를 설치합니다. Kaggle에서는 이 셀을 실행할 때
인터넷을 켜야 할 수 있습니다. 오프라인 wheel 폴더가 있으면 `WHEELHOUSE`를 지정하면 됩니다.
이미 같은 버전을 준비했다면 `INSTALL_REQUIREMENTS=False`로 바꿀 수 있습니다.
설치 후 각 작업은 별도 Python 프로세스로 실행하므로 이 노트북 커널의 기존 import 상태에 의존하지 않습니다.

대회 자체의 실행시간 제한과 플랫폼 세션 제한은 구별합니다. 제공된 대회 규정에는 별도 실행시간 제한이 없으며,
[Kaggle 일반 문서](https://www.kaggle.com/docs/notebooks)의 CPU/GPU 세션은 12시간, TPU는 9시간입니다.
실제 사용 가능 RAM·GPU 및 주간 GPU 잔여 시간은 계정의 현재 표시를 확인해야 합니다.
이 파이프라인의 기본 설정은 CPU 4개 스레드이며, 실행시간과 최대 RSS를 결과에 기록합니다.
""")
    add("code", '''INSTALL_REQUIREMENTS = True
WHEELHOUSE = ""  # Optional existing directory for offline pip wheels.
if INSTALL_REQUIREMENTS:
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(PROJECT_ROOT / "requirements.txt")]
    if WHEELHOUSE:
        wheel_dir = Path(WHEELHOUSE)
        if not wheel_dir.is_dir():
            raise ValueError(f"Wheel directory missing: {wheel_dir}")
        command += ["--no-index", "--find-links", str(wheel_dir)]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)

version_check = """
import importlib.metadata, json
from pathlib import Path
expected = dict(line.strip().split('==', 1) for line in Path('requirements.txt').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#'))
actual = {name: importlib.metadata.version(name) for name in expected}
print(json.dumps(actual, indent=2))
mismatched = {name: {'expected': version, 'actual': actual[name]} for name, version in expected.items() if actual[name].split('+', 1)[0] != version}
if mismatched:
    raise RuntimeError('Dependency versions differ from the reference: ' + json.dumps(mismatched))
"""
subprocess.run([sys.executable, "-c", version_check], check=True, cwd=PROJECT_ROOT)
''')

    add("markdown", """
## 4. 누수·지표·제출 검수 테스트

미래 타겟·하차량 변경이 특징에 영향을 주지 않는지, 123일 label window가 cutoff를 넘지 않는지,
RMSLE·계절 MASE·달력 ACF의 정의와 로그 공간 앙상블이 맞는지 검사합니다.
공식 전체 test 범위와 오래된 sample의 차이도 검사합니다.
""")
    add("code", '''RUN_TESTS = True
if RUN_TESTS:
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True, cwd=PROJECT_ROOT)
''')

    add("markdown", """
## 5. 전체 데이터 EDA

원본을 보존하면서 행·열, 결측·중복·시간 규칙성, 역별 관측 기간, 분포·급등 후보,
달력 패턴과 역 간 상관을 확인합니다. 전체 학습 자료의 EDA 상관을 모델 특징으로 재사용하지 않습니다.
""")
    add("code", '''subprocess.run([
    sys.executable, "-m", "src.eda", "--data-dir", str(DATA_DIR),
    "--out-dir", str(PROJECT_ROOT / "reports"),
], check=True, cwd=PROJECT_ROOT)
print("EDA report:", PROJECT_ROOT / "reports" / "EDA.md")
''')

    add("markdown", """
## 6. 고정 CV → 앙상블 → 최종 학습

개발 fold 2개에서 관측 이력이 있는 역을 대상으로 LightGBM 설정 16개를 비교합니다.
최대 1,000회, 100회 미개선 시 early stopping을 적용하며, 최소 개선량은 RMSLE 0.00001입니다.
두 개발 fold에서 공통으로 기록한 반복 범위의 행 수 가중 log-MSE를 최소화하는 설정과 반복 수를 선택합니다.
선택한 반복 수로 재학습한 뒤 마지막 fold를 재검증하고 전체 관측으로 최종 모델을 학습합니다.
세 번째 fold는 이미 이전 분석에 사용했으므로 새로운 독립 holdout 점수가 아닙니다.
AdamW는 사용하지 않습니다. 전역 최적점 보장이 아니라 탐색 범위의 최선입니다.

입력 특징은 각 예측 origin 이전 관측으로 고정하고, 123일 전체를 직접 예측합니다.
역사적 origin마다 공휴일의 당시 공개 여부도 반영합니다.
현재 모델은 LightGBM 단독입니다. 계절 특징은 입력 특징으로 유지합니다.
사전학습 가중치는 사용하지 않습니다. 전체 테스트를 기준으로 최종 제출 파일을 생성합니다.
""")
    add("code", '''config = json.loads((PROJECT_ROOT / "configs" / "default.json").read_text())
print(json.dumps(config, ensure_ascii=False, indent=2))
RUN_SEARCH = True  # False: refit the already selected configuration and rounds.
if RUN_SEARCH:
    command = [sys.executable, "-m", "src.tune_lightgbm", "--data-dir", str(DATA_DIR),
               "--config", str(PROJECT_ROOT / "configs" / "three_tree.json"),
               "--max-rounds", "1000", "--patience", "100", "--trials", "16",
               "--output", str(OUT_DIR)]
else:
    command = [sys.executable, "-m", "src.pipeline", "train", "--data-dir", str(DATA_DIR),
               "--config", str(PROJECT_ROOT / "configs" / "default.json"), "--output", str(OUT_DIR)]
subprocess.run(command, check=True, cwd=PROJECT_ROOT)
''')

    add("markdown", """
## 7. 저장 모델 재로딩·재추론 확인

같은 입력과 저장 모델에서 CSV가 완전히 동일하게 재생성되는지 SHA-256으로 비교합니다.
입력 파일의 해시, 학습 종료일, 특징 순서가 변경되면 추론을 중단합니다.
""")
    add("code", '''RECHECK_RELOADED_MODEL = True
if RECHECK_RELOADED_MODEL:
    reload_output = PROJECT_ROOT / "outputs" / "lightgbm_tuned_cli_reloaded"
    subprocess.run([
        sys.executable, "-m", "src.pipeline", "infer",
        "--data-dir", str(DATA_DIR), "--model", str(OUT_DIR / "model_bundle.joblib"),
        "--output", str(reload_output),
    ], check=True, cwd=PROJECT_ROOT)
    original_hash = hashlib.sha256((OUT_DIR / "submission.csv").read_bytes()).hexdigest()
    reloaded_hash = hashlib.sha256((reload_output / "submission.csv").read_bytes()).hexdigest()
    if original_hash != reloaded_hash:
        raise AssertionError("Reloaded model produced a different submission file.")
    print("Exact reload reproduction:", original_hash)
''')

    add("markdown", """
## 8. 선택 후처리 실험

현재 데이터에서는 후처리 후보가 마지막 재검증 점수를 악화시켜 기본 제출에 채택하지 않았습니다.
`RUN_POSTPROCESSING_EXPERIMENT=True`로 바꾸면 개발 1→2에서 후보를 선정한 뒤 마지막 구간에서 확인합니다.
실험 출력은 별도 폴더에 저장하며 아래 최종 제출은 기존 LightGBM 출력으로 유지합니다.
""")
    add("code", '''RUN_POSTPROCESSING_EXPERIMENT = False
if RUN_POSTPROCESSING_EXPERIMENT:
    post_out = PROJECT_ROOT / "outputs" / "lightgbm_postprocessed"
    subprocess.run([sys.executable, "-m", "src.evaluate_postprocessing", "--data-dir", str(DATA_DIR),
                    "--base-output", str(OUT_DIR), "--output", str(post_out)], check=True, cwd=PROJECT_ROOT)
    print(json.dumps(json.loads((post_out / "run_summary.json").read_text()), ensure_ascii=False, indent=2))
''')

    add("markdown", """
## 9. 최종 검수와 결과 확인

ID·순서·행/열·결측·무한대·음수를 검사한 전체 제출 파일만 복사합니다.
`template_diagnostic_DO_NOT_SUBMIT.csv`는 오래된 양식과의 비교용이므로 제출하지 않습니다.
Kaggle에서는 `/kaggle/working/submission.csv`, Colab에서는 프로젝트 폴더의 `submission.csv`가 최종 파일입니다.
""")
    add("code", '''subprocess.run([
    sys.executable, "-m", "src.validate_submission",
    "--submission", str(OUT_DIR / "submission.csv"),
    "--test", str(DATA_DIR / "test.csv"),
    "--sample", str(DATA_DIR / "sample_submission.csv"),
    "--scope", "test", "--report", str(OUT_DIR / "submission_validation_cli.json"),
], check=True, cwd=PROJECT_ROOT)
validation = json.loads((OUT_DIR / "submission_validation_cli.json").read_text())
if not validation["valid"] or not validation["matches_official_scope"] or validation["rows"] != 157194:
    raise AssertionError("Official full-test submission validation failed.")
FINAL_SUBMISSION = Path("/kaggle/working/submission.csv") if IS_KAGGLE else PROJECT_ROOT / "submission.csv"
shutil.copyfile(OUT_DIR / "submission.csv", FINAL_SUBMISSION)
print("Final submission:", FINAL_SUBMISSION)
print("Submission SHA-256:", hashlib.sha256(FINAL_SUBMISSION.read_bytes()).hexdigest())

import pandas as pd
from IPython.display import display, FileLink
display(pd.read_csv(OUT_DIR / "cv_scores.csv"))
print("Reused validation summary (weights fit on folds 1 and 2 only):")
print(json.dumps(json.loads((OUT_DIR / "audit_summary.json").read_text()), ensure_ascii=False, indent=2))
print("Run summary:")
print(json.dumps(json.loads((OUT_DIR / "run_summary.json").read_text()), ensure_ascii=False, indent=2))
display(FileLink(str(FINAL_SUBMISSION)))
display(FileLink(str(OUT_DIR / "cv_scores.csv")))
display(FileLink(str(PROJECT_ROOT / "reports" / "EDA.md")))
''')
    notebook = {
        "cells": blocks,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "sincon_reproduction": {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "source_snapshot_sha256": snapshot,
                "embedded_file_sha256": manifest,
                "competition_data_embedded": False,
                "automatically_submits": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, current in enumerate(blocks):
        if current["cell_type"] == "code":
            compile("".join(current["source"]), f"notebook_cell_{i}", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    saved = json.loads(output.read_text(encoding="utf-8"))
    if len({c["id"] for c in saved["cells"]}) != len(saved["cells"]):
        raise AssertionError("Duplicate notebook cell IDs.")
    return {
        "output": str(output.resolve()), "bytes": output.stat().st_size,
        "cells": len(blocks), "code_cells": sum(c["cell_type"] == "code" for c in blocks),
        "embedded_files": len(embedded), "source_snapshot_sha256": snapshot,
        "json_and_code_syntax_verified": True, "training_executed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.project_root / "notebooks" / "SINCON_reproduce.ipynb"
    print(json.dumps(build(args.project_root, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
