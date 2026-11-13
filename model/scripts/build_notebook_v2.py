"""Embed the current v2 code and tests into a standalone Kaggle/Colab notebook."""
import hashlib
import json
from pathlib import Path

def build(root):
    root=Path(root);paths=sorted((root/'src').glob('*.py'))+sorted((root/'tests').glob('test_*.py'))+[root/'configs/ensemble_v2.json',root/'requirements.txt',root/'reports/holiday_sources.md']
    payload={p.relative_to(root).as_posix():p.read_text() for p in paths if p.exists()}
    hashes={name:hashlib.sha256(value.encode()).hexdigest() for name,value in payload.items()};cells=[]
    def add(kind,text):
        cell={'cell_type':kind,'metadata':{},'id':f'v2-{len(cells):03d}','source':text.strip().splitlines(keepends=True)}
        if kind=='code':compile(text,'notebook','exec');cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','''# SINCON ensemble_v2 재현

4-fold expanding-origin CV → 로그 OOF 볼록 앙상블 → 2개 시드 전체 재학습 → 제출 파일.
각 fold는 80만 행, 최종 학습은 100만 행이며 기본 설정을 축소하지 않습니다.
LightGBM은 CPU, CatBoost는 CUDA 0, XGBoost는 CUDA 1(한 장이면 0), TabM은 CUDA 0/MPS/CPU를 자동 선택합니다.
TabM은 CUDA에서만 FP16 + GradScaler를 사용합니다. CUDA CatBoost의 비트 단위 결정성은 보장되지 않습니다.
Kaggle에서는 4개 대회 CSV를 Add Input하고 GPU T4 × 2를 선택한 뒤 Run All 하세요.
Colab에서는 CSV 폴더를 업로드하거나 마운트하고 아래 DATA_DIR_OVERRIDE를 설정하세요.
실행시간은 장치와 조기 종료 시점에 따라 달라집니다. 현재 런타임의 세션 제한은 별도로 확인하세요.
외부 사전학습 가중치는 사용하지 않으며, 달력 공개시점 필터 as_of=origin을 유지합니다.
노트북은 제출 파일을 만들며 Kaggle에 자동 제출하지 않습니다.
''')
    add('code',f'''from pathlib import Path
import hashlib, json, os, subprocess, sys
IS_KAGGLE = Path('/kaggle/working').exists()
PROJECT_ROOT = Path('/kaggle/working/SINCON' if IS_KAGGLE else '/content/SINCON')
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
EMBEDDED_FILES = {payload!r}
SOURCE_HASHES = {hashes!r}
for name, content in EMBEDDED_FILES.items():
    assert hashlib.sha256(content.encode()).hexdigest() == SOURCE_HASHES[name]
    path = PROJECT_ROOT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
print('Restored source files:', len(EMBEDDED_FILES))
''')
    add('code','''DATA_DIR_OVERRIDE = ''  # Colab: /content/sincon-2026
required = ['train.csv', 'test.csv', 'metaData.csv', 'sample_submission.csv']
if DATA_DIR_OVERRIDE:
    DATA_DIR = Path(DATA_DIR_OVERRIDE)
elif IS_KAGGLE:
    candidates = sorted({p.parent for p in Path('/kaggle/input').rglob('train.csv') if all((p.parent/n).exists() for n in required)})
    assert len(candidates) == 1, f'Set DATA_DIR_OVERRIDE; candidates={candidates}'
    DATA_DIR = candidates[0]
else:
    DATA_DIR = Path('/content/sincon-2026')
assert all((DATA_DIR/n).exists() for n in required), 'Prepare all four original CSV files'
OUT = PROJECT_ROOT/'outputs/ensemble_v2'
print('Data:', DATA_DIR, 'Output:', OUT)
''')
    add('markdown','''## 의존성과 테스트
참조 패키지 버전은 requirements.txt에 고정되어 있습니다. CUDA 런타임에 이미 설치된 PyTorch는
유지하며, 나머지 패키지 버전을 맞춥니다. 오프라인 환경에서는 해당 wheel을 미리 준비하세요.
환경 버전과 장치는 실행 결과 environment.json에 기록됩니다.
''')
    add('code','''INSTALL_REQUIREMENTS = True
if INSTALL_REQUIREMENTS:
    import importlib.util
    packages = [line for line in (PROJECT_ROOT/'requirements.txt').read_text().splitlines() if line and not line.startswith('torch==')]
    if importlib.util.find_spec('torch') is None:
        packages.append('torch==2.15.0')
    subprocess.run([sys.executable, '-m', 'pip', 'install', *packages], check=True)
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=PROJECT_ROOT, check=True)
''')
    add('markdown','''## 전체 실행
동일 코드·설정·입력 해시로 재실행하면 완료한 fold/model을 재사용합니다.
OOF 날짜가 폴드 사이에 중복되므로 LOFO 점수는 완전히 독립적인 holdout 점수가 아닙니다.
''')
    add('code','''subprocess.run([sys.executable, '-u', '-m', 'src.pipeline', 'v2', 'run', '--data-dir', str(DATA_DIR)], cwd=PROJECT_ROOT, check=True)
print((OUT/'ensemble_report.json').read_text())
print((OUT/'submission_validation.json').read_text())
''')
    add('code','''before = (OUT/'submission_ensemble_v2.csv').read_bytes()
subprocess.run([sys.executable, '-m', 'src.pipeline', 'v2', 'infer', '--data-dir', str(DATA_DIR)], cwd=PROJECT_ROOT, check=True)
import pandas as pd, numpy as np, io
old, new = pd.read_csv(io.BytesIO(before)), pd.read_csv(OUT/'submission_ensemble_v2.csv')
assert old.example_id.equals(new.example_id)
assert np.allclose(old.boardings, new.boardings, rtol=2e-5, atol=1e-5)
print('Reloaded prediction agreement verified. CPU/GPU floating-point differences are tolerated.')
from IPython.display import FileLink, display
display(FileLink(str(OUT/'submission_ensemble_v2.csv')))
''')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'sincon_v2':{'source_sha256':hashes,'auto_submit':False,'data_embedded':False}},'cells':cells}
    output=root/'notebooks/SINCON_reproduce.ipynb';output.parent.mkdir(exist_ok=True);output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    print(output,len(cells),'cells; compiled code cells; no hosted execution claimed')

if __name__=='__main__':build(Path(__file__).resolve().parents[1])
