"""Section 4 of handoff_for_gemini: runs the CPU tail of the pipeline (v15 stack -> v19 -> v20e, the same calls as
reproduce/verify_from_stored.py without --retrain) under a Python audit hook and records every file it opens for reading.
Run from the project root:  V11_THREADS=12 .venv/bin/python outputs/handoff_for_gemini/scripts/trace_inputs.py"""
import sys, os, json, hashlib
sys.path.insert(0, '.')
from pathlib import Path
ROOT = Path('.').resolve()
seen = {}
def hook(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else 'r'
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if ('w' in mode or 'a' in mode or 'x' in mode) or (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        seen.setdefault(str(p), 0); seen[str(p)] += 1
sys.addaudithook(hook)
import reproduce.verify_from_stored as V
V.main()
rows = []
for p, n in sorted(seen.items()):
    P = Path(p)
    if not P.is_file() or p.endswith('.pyc') or '/site-packages/' in p or '/lib/python' in p or '/_scratch/' in p:
        continue
    rel = str(P.relative_to(ROOT)) if P.is_relative_to(ROOT) else p
    if rel.startswith('src/') and rel.endswith('.py'):
        continue
    rows.append(dict(path=rel, inside_project=P.is_relative_to(ROOT), bytes=P.stat().st_size, sha256=hashlib.sha256(P.read_bytes()).hexdigest(), opens=n))
Path('outputs/handoff_for_gemini/04_reproducibility/traced_inputs_cpu_tail.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
print(len(rows), 'input files')
