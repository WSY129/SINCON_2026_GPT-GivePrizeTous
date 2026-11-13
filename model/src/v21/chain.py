"""v21 chain on the stored v21 members: v15 stack (stage-1 weights, event-day rule, events / school / low / holiday steps)
-> v19 steps (concert, lunar grave visits, mart, night, growth).  Same code as the delivered v15 / v19; only the member
files, the output folders and the calendar label fix differ.  No leaderboard value is used.

    V21_KINDS=none|stack|all V21_MEMBERS=lgb_h,xgb_i,... .venv/bin/python -m src.v21.chain
  none   the label fix only inside the frozen tree panels (trees were trained on it); stack and steps use the old labels
  stack  the stack (v15 run) also uses the fixed labels; the v19 steps use the old labels (3122 grave-visit days unchanged)
  all    every step uses the fixed labels"""
import json, os, shutil
VARIANT = os.environ.get('V21_KINDS', 'all')
os.environ['V15_MEMBERS'] = os.environ.get('V21_MEMBERS', 'lgb_hk,xgb_ik,xgb_g11k')
from src.v8 import core as c8
from src.v21 import kinds

MODELS = c8.ROOT / 'outputs/ensemble_v21/models'


def main():
    out = c8.ROOT / f"outputs/ensemble_v21/chain_{VARIANT}_tol{os.environ.get('V21_STEP_TOL', '1.5e-4')}_{os.environ['V15_MEMBERS'].replace(',', '+')}"
    if VARIANT in ('stack', 'all'):
        kinds.apply()
    from src.v15 import run as v15
    v15.TOL = float(os.environ.get('V21_STEP_TOL', '1.5e-4'))                  # 1.5e-4 keeps the v20e step set (events, school, low, holiday); fold moves of this size are seed noise
    v15.DIRS = [MODELS]; v15.OUT = out / 'v15'; v15.OUT.mkdir(parents=True, exist_ok=True)
    v15.main()
    if VARIANT == 'stack':
        c8.day_kinds = c8._day_kinds_orig
    from src.v19 import run as v19
    v19.OUT = out / 'v19'; (v19.OUT / 'external').mkdir(parents=True, exist_ok=True)
    shutil.copy(c8.ROOT / 'outputs/ensemble_v19/external/mart_closure_calendar.csv', v19.OUT / 'external')
    v19.BASE = out / 'v15'; v19.main()
    g = json.loads((v19.OUT / 'gates_v19.json').read_text())
    print('RESULT', VARIANT, os.environ['V15_MEMBERS'], json.dumps(g.get('generations', {}).get('v19', g.get('final', {})), default=float)[:600])


if __name__ == '__main__':
    main()
