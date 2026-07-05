"""F-VIZ: per-engine confirmatory RUL error distributions (tuned F5 configs,
test fleet), traditional vs AI, at 50/70/90 % life. Dumps signed errors to
data/processed/f5/rul_errors.json so the report figure regenerates cheaply.

The distribution (not just the RMSE) is the honest evidence: it shows the
traditional pipeline's optimism bias and wide spread against the AI's tight,
near-unbiased errors. Foreground (MPS). Usage: uv run python scripts/fig_rul_distribution.py
"""

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import eval_trad, eval_ai, fleet_cache        # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
F5 = REPO_ROOT / 'data' / 'processed' / 'f5'


def main():
    sel_t = json.loads((F5 / 'selected_trad.json').read_text())['selected']
    sel_a = json.loads((F5 / 'selected_ai.json').read_text())['selected']
    fleet_cache()
    mt = eval_trad(sel_t['rul']['params'], 'test')
    ma = eval_ai(sel_a['rul']['params'], 'train', 'test', seed=0)
    out = {'traditional': {}, 'ai': {}}
    for fam, m in (('traditional', mt), ('ai', ma)):
        for f in (0.5, 0.7, 0.9):
            out[fam][str(f)] = [r['err'] for r in m['rul_rows']
                                if abs(r['frac'] - f) < 1e-6]
    (F5 / 'rul_errors.json').write_text(json.dumps(out))
    print('rul_errors.json written:',
          {f: len(out['ai'][f]) for f in out['ai']})


if __name__ == '__main__':
    main()
