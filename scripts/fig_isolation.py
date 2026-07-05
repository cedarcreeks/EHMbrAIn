"""F-VIZ: per-episode isolation predictions (tuned F5 configs, test fleet),
traditional and AI, for the isolation confusion matrices. Dumps to
data/processed/f5/isolation.json so the report figure regenerates cheaply.

The confusion matrix is the mechanism behind the H2 refutation: on the
confusable pairs both families scatter the fault onto the wrong partner.

Foreground (MPS). Usage: uv run python scripts/fig_isolation.py
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import eval_trad, eval_ai, fleet_cache        # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
F5 = REPO_ROOT / 'data' / 'processed' / 'f5'


def main():
    sel_t = json.loads((F5 / 'selected_trad.json').read_text())['selected']
    sel_a = json.loads((F5 / 'selected_ai.json').read_text())['selected']
    fleet_cache()
    mt = eval_trad(sel_t['isolation']['params'], 'test')
    ma = eval_ai(sel_a['isolation']['params'], 'train', 'test', seed=0)
    out = {
        'traditional': [{'true': r['param'], 'pred': r['pred']} for r in mt['iso_rows']],
        'ai': [{'true': r['param'], 'pred': r['pred']} for r in ma['iso_rows']],
    }
    (F5 / 'isolation.json').write_text(json.dumps(out))
    print('isolation.json:', {'trad': len(out['traditional']), 'ai': len(out['ai'])})


if __name__ == '__main__':
    main()
