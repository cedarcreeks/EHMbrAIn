"""F-VIZ helper: dump per-trial Optuna metrics from the F5 tuning studies to
data/processed/f5/optuna_history.json (for the convergence figure).
Usage: uv run python scripts/dump_optuna_history.py
"""
import json
from pathlib import Path

import optuna

F5 = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'f5'


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f'sqlite:///{F5}/optuna.db'
    out = {}
    for fam in ('trad', 'ai'):
        st = optuna.load_study(study_name=f'{fam}-v1', storage=storage)
        trials = [t for t in st.trials if t.state.is_finished()]
        out[fam] = {'n': len(trials),
                    'rul_rmse': [t.user_attrs.get('rul_rmse') for t in trials],
                    'conf_acc': [t.user_attrs.get('conf_acc') for t in trials],
                    'det_score': [t.user_attrs.get('det_score') for t in trials]}
    (F5 / 'optuna_history.json').write_text(json.dumps(out))
    print('optuna_history.json:', {f: out[f]['n'] for f in out})


if __name__ == '__main__':
    main()
