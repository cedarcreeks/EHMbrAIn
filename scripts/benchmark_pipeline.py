"""Norm N5: measure wall-clock time of every computational stage on the
reference machine (Apple M5) and record it with the machine fingerprint.

Each stage runs as a fresh subprocess (cold start included — that is what a
replicator pays), cwd'd to a scratch directory so solver artifacts stay out
of the repo. Results MERGE into data/processed/compute_times.json, so stages
can be benchmarked in groups:

    uv run python scripts/benchmark_pipeline.py fleet audits trad
    uv run python scripts/benchmark_pipeline.py ai        # foreground: MPS
"""

import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'compute_times.json'

STAGES = {
    # group -> list of (stage name, script, args, uses_gpu)
    'model': [
        ('design_point', 'run_design_point.py', [], False),
        ('sls_anchors', 'run_anchors.py', [], False),
        ('icm_grid', 'make_icm.py', [], False),
        ('corrected_baseline', 'make_corrected_baseline.py', [], False),
    ],
    'decks': [('baseline_decks', 'make_decks.py', [], False)],
    'fleet': [('fleet_generation', 'make_fleet.py', [], False)],
    'audits': [
        ('audit_dataset', 'audit_dataset.py', [], False),
        ('audit_nonlinearity', 'audit_nonlinearity.py', ['40'], False),
    ],
    'trad': [('traditional_pipeline', 'run_trad.py', [], False)],
    'ai': [
        ('ai_suite', 'run_ai.py', [], True),
        ('pcs', 'run_pcs.py', [], True),
    ],
    'hybrid': [('hybrid_ablation', 'run_hybrid.py', [], True)],
    'f8': [
        ('surrogate_data_cruise', 'f8_surrogate_data.py', ['2400'], False),
        ('surrogate_train', 'f8_surrogate.py', [], True),
        ('fleet_v2', 'make_fleet.py', ['surrogate'], False),
        ('h4_v2_hybrid', 'f8_l6_hybrid.py', [], True),
        ('l4_recoverable', 'f8_l4_recoverable.py', [], False),
        ('l5_architectures', 'f8_l5_arch.py', [], True),
        ('l7_drift', 'f8_l7_drift.py', [], False),
        ('l9_pcs', 'f8_l9_pcs.py', [], True),
    ],
    'f10': [('identifiability_certificate', 'f10_certificate.py', [], False)],
}


def machine_fingerprint():
    def sysctl(key):
        try:
            return subprocess.check_output(['sysctl', '-n', key], text=True).strip()
        except Exception:
            return None
    fp = {'platform': platform.platform(),
          'python': platform.python_version()}
    if platform.system() == 'Darwin':
        fp['chip'] = sysctl('machdep.cpu.brand_string')
        fp['cores'] = sysctl('hw.ncpu')
        fp['mem_gb'] = round(int(sysctl('hw.memsize') or 0) / 2**30)
    try:
        import torch
        fp['torch'] = torch.__version__
        fp['gpu_backend'] = 'mps' if torch.backends.mps.is_available() else 'none'
    except ImportError:
        pass
    return fp


def main():
    groups = sys.argv[1:] or list(STAGES)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    stages = dict(existing.get('stages', {}))

    with tempfile.TemporaryDirectory() as scratch:
        for g in groups:
            for name, script, args, gpu in STAGES[g]:
                cmd = ['uv', 'run', '--project', str(REPO_ROOT), 'python',
                       str(REPO_ROOT / 'scripts' / script), *args]
                print(f'== {name} ({script}) ==', flush=True)
                t0 = time.perf_counter()
                r = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True)
                dt = time.perf_counter() - t0
                ok = r.returncode == 0
                stages[name] = {'script': script, 'wall_s': round(dt, 1),
                                'gpu': gpu, 'ok': ok,
                                'measured': time.strftime('%Y-%m-%d')}
                print(f'   {dt:7.1f} s  {"OK" if ok else "FAILED"}', flush=True)
                if not ok:
                    print(r.stderr[-800:], flush=True)

    OUT.write_text(json.dumps({'machine': machine_fingerprint(),
                               'stages': stages}, indent=2))
    print(json.dumps(stages, indent=1))


if __name__ == '__main__':
    main()
