"""WP2.3: generate the SynCFM56 synthetic fleet.

One worker process per engine (norm N1), each with its own deterministic RNG
(SeedSequence(seed, engine_id) — reproducible regardless of scheduling).
Outputs under data/processed/fleet/:
    snapshots.parquet   one row per flight cycle (takeoff + cruise channels,
                        measured & true, ground truth x, EGTM, RUL, labels)
    events.parquet      wash / FOD events per engine
    fleet_index.json    per-engine life, split, EGTM_new, severity multipliers

Usage: uv run python scripts/make_fleet.py [n_engines]
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ehmbrain.datagen.fleet import assign_splits, generate_engine, load_icm
from ehmbrain.datagen.snapshots import engine_snapshots

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / 'data' / 'processed' / 'fleet'


def build_engine(args):
    """Worker: one engine end-to-end (trajectory + snapshots)."""
    engine_id, catalog = args
    seed = catalog['fleet']['seed']
    rng_cfg = np.random.default_rng(np.random.SeedSequence([seed, engine_id, 0]))
    rng_meas = np.random.default_rng(np.random.SeedSequence([seed, engine_id, 1]))

    H, ch, base = load_icm('takeoff_hot')
    engine = generate_engine(engine_id, catalog, H, ch, base, rng_cfg)
    df = engine_snapshots(engine, engine['contributions'], catalog, rng_meas)

    cfg = engine['config']
    index_row = {'engine_id': engine_id, 'life_cycles': engine['life_cycles'],
                 'censored': engine['censored'], 'egtm_new_C': cfg.egtm_new_C,
                 'multipliers': cfg.multipliers,
                 'drift_channel': next(iter(cfg.drifts), None)}
    events = [{'engine_id': engine_id, **e} for e in engine['events']]
    return df, events, index_row


def main():
    n_engines = int(sys.argv[1]) if len(sys.argv) > 1 else None
    catalog = yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())
    if n_engines:
        catalog['fleet']['n_engines'] = n_engines
    n = catalog['fleet']['n_engines']

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dfs, all_events, index = [], [], []
    with ProcessPoolExecutor() as pool:
        for df, events, row in pool.map(build_engine,
                                        [(i, catalog) for i in range(n)]):
            dfs.append(df)
            all_events.extend(events)
            index.append(row)
            if len(dfs) % 10 == 0:
                print(f'{len(dfs)}/{n} engines  t {time.time()-t0:5.0f}s', flush=True)

    rng_split = np.random.default_rng(np.random.SeedSequence(
        [catalog['fleet']['seed'], 999]))
    splits = assign_splits([r['engine_id'] for r in index],
                           catalog['fleet']['split'], rng_split)
    for r in index:
        r['split'] = splits[r['engine_id']]

    snap = pd.concat(dfs, ignore_index=True)
    snap['split'] = snap['engine_id'].map(splits)
    snap.to_parquet(OUT_DIR / 'snapshots.parquet', index=False)
    pd.DataFrame(all_events).to_parquet(OUT_DIR / 'events.parquet', index=False)
    (OUT_DIR / 'fleet_index.json').write_text(json.dumps(
        {'generated_s': time.time() - t0, 'n_engines': n, 'engines': index},
        indent=2, default=float))

    lives = [r['life_cycles'] for r in index]
    print(f'{n} engines, {len(snap):,} snapshot rows, '
          f'lives median {np.median(lives):.0f} [{min(lives)}-{max(lives)}], '
          f'{time.time()-t0:.0f}s -> {OUT_DIR}')


if __name__ == '__main__':
    main()
