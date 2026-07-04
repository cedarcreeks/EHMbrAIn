"""EHMbrAIn dashboard: fleet risk, engine detail, and the head-to-head verdicts.

Run:  uv run streamlit run dashboard/app.py
Reads only versioned/regenerable artifacts under data/processed/.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / 'data' / 'processed' / 'fleet'
F5 = ROOT / 'data' / 'processed' / 'f5'

st.set_page_config(page_title='EHMbrAIn', layout='wide')
st.title('EHMbrAIn — AI vs traditional EHM on a synthetic CFM56-7B26 fleet')


@st.cache_data
def load_index():
    return json.loads((FLEET / 'fleet_index.json').read_text())['engines']


@st.cache_data
def load_engine(eid: int):
    return pd.read_parquet(FLEET / 'snapshots.parquet',
                           filters=[('engine_id', '==', eid)]).sort_values('cycle')


@st.cache_data
def load_events():
    return pd.read_parquet(FLEET / 'events.parquet')


tab_fleet, tab_engine, tab_verdicts = st.tabs(
    ['Fleet risk', 'Engine detail', 'F5 verdicts'])

with tab_fleet:
    idx = pd.DataFrame(load_index())
    idx['risk_rank'] = idx['life_cycles'].rank().astype(int)
    st.caption('Shorter life = harder degrader. Split shows evaluation role.')
    st.dataframe(idx[['engine_id', 'split', 'life_cycles', 'egtm_new_C',
                      'drift_channel']].sort_values('life_cycles'),
                 use_container_width=True, height=420)

with tab_engine:
    ids = [e['engine_id'] for e in load_index()]
    eid = st.selectbox('Engine', ids)
    e = load_engine(int(eid))
    ev = load_events()
    ev = ev[ev.engine_id == int(eid)]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('EGT margin [°C]')
        st.line_chart(e.set_index('cycle')['egtm_C'])
    with c2:
        st.subheader('True health deviations [%]')
        st.line_chart(e.set_index('cycle')[
            ['x_hpc_eta', 'x_hpt_eta', 'x_hpt_flow', 'x_fan_eta']])
    st.subheader('Event log')
    st.dataframe(ev, use_container_width=True)

with tab_verdicts:
    vpath = F5 / 'verdicts.json'
    if vpath.exists():
        v = json.loads(vpath.read_text())
        for h, d in v.items():
            badge = '✅ CONFIRMED' if d.get('confirmed') else '❌ refuted'
            st.markdown(f'**{h}** — {badge}')
            st.json({k: x for k, x in d.items() if k != 'confirmed'},
                    expanded=False)
    else:
        st.info('Run scripts/f5_confirm.py first.')
