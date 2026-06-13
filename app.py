import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import datetime
from datetime import date, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Manual Target Encoder ──────────────────────────────────────────────────────
class ManualTargetEncoder:
    def __init__(self, cols, smoothing=10):
        self.cols = cols; self.smoothing = smoothing
        self.stats_ = {}; self.globals_ = {}
    def fit(self, X, y):
        y_s = pd.Series(y, index=X.index)
        for col in self.cols:
            agg = pd.DataFrame({'y': y_s, 'col': X[col].values}).groupby('col')['y'].agg(['mean','count'])
            gm  = y_s.mean()
            agg['s'] = (agg['count'] * agg['mean'] + self.smoothing * gm) / (agg['count'] + self.smoothing)
            self.stats_[col] = agg['s'].to_dict(); self.globals_[col] = gm
        return self
    def transform(self, X):
        Xo = X.copy()
        for col in self.cols:
            Xo[col] = Xo[col].map(self.stats_[col]).fillna(self.globals_[col])
        return Xo

# ── Manual Preprocessor ────────────────────────────────────────────────────────
class ManualPreprocessor:
    def __init__(self):
        self.numeric_log_cols = self.numeric_scale_cols = self.binary_cols = None
        self.ohe_cols = self.target_enc_cols = None
        self.medians_ = {}; self.scale_means_ = {}; self.scale_stds_ = {}; self.ohe_categories_ = {}
    def fit(self, X, numeric_log, numeric_scale, binary, ohe, target_enc):
        self.numeric_log_cols = numeric_log; self.numeric_scale_cols = numeric_scale
        self.binary_cols = binary; self.ohe_cols = ohe; self.target_enc_cols = target_enc
        for col in numeric_log:
            med = float(np.nanmedian(X[col])); self.medians_[col] = med
            vals = np.log1p(np.where(np.isnan(X[col].astype(float)), med, X[col].astype(float)))
            self.scale_means_[col] = float(vals.mean()); self.scale_stds_[col] = float(vals.std())
        for col in numeric_scale:
            med = float(np.nanmedian(X[col])); self.medians_[col] = med
            vals = np.where(np.isnan(X[col].astype(float)), med, X[col].astype(float))
            self.scale_means_[col] = float(vals.mean()); self.scale_stds_[col] = float(vals.std())
        for col in ohe:
            self.ohe_categories_[col] = sorted(X[col].dropna().unique().tolist())
        return self
    def transform(self, X):
        parts = []
        for col in self.numeric_log_cols:
            med = self.medians_[col]
            vals = np.log1p(np.where(pd.isna(X[col]), med, X[col].astype(float)))
            vals = (vals - self.scale_means_[col]) / (self.scale_stds_[col] + 1e-8)
            parts.append(vals.reshape(-1,1))
        for col in self.numeric_scale_cols:
            med = self.medians_[col]
            vals = np.where(pd.isna(X[col]), med, X[col].astype(float))
            vals = (vals - self.scale_means_[col]) / (self.scale_stds_[col] + 1e-8)
            parts.append(vals.reshape(-1,1))
        for col in self.binary_cols:
            parts.append(X[col].fillna(0).astype(float).values.reshape(-1,1))
        for col in self.ohe_cols:
            for cat in self.ohe_categories_[col]:
                parts.append((X[col]==cat).astype(float).values.reshape(-1,1))
        for col in self.target_enc_cols:
            parts.append(X[col].astype(float).values.reshape(-1,1))
        return np.hstack(parts)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Match Priority Predictor", page_icon="⚽", layout="wide")

# ── Theme (clean & modern) ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:#fafbfc; --bg2:#ffffff; --bg3:#f1f4f7; --border:#e7ebef; --border2:#d6dde4;
    --text:#10151f; --text2:#5b6573; --text3:#9aa3b0;
    --accent:#1b6e4c; --accent-soft:#e6f3ec;
    --high:#d64545; --med:#2e6fe0; --low:#1b6e4c;
    --shadow:0 12px 32px rgba(16,21,31,0.06); --shadow2:0 1px 3px rgba(16,21,31,0.04);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

.main .block-container { max-width: 880px; padding-top: 0 !important; }
#MainMenu, footer, header { visibility:hidden; }

/* Header */
.app-header { display:flex; align-items:baseline; gap:0.6rem; padding:2.2rem 0 0.3rem 0; }
.app-title { font-family:'Sora',sans-serif; font-size:1.7rem; font-weight:800; color:var(--text); letter-spacing:-0.5px; }
.app-title span { color:var(--accent); }
.app-sub { font-size:0.85rem; color:var(--text3); font-weight:500; }
.app-divider { height:1px; background:var(--border); margin:1.2rem 0 1.8rem 0; border:none; }

/* Tab navigation */
.stButton > button {
    background:var(--bg2) !important; color:var(--text2) !important;
    font-family:'Inter',sans-serif !important; font-size:0.92rem !important;
    font-weight:600 !important; letter-spacing:0 !important; text-transform:none !important;
    border:1px solid var(--border) !important; border-radius:10px !important; padding:0.7rem 1.6rem !important;
    width:100% !important; box-shadow:none !important;
    transition:all 0.15s !important;
}
.stButton > button:hover { border-color:var(--accent) !important; color:var(--accent) !important; transform:none !important; }
button[kind="primary"] {
    background:var(--text) !important; color:#fff !important; border-color:var(--text) !important;
}
button[kind="primary"]:hover { background:var(--accent) !important; border-color:var(--accent) !important; color:#fff !important; }

/* Section group */
.group-title { font-family:'Sora',sans-serif; font-size:0.95rem; font-weight:700; color:var(--text); margin:2rem 0 0.9rem 0; }
.group-title:first-of-type { margin-top:0; }

.card { background:var(--bg2); border:1px solid var(--border); border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:0.6rem; box-shadow:var(--shadow2); }
.card-label { font-size:0.7rem; font-weight:600; letter-spacing:0.5px; text-transform:uppercase; color:var(--text3); margin-bottom:0.35rem; }
.card-value { font-size:0.92rem; font-weight:600; color:var(--text); line-height:1.4; }

.info-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.6rem; margin-bottom:0.8rem; }

/* History cards */
.hist-card { background:var(--bg2); border:1px solid var(--border); border-radius:14px; padding:1.1rem 1.3rem; box-shadow:var(--shadow2); }
.hist-team { font-family:'Sora',sans-serif; font-size:0.95rem; font-weight:700; color:var(--text); margin-bottom:0.7rem; }
.hist-row { display:flex; justify-content:space-between; align-items:center; padding:0.3rem 0; border-bottom:1px solid var(--border); }
.hist-row:last-of-type { border-bottom:none; }
.hist-key { font-size:0.8rem; color:var(--text2); }
.hist-val { font-size:0.9rem; font-weight:700; color:var(--text); }
.hist-badge { display:inline-block; font-size:0.68rem; font-weight:600; padding:0.2rem 0.6rem; border-radius:6px; margin-top:0.6rem; }
.badge-ok { background:var(--accent-soft); color:var(--accent); }
.badge-warn { background:#fdf3e3; color:#b87213; }

/* Result hero — ticket style */
.result-wrap {
    border-radius:18px; padding:2.4rem 2rem; text-align:center;
    background:var(--bg2); border:1px solid var(--border); box-shadow:var(--shadow);
    margin-bottom:1.5rem; position:relative;
}
.result-wrap::before {
    content:''; position:absolute; top:0; left:0; right:0; height:6px; border-radius:18px 18px 0 0;
}
.result-HIGH::before { background:var(--high); }
.result-MEDIUM::before { background:var(--med); }
.result-LOW::before { background:var(--low); }
.result-eyebrow { font-size:0.72rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:var(--text3); margin-bottom:0.6rem; }
.result-label { font-family:'Sora',sans-serif; font-size:3.4rem; font-weight:800; line-height:1; letter-spacing:-1px; margin-bottom:0.7rem; }
.color-HIGH { color:var(--high); } .color-MEDIUM { color:var(--med); } .color-LOW { color:var(--low); }
.confidence-pill { display:inline-flex; align-items:center; gap:0.4rem; background:var(--bg3); border-radius:20px; padding:0.35rem 0.9rem; font-size:0.85rem; color:var(--text2); font-weight:500; }

/* Probability bars */
.prob-item { margin-bottom:0.9rem; }
.prob-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; }
.prob-name { font-size:0.85rem; font-weight:600; }
.prob-pct { font-size:0.85rem; font-weight:700; }
.prob-track { height:8px; background:var(--bg3); border-radius:4px; overflow:hidden; }
.prob-fill { height:100%; border-radius:4px; }

/* Summary chips */
.chips { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.6rem; }
.chip { background:var(--bg3); border-radius:8px; padding:0.35rem 0.8rem; font-size:0.8rem; color:var(--text2); font-weight:500; }

/* Bulk section */
.bulk-info { background:var(--bg2); border:1px solid var(--border); border-radius:14px; padding:1.3rem 1.5rem; margin-bottom:1.2rem; box-shadow:var(--shadow2); }
.bulk-info-title { font-family:'Sora',sans-serif; font-size:0.95rem; font-weight:700; color:var(--text); margin-bottom:0.5rem; }
.bulk-info-text { font-size:0.85rem; color:var(--text2); line-height:1.7; }
.required-col { display:inline-block; background:var(--bg3); border-radius:6px; padding:0.2rem 0.55rem; font-size:0.75rem; font-family:monospace; color:var(--text2); margin:0.15rem; }

/* Result table */
.result-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.result-table th { background:var(--bg3); color:var(--text2); font-weight:600; font-size:0.72rem; letter-spacing:0.5px; text-transform:uppercase; padding:0.7rem 0.9rem; text-align:left; }
.result-table td { padding:0.65rem 0.9rem; border-bottom:1px solid var(--border); color:var(--text); vertical-align:middle; }
.result-table tr:last-child td { border-bottom:none; }
.result-table tr:hover td { background:var(--bg3); }
.badge { display:inline-block; font-size:0.72rem; font-weight:700; padding:0.25rem 0.65rem; border-radius:6px; }
.badge-HIGH { background:#fbeaea; color:var(--high); }
.badge-MEDIUM { background:#e9f0fd; color:var(--med); }
.badge-LOW { background:var(--accent-soft); color:var(--low); }

/* Stat row */
.stat-row { display:flex; gap:0.8rem; margin-bottom:1.2rem; }
.stat-box { flex:1; background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:1rem; text-align:center; box-shadow:var(--shadow2); }
.stat-num { font-family:'Sora',sans-serif; font-size:1.7rem; font-weight:800; }
.stat-lbl { font-size:0.72rem; color:var(--text3); margin-top:0.25rem; }

/* Input widgets — clean light theme */
div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stTimeInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stFileUploader"] label {
    color:var(--text2) !important; font-size:0.78rem !important;
    font-weight:600 !important; letter-spacing:0.2px !important;
    text-transform:none !important; font-family:'Inter',sans-serif !important;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stDateInput"] input,
div[data-testid="stTimeInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] > div > div,
div[data-testid="stTimeInput"] > div > div,
div[data-testid="stNumberInput"] > div,
div[data-testid="stFileUploader"] section {
    background:var(--bg2) !important;
    border-color:var(--border) !important;
    color:var(--text) !important;
    border-radius:10px !important;
    font-size:0.95rem !important;
}
div[data-testid="stSelectbox"] > div > div > div { color:var(--text) !important; font-size:0.95rem !important; }
div[data-testid="stNumberInput"] button { background:var(--bg3) !important; color:var(--text) !important; border-color:var(--border) !important; }
div[data-testid="stFileUploader"] > div { background:var(--bg2) !important; border-color:var(--border) !important; border-radius:12px !important; }
div[data-testid="stFileUploader"] section span,
div[data-testid="stFileUploader"] section small { color:var(--text2) !important; }
div[data-testid="stCheckbox"] { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:0.65rem 0.9rem; }
div[data-testid="stCheckbox"] label p { color:var(--text) !important; font-size:0.88rem !important; text-transform:none !important; letter-spacing:normal !important; font-weight:500 !important; }

/* Primary action buttons (Predict / Run) get accent treatment via id targeting */
.primary-action .stButton > button {
    background:var(--accent) !important; color:#fff !important; border:none !important;
    font-family:'Sora',sans-serif !important; font-weight:700 !important; font-size:0.95rem !important;
    padding:0.85rem 2rem !important; box-shadow:0 6px 18px rgba(27,110,76,0.22) !important;
}
.primary-action .stButton > button:hover { background:#155c3f !important; color:#fff !important; }

/* Back link button */
.back-btn .stButton > button {
    background:transparent !important; color:var(--text2) !important;
    border:none !important; box-shadow:none !important;
    width:auto !important; padding:0.4rem 0 !important; font-size:0.88rem !important;
}
.back-btn .stButton > button:hover { color:var(--accent) !important; }

/* Download / upload browse buttons */
.stDownloadButton > button,
div[data-testid="stFileUploader"] section button {
    background:var(--bg2) !important; color:var(--text) !important;
    border:1px solid var(--border2) !important; box-shadow:none !important;
    font-family:'Inter',sans-serif !important; font-weight:600 !important;
    letter-spacing:0 !important; text-transform:none !important;
    width:auto !important;
}
.stDownloadButton > button:hover,
div[data-testid="stFileUploader"] section button:hover { border-color:var(--accent) !important; color:var(--accent) !important; }

.stMarkdown p, .stMarkdown li { font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Load ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model     = joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl'))
    prep      = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocessor.pkl'))
    tenc      = joblib.load(os.path.join(BASE_DIR, 'models', 'target_encoder.pkl'))
    threshold = joblib.load(os.path.join(BASE_DIR, 'models', 'threshold.pkl'))
    with open(os.path.join(BASE_DIR, 'models', 'label_encoder.json'), 'r') as f:
        le_classes = json.load(f)['classes']
    return model, prep, tenc, threshold, le_classes

@st.cache_data
def load_reference():
    xl = pd.read_excel(os.path.join(BASE_DIR, 'data', 'reference_data.xlsx'), sheet_name=None)
    return xl['tournaments'], xl['leagues'], xl['teams']

@st.cache_data
def load_matches():
    df = pd.read_excel(os.path.join(BASE_DIR, 'data', 'matches_data.xlsx'))
    df = df.sort_values('match_date_start').reset_index(drop=True)
    return df[
        (df['match_main_genre']=='Football') &
        df['match_plays'].notna() & df['match_watchers'].notna()
    ][['match_date_start','team_home','team_away','match_plays','match_watchers']].copy()

try:
    model, prep, tenc, threshold, le_classes = load_models()
    tournaments_df, leagues_df, teams_df     = load_reference()
    football_df                               = load_matches()
    ih = le_classes.index('High')
    il = le_classes.index('Low')
    im = le_classes.index('Medium')
    g_plays    = football_df['match_plays'].median()
    g_watchers = football_df['match_watchers'].median()
except Exception as e:
    st.error(f"Gagal load: {e}")
    st.stop()

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_teams_for_tournament(t_row, leagues_df, teams_df):
    names = [x.strip() for x in str(t_row['tournament_league']).split(',') if x.strip()]
    ids   = leagues_df[leagues_df['league_name'].isin(names)]['league_id'].tolist()
    return teams_df[teams_df['team_league'].isin(ids)]['team_name'].sort_values().tolist()

def get_hist(team, match_dt, football_df):
    past = football_df[football_df['match_date_start'] < match_dt]
    tm   = past[(past['team_home']==team)|(past['team_away']==team)]
    if len(tm)==0: return None, None, 0
    return tm['match_plays'].mean(), tm['match_watchers'].mean(), len(tm)

def apply_threshold(proba, th_h, th_l):
    return np.array([ih if p[ih]>=th_h else il if p[il]>=th_l else im for p in proba])

def fmt(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return '—'
    if val >= 1_000_000: return f'{val/1_000_000:.2f}M'
    if val >= 1_000: return f'{val/1_000:.1f}K'
    return f'{val:.0f}'

def predict_one(row_data):
    df_in = pd.DataFrame([row_data])
    TENC  = ['match_premier_status','match_tournament','match_channel','match_organization']
    df_in[TENC] = tenc.transform(df_in[TENC])
    X     = prep.transform(df_in)
    proba = model.predict_proba(X)
    pred  = apply_threshold(proba, threshold['th_high'], threshold['th_low'])[0]
    return le_classes[pred], proba[0][ih], proba[0][im], proba[0][il], proba[0]

def build_features(team_home, team_away, match_dt, tournament, channel,
                   premier_status, coverage, gender, organization,
                   exclusive, login_gating, drm, duration):
    hp, hw, hn = get_hist(team_home, match_dt, football_df)
    ap, aw, an = get_hist(team_away, match_dt, football_df)
    home_p = hp if hp is not None else g_plays
    away_p = ap if ap is not None else g_plays
    home_w = hw if hw is not None else g_watchers
    away_w = aw if aw is not None else g_watchers
    return {
        'home_hist_avg_plays':     home_p,
        'away_hist_avg_plays':     away_p,
        'match_hist_avg_plays':    (home_p+away_p)/2,
        'match_hist_max_plays':    max(home_p,away_p),
        'home_hist_avg_watchers':  home_w,
        'away_hist_avg_watchers':  away_w,
        'match_hist_avg_watchers': (home_w+away_w)/2,
        'home_n_past': hn, 'away_n_past': an,
        'is_reliable': int(hn>=3 and an>=3),
        'hour': match_dt.hour, 'month': match_dt.month,
        'duration_minutes': duration,
        'match_exclusive':    int(exclusive),
        'match_login_gating': int(login_gating),
        'match_drm':          int(drm),
        'match_gender':            str(gender)        if pd.notna(gender)        else 'Men',
        'match_coverage':          str(coverage)      if pd.notna(coverage)      else 'INDONESIA',
        'match_premier_status':    str(premier_status) if pd.notna(premier_status) else 'FREE',
        'match_tournament':        str(tournament)    if pd.notna(tournament)    else 'Unknown',
        'match_channel':           str(channel)       if pd.notna(channel)       else 'Unknown',
        'match_organization':      str(organization)  if pd.notna(organization)  else 'Unknown',
    }, hn, an, home_p, home_w, away_p, away_w

# ── App State ──────────────────────────────────────────────────────────────────
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'single'          # 'single' | 'bulk'
if 'view' not in st.session_state:
    st.session_state.view = 'form'                   # 'form' | 'result'
if 'single_result' not in st.session_state:
    st.session_state.single_result = None
if 'bulk_result' not in st.session_state:
    st.session_state.bulk_result = None
if 'bulk_res_df' not in st.session_state:
    st.session_state.bulk_res_df = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">⚽ Match <span>Priority</span></div>
    <div class="app-sub">&nbsp;· Sport Calendar Predictor</div>
</div>
<hr class="app-divider">
""", unsafe_allow_html=True)

# ── Tab Navigation (only on form view) ───────────────────────────────────────
if st.session_state.view == 'form':
    tab_c1, tab_c2 = st.columns(2)
    with tab_c1:
        if st.button("Single Predict", key="tab_single",
                     type="primary" if st.session_state.active_tab=='single' else "secondary"):
            st.session_state.active_tab = 'single'
            st.rerun()
    with tab_c2:
        if st.button("Bulk Upload CSV", key="tab_bulk",
                     type="primary" if st.session_state.active_tab=='bulk' else "secondary"):
            st.session_state.active_tab = 'bulk'
            st.rerun()
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW: RESULT — SINGLE PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.view == 'result' and st.session_state.active_tab == 'single':
    r = st.session_state.single_result

    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Kembali ke form"):
        st.session_state.view = 'form'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-wrap result-{r['label']}">
        <div class="result-eyebrow">Match Priority Level</div>
        <div class="result-label color-{r['label']}">{r['label']}</div>
        <div class="confidence-pill">Confidence&nbsp;<strong>{r['conf']:.1f}%</strong></div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="chips">
        <span class="chip">🏆 {r['tournament']}</span>
        <span class="chip">🏠 {r['home']} vs ✈️ {r['away']}</span>
        <span class="chip">📺 {r['channel']}</span>
        <span class="chip">📅 {r['date']} {r['time']}</span>
        <span class="chip">⏱️ {r['duration']} mnt</span>
        <span class="chip">{"🔒 Exclusive" if r['exclusive'] else "🔓 Open"}</span>
        <span class="chip">{"🔑 Login" if r['login_gating'] else "🚪 Free"}</span>
    </div>""", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        st.markdown('<div class="group-title">Probabilitas</div>', unsafe_allow_html=True)
        for lbl, prob, color in [('High',r['ph'],'var(--high)'),('Medium',r['pm'],'var(--med)'),('Low',r['pl'],'var(--low)')]:
            pct = prob*100
            st.markdown(f"""<div class="prob-item">
                <div class="prob-header">
                    <span class="prob-name" style="color:{color}">{lbl}</span>
                    <span class="prob-pct" style="color:{color}">{pct:.1f}%</span>
                </div>
                <div class="prob-track"><div class="prob-fill" style="width:{pct}%;background:{color};"></div></div>
            </div>""", unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="group-title">Historis Tim</div>', unsafe_allow_html=True)
        for team, p, w, n in [(r['home'],r['home_p'],r['home_w'],r['hn']),(r['away'],r['away_p'],r['away_w'],r['an'])]:
            bc = "badge-ok" if n>=3 else "badge-warn"
            bt = f"✓ {n} matches" if n>=3 else f"⚠ {n} matches"
            st.markdown(f"""<div class="hist-card" style="margin-bottom:0.6rem">
                <div class="hist-team">{team}</div>
                <div class="hist-row"><span class="hist-key">Avg Plays</span><span class="hist-val">{fmt(p)}</span></div>
                <div class="hist-row"><span class="hist-key">Avg Watchers</span><span class="hist-val">{fmt(w)}</span></div>
                <span class="hist-badge {bc}">{bt}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW: RESULT — BULK PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.view == 'result' and st.session_state.active_tab == 'bulk':
    df_result = st.session_state.bulk_result
    res_df    = st.session_state.bulk_res_df

    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Kembali ke upload"):
        st.session_state.view = 'form'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    counts = res_df['match_priority_level'].value_counts()
    n_h = counts.get('High',0)
    n_m = counts.get('Medium',0)
    n_l = counts.get('Low',0)

    st.markdown('<div class="group-title">Hasil Prediksi</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="stat-row">
        <div class="stat-box"><div class="stat-num color-HIGH">{n_h}</div><div class="stat-lbl">High</div></div>
        <div class="stat-box"><div class="stat-num color-MEDIUM">{n_m}</div><div class="stat-lbl">Medium</div></div>
        <div class="stat-box"><div class="stat-num color-LOW">{n_l}</div><div class="stat-lbl">Low</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="group-title">Preview</div>', unsafe_allow_html=True)
    show_cols = ['team_home','team_away','match_tournament','match_date_start',
                 'match_priority_level','prob_high','prob_medium','prob_low']
    show_cols = [c for c in show_cols if c in df_result.columns]
    df_show   = df_result[show_cols].head(20)

    rows_html = ""
    for _, rr in df_show.iterrows():
        lbl = rr.get('match_priority_level','—')
        badge = f'<span class="badge badge-{lbl}">{lbl}</span>' if lbl in ['High','Medium','Low'] else lbl
        dt_str = str(rr.get('match_date_start',''))[:16] if pd.notna(rr.get('match_date_start')) else '—'
        rows_html += f"""<tr>
            <td>{rr.get('team_home','—')}</td>
            <td>{rr.get('team_away','—')}</td>
            <td style="font-size:0.72rem;color:var(--text2)">{str(rr.get('match_tournament','—'))[:25]}</td>
            <td style="font-size:0.72rem">{dt_str}</td>
            <td>{badge}</td>
            <td style="color:var(--high)">{rr.get('prob_high','—')}%</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto; border:1px solid var(--border); border-radius:10px; overflow:hidden;">
    <table class="result-table">
        <thead><tr>
            <th>Home</th><th>Away</th><th>Tournament</th><th>Date</th><th>Priority</th><th>P(High)</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    <div style="font-size:0.7rem;color:var(--text3);margin-top:0.4rem">
        Menampilkan {min(20,len(df_result))} dari {len(df_result)} baris
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    csv_out = df_result.to_csv(index=False)
    st.download_button(
        label="Download hasil CSV",
        data=csv_out,
        file_name="bulk_predict_result.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# VIEW: FORM — SINGLE PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == 'single':

    col_left = st.container()

    with col_left:
        st.markdown('<div class="group-title">Tournament</div>', unsafe_allow_html=True)
        tournament_list     = sorted(tournaments_df['tournament_title'].dropna().tolist())
        selected_tournament = st.selectbox("Pilih tournament", tournament_list, index=0)
        t_row = tournaments_df[tournaments_df['tournament_title']==selected_tournament].iloc[0]
        channels_raw    = str(t_row['tournament_channel']) if pd.notna(t_row['tournament_channel']) else ''
        channel_options = [x.strip() for x in channels_raw.split(',') if x.strip()]

        af1, af2, af3, af4 = st.columns(4)
        def af_card(col, label, val):
            col.markdown(f"""<div class="card"><div class="card-label">{label}</div>
                <div class="card-value">{val if val and str(val)!='nan' else '—'}</div></div>""", unsafe_allow_html=True)
        af_card(af1, "Premier", t_row['tournament_premier'] if pd.notna(t_row['tournament_premier']) else '—')
        af_card(af2, "Coverage", t_row['tournament_coverage'] if pd.notna(t_row['tournament_coverage']) else '—')
        af_card(af3, "Gender",   t_row['tournament_gender']   if pd.notna(t_row['tournament_gender'])   else '—')
        af_card(af4, "Org",      t_row['tournament_organization'] if pd.notna(t_row['tournament_organization']) else '—')
        selected_channel = st.selectbox("Channel siaran", options=channel_options if channel_options else ['Unknown'])

        st.markdown('<div class="group-title">Tim</div>', unsafe_allow_html=True)
        available_teams = get_teams_for_tournament(t_row, leagues_df, teams_df)
        if not available_teams:
            st.warning("Tidak ada tim terdaftar."); st.stop()
        tc1, tc2 = st.columns(2)
        with tc1: selected_home = st.selectbox("🏠 Team home", available_teams, index=0)
        with tc2:
            away_opts     = [t for t in available_teams if t!=selected_home]
            selected_away = st.selectbox("✈️ Team away", away_opts, index=0)

        st.markdown('<div class="group-title">Jadwal & akses</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1: match_date = st.date_input("Tanggal", value=date.today())
        with sc2: match_time_val = st.time_input("Kick-off", value=time(20, 0))
        with sc3: duration = st.number_input("Durasi (menit)", min_value=30, max_value=600, value=120, step=15)

        fl1, fl2, fl3 = st.columns(3)
        with fl1: match_exclusive    = st.checkbox("🔒 Exclusive",    value=False)
        with fl2: match_login_gating = st.checkbox("🔑 Login Gating", value=False)
        with fl3: match_drm          = st.checkbox("🛡️ DRM",          value=True)

        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="primary-action">', unsafe_allow_html=True)
        predict_btn = st.button("Predict Priority")
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Memprediksi..."):
                match_dt = datetime.datetime.combine(match_date, match_time_val)
                row_data, hn, an, home_p, home_w, away_p, away_w = build_features(
                    selected_home, selected_away, match_dt,
                    selected_tournament, selected_channel,
                    t_row.get('tournament_premier'), t_row.get('tournament_coverage'),
                    t_row.get('tournament_gender'), t_row.get('tournament_organization'),
                    match_exclusive, match_login_gating, match_drm, duration
                )
                label, ph, pm, pl, proba_arr = predict_one(row_data)
                conf = max(ph, pm, pl) * 100

            st.session_state.single_result = {
                'label': label, 'conf': conf, 'ph': ph, 'pm': pm, 'pl': pl,
                'tournament': selected_tournament, 'home': selected_home, 'away': selected_away,
                'channel': selected_channel, 'date': match_date, 'time': match_time_val.strftime('%H:%M'),
                'duration': duration, 'exclusive': match_exclusive, 'login_gating': match_login_gating,
                'home_p': home_p, 'home_w': home_w, 'away_p': away_p, 'away_w': away_w,
                'hn': hn, 'an': an,
            }
            st.session_state.view = 'result'
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: FORM — BULK UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
else:
    bl = st.container()

    with bl:
        st.markdown('<div class="group-title">Upload CSV</div>', unsafe_allow_html=True)

        st.markdown("""<div class="bulk-info">
            <div class="bulk-info-title">Format file</div>
            <div class="bulk-info-text">
                Upload CSV dengan kolom berikut (sama seperti format <code>matches_data.xlsx</code>).
                Kolom <code>match_priority_level</code> akan diisi otomatis oleh model.
            </div>
        </div>""", unsafe_allow_html=True)

        required_cols = [
            'match_date_start', 'team_home', 'team_away',
            'match_tournament', 'match_channel', 'match_premier_status',
            'match_coverage', 'match_gender', 'match_organization',
            'match_exclusive', 'match_login_gating', 'match_drm', 'match_duration'
        ]
        chips_html = ''.join([f'<span class="required-col">{c}</span>' for c in required_cols])
        st.markdown(f'<div style="margin-bottom:1rem">{chips_html}</div>', unsafe_allow_html=True)

        template_df = pd.DataFrame(columns=required_cols + ['match_main_genre', 'match_priority_level'])
        template_df.loc[0] = [
            '2026-05-10 19:30:00', 'Manchester United', 'Liverpool',
            'Premier League 2025/26', 'CTV 5', 'PREMIER LEAGUE , ULTIMATE',
            'INDONESIA , TIMOR LESTE', 'Men', 'FIFA; UEFA; European Leagues; The FA',
            0, 1, 1, '02:00:00', 'Football', ''
        ]
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="Download template CSV",
            data=csv_template,
            file_name="template_bulk_predict.csv",
            mime="text/csv",
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)

            missing = [c for c in required_cols if c not in df_upload.columns]
            if missing:
                st.error(f"Kolom tidak ditemukan: {', '.join(missing)}")
            else:
                df_valid = df_upload.copy()
                if 'match_main_genre' in df_valid.columns:
                    df_valid = df_valid[df_valid['match_main_genre'].str.lower().str.strip() == 'football']
                df_valid = df_valid[df_valid['team_home'].notna() & df_valid['team_away'].notna()]
                df_valid = df_valid.reset_index(drop=True)

                n_total   = len(df_upload)
                n_valid   = len(df_valid)
                n_skipped = n_total - n_valid

                st.markdown(f"""<div class="stat-row">
                    <div class="stat-box"><div class="stat-num">{n_total}</div><div class="stat-lbl">Total rows</div></div>
                    <div class="stat-box"><div class="stat-num" style="color:var(--accent)">{n_valid}</div><div class="stat-lbl">Valid (football)</div></div>
                    <div class="stat-box"><div class="stat-num" style="color:var(--text3)">{n_skipped}</div><div class="stat-lbl">Skipped</div></div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="primary-action">', unsafe_allow_html=True)
                run_bulk = st.button("Jalankan bulk predict")
                st.markdown('</div>', unsafe_allow_html=True)

                if run_bulk and n_valid > 0:
                    results = []
                    progress = st.progress(0, text="Memproses...")

                    for i, row in df_valid.iterrows():
                        try:
                            match_dt = pd.to_datetime(row['match_date_start'])
                            if pd.isna(match_dt):
                                match_dt = datetime.datetime.now()
                        except:
                            match_dt = datetime.datetime.now()

                        try:
                            dur_raw = str(row['match_duration'])
                            parts   = dur_raw.split(':')
                            dur_min = int(parts[0])*60 + int(parts[1]) if len(parts)>=2 else 120
                        except:
                            dur_min = 120

                        row_data, hn, an, hp, hw, ap, aw = build_features(
                            str(row['team_home']), str(row['team_away']), match_dt,
                            str(row['match_tournament'])    if pd.notna(row.get('match_tournament'))    else 'Unknown',
                            str(row['match_channel'])       if pd.notna(row.get('match_channel'))       else 'Unknown',
                            row.get('match_premier_status'), row.get('match_coverage'),
                            row.get('match_gender'), row.get('match_organization'),
                            bool(row.get('match_exclusive', 0)),
                            bool(row.get('match_login_gating', 0)),
                            bool(row.get('match_drm', 1)),
                            dur_min
                        )
                        label, ph, pm, pl, _ = predict_one(row_data)
                        results.append({
                            'match_priority_level': label,
                            'prob_high':   round(ph*100,1),
                            'prob_medium': round(pm*100,1),
                            'prob_low':    round(pl*100,1),
                            'home_n_matches': hn,
                            'away_n_matches': an,
                        })
                        progress.progress((i+1)/n_valid, text=f"Memproses {i+1}/{n_valid}...")

                    progress.empty()

                    df_result = df_valid.copy()
                    res_df    = pd.DataFrame(results)
                    df_result['match_priority_level'] = res_df['match_priority_level'].values
                    df_result['prob_high']   = res_df['prob_high'].values
                    df_result['prob_medium'] = res_df['prob_medium'].values
                    df_result['prob_low']    = res_df['prob_low'].values

                    st.session_state.bulk_result = df_result
                    st.session_state.bulk_res_df = res_df
                    st.session_state.view = 'result'
                    st.rerun()
