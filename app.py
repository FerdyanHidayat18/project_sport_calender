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

# ── Global Styles (Clean White & Red) ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --red:        #E0142C;
    --red-dark:   #B30E22;
    --red-soft:   #FFEDEF;
    --red-tint:   #FFF5F6;
    --ink:        #1A1A1E;
    --ink-soft:   #5C5C66;
    --line:       #EBEBEF;
    --bg:         #FFFFFF;
    --bg-grey:    #FAFAFC;
    --green:      #1A9E5C;
    --blue:       #1B6FE0;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    font-size: 16px;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1180px; }

/* ── Top Navbar ───────────────────────────────────────────────────────── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 0; border-bottom: 2px solid var(--ink);
    margin-bottom: 0;
}
.navbar-brand {
    display: flex; align-items: center; gap: 0.6rem;
    font-family: 'Manrope', sans-serif; font-weight: 800;
    font-size: 1.5rem; letter-spacing: -0.5px; color: var(--ink);
}
.navbar-brand .mark {
    background: var(--red); color: #fff; width: 38px; height: 38px;
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem;
}
.navbar-tagline {
    font-size: 0.75rem; color: var(--ink-soft); font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
}

/* ── Nav Tabs ─────────────────────────────────────────────────────────── */
.nav-row { display: flex; gap: 0.5rem; padding: 1.1rem 0; }

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.2px !important;
    border-radius: 999px !important;
    padding: 0.65rem 1.6rem !important;
    border: 2px solid var(--line) !important;
    background: var(--bg) !important;
    color: var(--ink-soft) !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
    border-color: var(--red) !important;
    color: var(--red) !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--red) !important;
    border-color: var(--red) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(224,20,44,0.25) !important;
}

/* ── Section Headings ─────────────────────────────────────────────────── */
.section-title {
    font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.3rem;
    color: var(--ink); margin: 2rem 0 0.4rem 0; letter-spacing: -0.3px;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-title .dot {
    width: 10px; height: 10px; border-radius: 50%; background: var(--red);
    display: inline-block; flex-shrink: 0;
}
.section-sub {
    font-size: 0.92rem; color: var(--ink-soft); margin-bottom: 1.1rem;
    font-weight: 400;
}

/* ── Cards ────────────────────────────────────────────────────────────── */
.info-card {
    background: var(--bg-grey); border: 1px solid var(--line); border-radius: 14px;
    padding: 1rem 1.2rem; height: 100%;
}
.info-card-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--ink-soft); margin-bottom: 0.35rem;
}
.info-card-value {
    font-size: 1.05rem; font-weight: 700; color: var(--ink); line-height: 1.35;
}

/* ── Result Hero ──────────────────────────────────────────────────────── */
.result-hero {
    border-radius: 20px; padding: 2.4rem 1.8rem; text-align: center;
    margin-bottom: 1.5rem; border: 2px solid var(--ink);
}
.result-hero.HIGH   { background: var(--red-soft); border-color: var(--red); }
.result-hero.MEDIUM { background: #EAF2FF; border-color: var(--blue); }
.result-hero.LOW    { background: #E9F8EF; border-color: var(--green); }

.result-eyebrow {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
    color: var(--ink-soft); margin-bottom: 0.6rem;
}
.result-label {
    font-family: 'Manrope', sans-serif; font-size: 4rem; font-weight: 800; line-height: 1;
    letter-spacing: -2px; margin-bottom: 0.7rem;
}
.result-label.HIGH   { color: var(--red); }
.result-label.MEDIUM { color: var(--blue); }
.result-label.LOW    { color: var(--green); }

.confidence-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: #fff; border: 2px solid var(--ink); border-radius: 999px;
    padding: 0.5rem 1.3rem; font-size: 1.05rem; font-weight: 700; color: var(--ink);
}

/* ── Probability Bars ─────────────────────────────────────────────────── */
.prob-item { margin-bottom: 1rem; }
.prob-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem; }
.prob-name { font-size: 0.98rem; font-weight: 700; }
.prob-pct  { font-size: 1.1rem; font-weight: 800; }
.prob-track { height: 12px; background: var(--bg-grey); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 8px; }

/* ── Team History Cards ───────────────────────────────────────────────── */
.hist-card {
    background: var(--bg); border: 2px solid var(--ink); border-radius: 14px;
    padding: 1.1rem 1.3rem; height: 100%;
}
.hist-team {
    font-family: 'Manrope', sans-serif; font-size: 1.05rem; font-weight: 800;
    color: var(--ink); margin-bottom: 0.7rem; padding-bottom: 0.6rem;
    border-bottom: 2px solid var(--red);
}
.hist-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.35rem 0; font-size: 0.95rem;
}
.hist-key { color: var(--ink-soft); font-weight: 500; }
.hist-val { font-weight: 800; color: var(--ink); }
.hist-badge {
    display: inline-block; font-size: 0.7rem; font-weight: 700; padding: 0.25rem 0.7rem;
    border-radius: 999px; margin-top: 0.6rem; letter-spacing: 0.5px;
}
.badge-ok   { background: #E9F8EF; color: var(--green); }
.badge-warn { background: #FFF7E5; color: #C77700; }

/* ── Summary Chips ────────────────────────────────────────────────────── */
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.6rem; }
.chip {
    background: var(--bg-grey); border: 1px solid var(--line); border-radius: 999px;
    padding: 0.4rem 1rem; font-size: 0.85rem; font-weight: 600; color: var(--ink);
    white-space: nowrap;
}

/* ── Bulk: stat boxes ─────────────────────────────────────────────────── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; }
.stat-box {
    flex: 1; background: var(--bg-grey); border: 2px solid var(--line); border-radius: 14px;
    padding: 1.1rem 1rem; text-align: center;
}
.stat-num { font-family: 'Manrope', sans-serif; font-size: 2rem; font-weight: 800; color: var(--ink); }
.stat-lbl { font-size: 0.78rem; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0.3rem; font-weight: 700; }
.stat-num.HIGH   { color: var(--red); }
.stat-num.MEDIUM { color: var(--blue); }
.stat-num.LOW    { color: var(--green); }

/* ── Required cols chips ─────────────────────────────────────────────── */
.required-col {
    display: inline-block; background: var(--bg-grey); border: 1px solid var(--line);
    border-radius: 6px; padding: 0.25rem 0.6rem; font-size: 0.8rem; font-family: monospace;
    color: var(--red-dark); margin: 0.2rem; font-weight: 600;
}

/* ── Result Table ─────────────────────────────────────────────────────── */
.result-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.result-table th {
    background: var(--ink); color: #fff; font-weight: 700; font-size: 0.72rem;
    letter-spacing: 1.5px; text-transform: uppercase; padding: 0.8rem 1rem; text-align: left;
}
.result-table td { padding: 0.75rem 1rem; border-bottom: 1px solid var(--line); color: var(--ink); vertical-align: middle; }
.result-table tr:last-child td { border-bottom: none; }
.result-table tr:hover td { background: var(--bg-grey); }
.badge { display: inline-block; font-size: 0.75rem; font-weight: 800; padding: 0.3rem 0.8rem; border-radius: 999px; letter-spacing: 0.5px; }
.badge-High   { background: var(--red-soft); color: var(--red); }
.badge-Medium { background: #EAF2FF; color: var(--blue); }
.badge-Low    { background: #E9F8EF; color: var(--green); }

/* ── Empty State ──────────────────────────────────────────────────────── */
.empty-state { text-align: center; padding: 4rem 1rem; }
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-title {
    font-family: 'Manrope', sans-serif; font-size: 1.3rem; font-weight: 800;
    color: var(--ink); margin-bottom: 0.5rem;
}
.empty-sub { font-size: 1rem; color: var(--ink-soft); line-height: 1.6; }

/* ── Form Controls ────────────────────────────────────────────────────── */
div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stTimeInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stFileUploader"] label {
    color: var(--ink) !important; font-size: 0.85rem !important;
    font-weight: 700 !important; letter-spacing: 0.5px !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: var(--bg) !important; border: 2px solid var(--line) !important;
    color: var(--ink) !important; border-radius: 10px !important; font-size: 1rem !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--red) !important;
}
div[data-testid="stDateInput"] input, div[data-testid="stTimeInput"] input, div[data-testid="stNumberInput"] input {
    border: 2px solid var(--line) !important; border-radius: 10px !important;
    font-size: 1rem !important; color: var(--ink) !important;
}
div[data-testid="stFileUploader"] > div {
    background: var(--bg-grey) !important; border: 2px dashed var(--line) !important; border-radius: 14px !important;
}
div[data-testid="stCheckbox"] {
    background: var(--bg-grey); border: 2px solid var(--line); border-radius: 12px; padding: 0.7rem 1rem;
}
div[data-testid="stCheckbox"] label p { font-size: 1rem !important; font-weight: 600 !important; }

/* ── Main Predict Buttons (full width, not in horizontal block) ─────────── */
div[data-testid="stVerticalBlock"] > div > div[data-testid="stButton"] > button {
    background: var(--red) !important;
    color: #fff !important; font-family: 'Manrope', sans-serif !important; font-size: 1.05rem !important;
    font-weight: 800 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important;
    border: none !important; border-radius: 12px !important; padding: 0.95rem 2rem !important;
    width: 100% !important; box-shadow: 0 6px 18px rgba(224,20,44,0.28) !important;
    transition: transform 0.1s ease, opacity 0.15s ease !important;
}
div[data-testid="stVerticalBlock"] > div > div[data-testid="stButton"] > button:hover {
    opacity: 0.92 !important; transform: translateY(-1px) !important;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid var(--line); margin: 1.5rem 0; }

/* ── Alert ────────────────────────────────────────────────────────────── */
.alert-banner {
    background: var(--red-tint); border: 2px solid var(--red); border-radius: 12px;
    padding: 0.9rem 1.2rem; font-size: 0.95rem; font-weight: 600; color: var(--red-dark);
    margin-bottom: 1rem;
}
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

# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand"><span class="mark">⚽</span> Match Priority</div>
    <div class="navbar-tagline">Sport Calendar · Prediction Tool</div>
</div>
""", unsafe_allow_html=True)

# ── Tab State ──────────────────────────────────────────────────────────────────
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'single'

nav_c1, nav_c2, nav_spacer = st.columns([2, 2, 6])
with nav_c1:
    if st.button("Single Predict", key="tab_single",
                 type="primary" if st.session_state.active_tab=='single' else "secondary"):
        st.session_state.active_tab = 'single'
        st.rerun()
with nav_c2:
    if st.button("Bulk Upload", key="tab_bulk",
                 type="primary" if st.session_state.active_tab=='bulk' else "secondary"):
        st.session_state.active_tab = 'bulk'
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: SINGLE PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == 'single':

    col_left, col_right = st.columns([5, 4], gap="large")

    with col_left:
        st.markdown('<div class="section-title"><span class="dot"></span>Tournament</div>', unsafe_allow_html=True)
        tournament_list     = sorted(tournaments_df['tournament_title'].dropna().tolist())
        selected_tournament = st.selectbox("Pilih Tournament", tournament_list, index=0, label_visibility="collapsed")
        t_row = tournaments_df[tournaments_df['tournament_title']==selected_tournament].iloc[0]
        channels_raw    = str(t_row['tournament_channel']) if pd.notna(t_row['tournament_channel']) else ''
        channel_options = [x.strip() for x in channels_raw.split(',') if x.strip()]

        st.markdown('<div class="section-title"><span class="dot"></span>Match Info</div>', unsafe_allow_html=True)
        af1, af2, af3, af4 = st.columns(4)
        def af_card(col, label, val):
            col.markdown(f"""<div class="info-card"><div class="info-card-label">{label}</div>
                <div class="info-card-value">{val if val and str(val)!='nan' else '—'}</div></div>""", unsafe_allow_html=True)
        af_card(af1, "Premier", t_row['tournament_premier'] if pd.notna(t_row['tournament_premier']) else '—')
        af_card(af2, "Coverage", t_row['tournament_coverage'] if pd.notna(t_row['tournament_coverage']) else '—')
        af_card(af3, "Gender",   t_row['tournament_gender']   if pd.notna(t_row['tournament_gender'])   else '—')
        af_card(af4, "Org",      t_row['tournament_organization'] if pd.notna(t_row['tournament_organization']) else '—')
        st.markdown("<br>", unsafe_allow_html=True)
        selected_channel = st.selectbox("Channel Siaran", options=channel_options if channel_options else ['Unknown'])

        st.markdown('<div class="section-title"><span class="dot"></span>Teams</div>', unsafe_allow_html=True)
        available_teams = get_teams_for_tournament(t_row, leagues_df, teams_df)
        if not available_teams:
            st.warning("Tidak ada tim terdaftar untuk turnamen ini."); st.stop()
        tc1, tc2 = st.columns(2)
        with tc1: selected_home = st.selectbox("Team Home", available_teams, index=0)
        with tc2:
            away_opts     = [t for t in available_teams if t!=selected_home]
            selected_away = st.selectbox("Team Away", away_opts, index=0)

        st.markdown('<div class="section-title"><span class="dot"></span>Schedule</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1: match_date = st.date_input("Tanggal", value=date.today())
        with sc2: match_time_val = st.time_input("Kick-off", value=time(20, 0))
        with sc3: duration = st.number_input("Durasi (menit)", min_value=30, max_value=600, value=120, step=15)

        st.markdown('<div class="section-title"><span class="dot"></span>Access Flags</div>', unsafe_allow_html=True)
        fl1, fl2, fl3 = st.columns(3)
        with fl1: match_exclusive    = st.checkbox("Exclusive", value=False)
        with fl2: match_login_gating = st.checkbox("Login Gating", value=False)
        with fl3: match_drm          = st.checkbox("DRM", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Predict Priority")

    with col_right:
        if predict_btn:
            with st.spinner(""):
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

            st.markdown(f"""
            <div class="result-hero {label.upper()}">
                <div class="result-eyebrow">Match Priority Level</div>
                <div class="result-label {label.upper()}">{label}</div>
                <div class="confidence-pill">Confidence&nbsp;&nbsp;<strong>{conf:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-title"><span class="dot"></span>Probabilitas</div>', unsafe_allow_html=True)
            color_map = {'High':'var(--red)','Medium':'var(--blue)','Low':'var(--green)'}
            for lbl, prob in [('High',ph),('Medium',pm),('Low',pl)]:
                pct = prob*100
                color = color_map[lbl]
                st.markdown(f"""<div class="prob-item">
                    <div class="prob-header">
                        <span class="prob-name" style="color:{color}">{lbl}</span>
                        <span class="prob-pct" style="color:{color}">{pct:.1f}%</span>
                    </div>
                    <div class="prob-track"><div class="prob-fill" style="width:{pct}%;background:{color};"></div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-title"><span class="dot"></span>Historis Tim</div>', unsafe_allow_html=True)
            hc1, hc2 = st.columns(2)
            for col_h, team, p, w, n in [(hc1,selected_home,home_p,home_w,hn),(hc2,selected_away,away_p,away_w,an)]:
                bc = "badge-ok" if n>=3 else "badge-warn"
                bt = f"{n} matches tersedia" if n>=3 else f"Data terbatas — {n} matches"
                col_h.markdown(f"""<div class="hist-card">
                    <div class="hist-team">{team}</div>
                    <div class="hist-row"><span class="hist-key">Avg Plays</span><span class="hist-val">{fmt(p)}</span></div>
                    <div class="hist-row"><span class="hist-key">Avg Watchers</span><span class="hist-val">{fmt(w)}</span></div>
                    <span class="hist-badge {bc}">{bt}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-title"><span class="dot"></span>Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="chips">
                <span class="chip">{selected_tournament}</span>
                <span class="chip">{selected_home} vs {selected_away}</span>
                <span class="chip">{selected_channel}</span>
                <span class="chip">{match_date} · {match_time_val.strftime('%H:%M')}</span>
                <span class="chip">{duration} menit</span>
                <span class="chip">{"Exclusive" if match_exclusive else "Open"}</span>
                <span class="chip">{"Login Gating" if match_login_gating else "Akses Bebas"}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="empty-state">
                <div class="empty-icon">⚽</div>
                <div class="empty-title">Siap Memprediksi</div>
                <div class="empty-sub">Lengkapi form di sebelah kiri, lalu klik<br><strong>Predict Priority</strong> untuk melihat hasilnya.</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: BULK UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
else:
    bl, br = st.columns([5, 4], gap="large")

    with bl:
        st.markdown('<div class="section-title"><span class="dot"></span>Upload CSV</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Unggah file CSV dengan format yang sama seperti <strong>matches_data.xlsx</strong>. Kolom <strong>match_priority_level</strong> akan diisi otomatis oleh model.</div>', unsafe_allow_html=True)

        required_cols = [
            'match_date_start', 'team_home', 'team_away',
            'match_tournament', 'match_channel', 'match_premier_status',
            'match_coverage', 'match_gender', 'match_organization',
            'match_exclusive', 'match_login_gating', 'match_drm', 'match_duration'
        ]
        chips_html = ''.join([f'<span class="required-col">{c}</span>' for c in required_cols])
        st.markdown(f'<div style="margin-bottom:1.2rem">{chips_html}</div>', unsafe_allow_html=True)

        # Download template
        template_df = pd.DataFrame(columns=required_cols + ['match_main_genre', 'match_priority_level'])
        template_df.loc[0] = [
            '2026-05-10 19:30:00', 'Manchester United', 'Liverpool',
            'Premier League 2025/26', 'CTV 5', 'PREMIER LEAGUE , ULTIMATE',
            'INDONESIA , TIMOR LESTE', 'Men', 'FIFA; UEFA; European Leagues; The FA',
            0, 1, 1, '02:00:00', 'Football', ''
        ]
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="Download Template CSV",
            data=csv_template,
            file_name="template_bulk_predict.csv",
            mime="text/csv",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")

        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)

            missing = [c for c in required_cols if c not in df_upload.columns]
            if missing:
                st.markdown(f'<div class="alert-banner">Kolom tidak ditemukan: {", ".join(missing)}</div>', unsafe_allow_html=True)
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
                    <div class="stat-box"><div class="stat-num">{n_total}</div><div class="stat-lbl">Total Rows</div></div>
                    <div class="stat-box"><div class="stat-num" style="color:var(--green)">{n_valid}</div><div class="stat-lbl">Valid (Football)</div></div>
                    <div class="stat-box"><div class="stat-num" style="color:var(--ink-soft)">{n_skipped}</div><div class="stat-lbl">Skipped</div></div>
                </div>""", unsafe_allow_html=True)

                run_bulk = st.button("Jalankan Bulk Predict")

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

                    st.session_state['bulk_result'] = df_result
                    st.session_state['bulk_res_df'] = res_df

    with br:
        if 'bulk_result' in st.session_state:
            df_result = st.session_state['bulk_result']
            res_df    = st.session_state['bulk_res_df']

            counts = res_df['match_priority_level'].value_counts()
            n_h = counts.get('High',0)
            n_m = counts.get('Medium',0)
            n_l = counts.get('Low',0)

            st.markdown('<div class="section-title"><span class="dot"></span>Hasil Prediksi</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="stat-row">
                <div class="stat-box"><div class="stat-num HIGH">{n_h}</div><div class="stat-lbl">High</div></div>
                <div class="stat-box"><div class="stat-num MEDIUM">{n_m}</div><div class="stat-lbl">Medium</div></div>
                <div class="stat-box"><div class="stat-num LOW">{n_l}</div><div class="stat-lbl">Low</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-title"><span class="dot"></span>Preview</div>', unsafe_allow_html=True)
            show_cols = ['team_home','team_away','match_tournament','match_date_start',
                         'match_priority_level','prob_high','prob_medium','prob_low']
            show_cols = [c for c in show_cols if c in df_result.columns]
            df_show   = df_result[show_cols].head(20)

            rows_html = ""
            for _, r in df_show.iterrows():
                lbl = r.get('match_priority_level','—')
                badge = f'<span class="badge badge-{lbl}">{lbl}</span>' if lbl in ['High','Medium','Low'] else lbl
                dt_str = str(r.get('match_date_start',''))[:16] if pd.notna(r.get('match_date_start')) else '—'
                tour = str(r.get('match_tournament','—'))[:25]
                rows_html += f"""<tr>
                    <td>{r.get('team_home','—')}</td>
                    <td>{r.get('team_away','—')}</td>
                    <td style="font-size:0.8rem;color:var(--ink-soft)">{tour}</td>
                    <td style="font-size:0.8rem">{dt_str}</td>
                    <td>{badge}</td>
                    <td style="color:var(--red);font-weight:700">{r.get('prob_high','—')}%</td>
                </tr>"""

            st.markdown(f"""
            <div style="overflow-x:auto; border:2px solid var(--ink); border-radius:14px; overflow:hidden;">
            <table class="result-table">
                <thead><tr>
                    <th>Home</th><th>Away</th><th>Tournament</th><th>Date</th><th>Priority</th><th>P(High)</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            </div>
            <div style="font-size:0.8rem;color:var(--ink-soft);margin-top:0.5rem">
                Menampilkan {min(20,len(df_result))} dari {len(df_result)} baris
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            csv_out = df_result.to_csv(index=False)
            st.download_button(
                label="Download Hasil CSV",
                data=csv_out,
                file_name="bulk_predict_result.csv",
                mime="text/csv",
            )
        else:
            st.markdown("""<div class="empty-state">
                <div class="empty-icon">📂</div>
                <div class="empty-title">Belum Ada Hasil</div>
                <div class="empty-sub">Unggah file CSV di sebelah kiri,<br>lalu klik <strong>Jalankan Bulk Predict</strong>.</div>
            </div>""", unsafe_allow_html=True)
