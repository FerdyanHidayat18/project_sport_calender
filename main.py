import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import datetime
from datetime import date, time
import io

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

# ── Theme State ────────────────────────────────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

def get_css(dark):
    if dark:
        return """
        --bg:#080c14; --bg2:#0e1520; --bg3:#141e2e; --border:#1e2d45; --border2:#243550;
        --text:#e8edf5; --text2:#7a90b0; --text3:#4a6080;
        --accent:#00e5a0; --accent2:#0099ff;
        --high:#ff4d6d; --med:#4d9fff; --low:#00e5a0;
        --card-high:linear-gradient(135deg,#1a0810 0%,#2a0d18 100%);
        --card-med:linear-gradient(135deg,#080f1e 0%,#0d1830 100%);
        --card-low:linear-gradient(135deg,#081510 0%,#0d2218 100%);
        --shadow:0 8px 32px rgba(0,0,0,0.5); --shadow2:0 2px 12px rgba(0,0,0,0.3);
        --tab-active:#0e1520; --tab-bg:#080c14;
        """
    else:
        return """
        --bg:#f0f4f8; --bg2:#ffffff; --bg3:#e8edf5; --border:#d0dae8; --border2:#b8c8dc;
        --text:#1a2535; --text2:#4a6080; --text3:#8aa0be;
        --accent:#00a870; --accent2:#0066cc;
        --high:#e0193a; --med:#0066cc; --low:#00a870;
        --card-high:linear-gradient(135deg,#fff0f3 0%,#ffe0e7 100%);
        --card-med:linear-gradient(135deg,#f0f6ff 0%,#e0edff 100%);
        --card-low:linear-gradient(135deg,#f0fff8 0%,#e0fff0 100%);
        --shadow:0 8px 32px rgba(0,0,0,0.12); --shadow2:0 2px 12px rgba(0,0,0,0.08);
        --tab-active:#ffffff; --tab-bg:#e8edf5;
        """

dark = st.session_state.dark_mode
css_vars = get_css(dark)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@300;400;500;600&display=swap');
:root {{ {css_vars} }}
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"], .stApp {{
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg2); }}
::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}

.app-header {{ display:flex; align-items:center; justify-content:space-between; padding:1.5rem 0 1rem 0; border-bottom:1px solid var(--border); margin-bottom:1.5rem; }}
.app-title {{ font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:var(--text); letter-spacing:-0.5px; line-height:1; }}
.app-title span {{ color:var(--accent); }}
.app-sub {{ font-size:0.78rem; color:var(--text3); font-weight:400; letter-spacing:0.5px; text-transform:uppercase; margin-top:0.2rem; }}

/* Tab navigation */
.tab-nav {{ display:flex; gap:0; background:var(--bg3); border-radius:10px; padding:4px; margin-bottom:1.5rem; border:1px solid var(--border); }}
.tab-btn {{
    flex:1; padding:0.6rem 1rem; border:none; border-radius:8px; cursor:pointer;
    font-family:'Outfit',sans-serif; font-size:0.82rem; font-weight:600;
    letter-spacing:0.5px; transition:all 0.2s; background:transparent; color:var(--text2);
}}
.tab-btn.active {{ background:var(--tab-active); color:var(--text); box-shadow:var(--shadow2); }}
.tab-btn:hover:not(.active) {{ color:var(--accent); }}

.sec-head {{
    font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700; letter-spacing:3px;
    text-transform:uppercase; color:var(--accent); margin:1.5rem 0 0.8rem 0;
    display:flex; align-items:center; gap:0.5rem;
}}
.sec-head::after {{ content:''; flex:1; height:1px; background:var(--border); }}

.card {{ background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.6rem; box-shadow:var(--shadow2); }}
.card-label {{ font-size:0.68rem; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:var(--text3); margin-bottom:0.3rem; }}
.card-value {{ font-size:0.9rem; font-weight:500; color:var(--text); line-height:1.4; }}

.hist-card {{ background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.5rem; box-shadow:var(--shadow2); }}
.hist-team {{ font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:var(--accent); margin-bottom:0.6rem; }}
.hist-row {{ display:flex; justify-content:space-between; align-items:center; padding:0.25rem 0; border-bottom:1px solid var(--border); }}
.hist-row:last-of-type {{ border-bottom:none; }}
.hist-key {{ font-size:0.75rem; color:var(--text2); }}
.hist-val {{ font-size:0.85rem; font-weight:600; color:var(--text); }}
.hist-badge {{ display:inline-block; font-size:0.65rem; font-weight:600; padding:0.15rem 0.5rem; border-radius:10px; margin-top:0.5rem; letter-spacing:0.5px; }}
.badge-ok {{ background:rgba(0,229,160,0.15); color:var(--low); }}
.badge-warn {{ background:rgba(255,180,0,0.15); color:#ffb400; }}

.result-wrap {{ border-radius:16px; padding:2rem 1.5rem; text-align:center; box-shadow:var(--shadow); margin-bottom:1.5rem; position:relative; overflow:hidden; }}
.result-HIGH {{ background:var(--card-high); border:1.5px solid var(--high); }}
.result-MEDIUM {{ background:var(--card-med); border:1.5px solid var(--med); }}
.result-LOW {{ background:var(--card-low); border:1.5px solid var(--low); }}
.result-eyebrow {{ font-size:0.65rem; font-weight:700; letter-spacing:4px; text-transform:uppercase; color:var(--text3); margin-bottom:0.5rem; }}
.result-label {{ font-family:'Syne',sans-serif; font-size:3.8rem; font-weight:800; line-height:1; letter-spacing:-1px; margin-bottom:0.5rem; }}
.color-HIGH {{ color:var(--high); }} .color-MEDIUM {{ color:var(--med); }} .color-LOW {{ color:var(--low); }}
.confidence-pill {{ display:inline-flex; align-items:center; gap:0.4rem; background:rgba(255,255,255,0.07); border:1px solid var(--border2); border-radius:20px; padding:0.3rem 0.8rem; font-size:0.8rem; color:var(--text2); margin-top:0.3rem; }}

.prob-item {{ margin-bottom:0.8rem; }}
.prob-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem; }}
.prob-name {{ font-size:0.78rem; font-weight:600; letter-spacing:0.5px; }}
.prob-pct {{ font-size:0.78rem; font-weight:700; }}
.prob-track {{ height:6px; background:var(--border); border-radius:3px; overflow:hidden; }}
.prob-fill {{ height:100%; border-radius:3px; }}

.chips {{ display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.5rem; }}
.chip {{ background:var(--bg3); border:1px solid var(--border); border-radius:20px; padding:0.25rem 0.7rem; font-size:0.72rem; color:var(--text2); white-space:nowrap; }}

/* Bulk section */
.bulk-info {{ background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem; }}
.bulk-info-title {{ font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:var(--text); margin-bottom:0.5rem; }}
.bulk-info-text {{ font-size:0.8rem; color:var(--text2); line-height:1.6; }}
.required-col {{ display:inline-block; background:var(--bg3); border:1px solid var(--border2); border-radius:6px; padding:0.15rem 0.5rem; font-size:0.72rem; font-family:monospace; color:var(--accent); margin:0.15rem; }}

.result-table {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
.result-table th {{ background:var(--bg3); color:var(--text2); font-weight:600; font-size:0.7rem; letter-spacing:1px; text-transform:uppercase; padding:0.6rem 0.8rem; text-align:left; border-bottom:2px solid var(--border); }}
.result-table td {{ padding:0.55rem 0.8rem; border-bottom:1px solid var(--border); color:var(--text); vertical-align:middle; }}
.result-table tr:last-child td {{ border-bottom:none; }}
.result-table tr:hover td {{ background:var(--bg3); }}
.badge {{ display:inline-block; font-size:0.68rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:8px; letter-spacing:0.5px; }}
.badge-HIGH {{ background:rgba(255,77,109,0.15); color:var(--high); }}
.badge-MEDIUM {{ background:rgba(77,159,255,0.15); color:var(--med); }}
.badge-LOW {{ background:rgba(0,229,160,0.15); color:var(--low); }}

.stat-row {{ display:flex; gap:1rem; margin-bottom:1rem; }}
.stat-box {{ flex:1; background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:0.8rem 1rem; text-align:center; }}
.stat-num {{ font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; }}
.stat-lbl {{ font-size:0.68rem; color:var(--text3); text-transform:uppercase; letter-spacing:1px; margin-top:0.2rem; }}

.empty-state {{ text-align:center; padding:3rem 1rem; color:var(--text3); }}
.empty-icon {{ font-size:2.5rem; margin-bottom:0.8rem; opacity:0.5; }}
.empty-title {{ font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.4rem; color:var(--text2); }}
.empty-sub {{ font-size:0.8rem; color:var(--text3); }}

div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stTimeInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stFileUploader"] label {{
    color:var(--text2) !important; font-size:0.72rem !important;
    font-weight:600 !important; letter-spacing:1px !important;
    text-transform:uppercase !important; font-family:'Outfit',sans-serif !important;
}}
div[data-testid="stSelectbox"] > div > div {{ background:var(--bg2) !important; border-color:var(--border) !important; color:var(--text) !important; border-radius:8px !important; }}
div[data-testid="stFileUploader"] > div {{ background:var(--bg2) !important; border-color:var(--border) !important; border-radius:10px !important; }}
.stButton > button {{
    background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%) !important;
    color:#fff !important; font-family:'Syne',sans-serif !important; font-size:0.9rem !important;
    font-weight:700 !important; letter-spacing:2px !important; text-transform:uppercase !important;
    border:none !important; border-radius:10px !important; padding:0.8rem 2rem !important;
    width:100% !important; box-shadow:0 4px 16px rgba(0,229,160,0.25) !important;
    transition:opacity 0.2s,transform 0.1s !important;
}}
.stButton > button:hover {{ opacity:0.9 !important; transform:translateY(-1px) !important; }}
div[data-testid="stCheckbox"] {{ background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:0.6rem 0.8rem; }}
#MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{ padding-top:1rem !important; }}
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

# ── Header ─────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([6, 1])
with h1:
    st.markdown("""
    <div class="app-header">
        <div>
            <div class="app-title">⚽ Match <span>Priority</span></div>
            <div class="app-sub">BUSPRO · Sport Calendar · XGBoost</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with h2:
    if st.button("☀️ Light" if dark else "🌙 Dark", key="theme"):
        st.session_state.dark_mode = not dark
        st.rerun()

# ── Tab State ──────────────────────────────────────────────────────────────────
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'single'

tab_c1, tab_c2 = st.columns(2)
with tab_c1:
    if st.button("⚡ Single Predict", key="tab_single",
                 type="primary" if st.session_state.active_tab=='single' else "secondary"):
        st.session_state.active_tab = 'single'
        st.rerun()
with tab_c2:
    if st.button("📂 Bulk Upload CSV", key="tab_bulk",
                 type="primary" if st.session_state.active_tab=='bulk' else "secondary"):
        st.session_state.active_tab = 'bulk'
        st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:0 0 1.5rem 0'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: SINGLE PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == 'single':

    col_left, col_right = st.columns([5, 4], gap="large")

    with col_left:
        st.markdown('<div class="sec-head">01 · Tournament</div>', unsafe_allow_html=True)
        tournament_list     = sorted(tournaments_df['tournament_title'].dropna().tolist())
        selected_tournament = st.selectbox("Pilih Tournament", tournament_list, index=0, label_visibility="collapsed")
        t_row = tournaments_df[tournaments_df['tournament_title']==selected_tournament].iloc[0]
        channels_raw    = str(t_row['tournament_channel']) if pd.notna(t_row['tournament_channel']) else ''
        channel_options = [x.strip() for x in channels_raw.split(',') if x.strip()]

        st.markdown('<div class="sec-head">02 · Match Info</div>', unsafe_allow_html=True)
        af1, af2, af3, af4 = st.columns(4)
        def af_card(col, label, val):
            col.markdown(f"""<div class="card"><div class="card-label">{label}</div>
                <div class="card-value">{val if val and str(val)!='nan' else '—'}</div></div>""", unsafe_allow_html=True)
        af_card(af1, "Premier", t_row['tournament_premier'] if pd.notna(t_row['tournament_premier']) else '—')
        af_card(af2, "Coverage", t_row['tournament_coverage'] if pd.notna(t_row['tournament_coverage']) else '—')
        af_card(af3, "Gender",   t_row['tournament_gender']   if pd.notna(t_row['tournament_gender'])   else '—')
        af_card(af4, "Org",      t_row['tournament_organization'] if pd.notna(t_row['tournament_organization']) else '—')
        selected_channel = st.selectbox("Channel Siaran", options=channel_options if channel_options else ['Unknown'])

        st.markdown('<div class="sec-head">03 · Teams</div>', unsafe_allow_html=True)
        available_teams = get_teams_for_tournament(t_row, leagues_df, teams_df)
        if not available_teams:
            st.warning("Tidak ada tim terdaftar."); st.stop()
        tc1, tc2 = st.columns(2)
        with tc1: selected_home = st.selectbox("🏠 Team Home", available_teams, index=0)
        with tc2:
            away_opts     = [t for t in available_teams if t!=selected_home]
            selected_away = st.selectbox("✈️ Team Away", away_opts, index=0)

        st.markdown('<div class="sec-head">04 · Schedule</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1: match_date = st.date_input("Tanggal", value=date.today())
        with sc2: match_time_val = st.time_input("Kick-off", value=time(20, 0))
        with sc3: duration = st.number_input("Durasi (menit)", min_value=30, max_value=600, value=120, step=15)

        st.markdown('<div class="sec-head">05 · Access Flags</div>', unsafe_allow_html=True)
        fl1, fl2, fl3 = st.columns(3)
        with fl1: match_exclusive    = st.checkbox("🔒 Exclusive",    value=False)
        with fl2: match_login_gating = st.checkbox("🔑 Login Gating", value=False)
        with fl3: match_drm          = st.checkbox("🛡️ DRM",          value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ PREDICT PRIORITY")

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
                th_h = threshold['th_high']; th_l = threshold['th_low']

            icon_map = {'High':'🔴','Medium':'🔵','Low':'🟢'}
            st.markdown(f"""
            <div class="result-wrap result-{label}">
                <div class="result-eyebrow">Match Priority Level</div>
                <div class="result-label color-{label}">{label}</div>
                <div class="confidence-pill">{icon_map[label]} Confidence &nbsp;<strong>{conf:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-head">Probabilitas</div>', unsafe_allow_html=True)
            for lbl, prob, color in [('High',ph,'var(--high)'),('Medium',pm,'var(--med)'),('Low',pl,'var(--low)')]:
                pct = prob*100
                st.markdown(f"""<div class="prob-item">
                    <div class="prob-header">
                        <span class="prob-name" style="color:{color}">{lbl}</span>
                        <span class="prob-pct" style="color:{color}">{pct:.1f}%</span>
                    </div>
                    <div class="prob-track"><div class="prob-fill" style="width:{pct}%;background:{color};"></div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-head">Historis Tim</div>', unsafe_allow_html=True)
            hc1, hc2 = st.columns(2)
            for col_h, team, p, w, n in [(hc1,selected_home,home_p,home_w,hn),(hc2,selected_away,away_p,away_w,an)]:
                bc = "badge-ok" if n>=3 else "badge-warn"
                bt = f"✓ {n} matches" if n>=3 else f"⚠ {n} matches"
                col_h.markdown(f"""<div class="hist-card">
                    <div class="hist-team">{team}</div>
                    <div class="hist-row"><span class="hist-key">Avg Plays</span><span class="hist-val">{fmt(p)}</span></div>
                    <div class="hist-row"><span class="hist-key">Avg Watchers</span><span class="hist-val">{fmt(w)}</span></div>
                    <span class="hist-badge {bc}">{bt}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-head">Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="chips">
                <span class="chip">🏆 {selected_tournament}</span>
                <span class="chip">🏠 {selected_home}</span>
                <span class="chip">✈️ {selected_away}</span>
                <span class="chip">📺 {selected_channel}</span>
                <span class="chip">📅 {match_date} {match_time_val.strftime('%H:%M')}</span>
                <span class="chip">⏱️ {duration} mnt</span>
                <span class="chip">{"🔒 Exclusive" if match_exclusive else "🔓 Open"}</span>
                <span class="chip">{"🔑 Login" if match_login_gating else "🚪 Free"}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="empty-state">
                <div class="empty-icon">⚽</div>
                <div class="empty-title">Siap Memprediksi</div>
                <div class="empty-sub">Isi form di kiri, lalu tekan<br>tombol Predict Priority</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: BULK UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
else:
    bl, br = st.columns([5, 4], gap="large")

    with bl:
        st.markdown('<div class="sec-head">Upload CSV</div>', unsafe_allow_html=True)

        # Info box
        st.markdown("""<div class="bulk-info">
            <div class="bulk-info-title">📋 Format CSV yang Dibutuhkan</div>
            <div class="bulk-info-text">
                Upload file CSV dengan kolom berikut (sama seperti format <code>matches_data.xlsx</code>).
                Kolom <code>match_priority_level</code> akan diisi otomatis oleh model.<br><br>
                <strong>Kolom wajib:</strong><br>
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
            label="⬇️ Download Template CSV",
            data=csv_template,
            file_name="template_bulk_predict.csv",
            mime="text/csv",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")

        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)

            # Validasi
            missing = [c for c in required_cols if c not in df_upload.columns]
            if missing:
                st.error(f"Kolom tidak ditemukan: {', '.join(missing)}")
            else:
                # Filter Football only dan punya team
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
                    <div class="stat-box"><div class="stat-num" style="color:var(--accent)">{n_valid}</div><div class="stat-lbl">Valid (Football)</div></div>
                    <div class="stat-box"><div class="stat-num" style="color:var(--text3)">{n_skipped}</div><div class="stat-lbl">Skipped</div></div>
                </div>""", unsafe_allow_html=True)

                run_bulk = st.button("⚡ JALANKAN BULK PREDICT")

                if run_bulk and n_valid > 0:
                    results = []
                    progress = st.progress(0, text="Memproses...")

                    for i, row in df_valid.iterrows():
                        # Parse tanggal
                        try:
                            match_dt = pd.to_datetime(row['match_date_start'])
                            if pd.isna(match_dt):
                                match_dt = datetime.datetime.now()
                        except:
                            match_dt = datetime.datetime.now()

                        # Parse durasi menit
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

                    # Gabung hasil ke df_valid
                    df_result = df_valid.copy()
                    res_df    = pd.DataFrame(results)
                    df_result['match_priority_level'] = res_df['match_priority_level'].values
                    df_result['prob_high']   = res_df['prob_high'].values
                    df_result['prob_medium'] = res_df['prob_medium'].values
                    df_result['prob_low']    = res_df['prob_low'].values

                    st.session_state['bulk_result'] = df_result
                    st.session_state['bulk_res_df'] = res_df
                    st.success(f"✅ {n_valid} match berhasil diprediksi!")

    with br:
        if 'bulk_result' in st.session_state:
            df_result = st.session_state['bulk_result']
            res_df    = st.session_state['bulk_res_df']

            # Summary stats
            counts = res_df['match_priority_level'].value_counts()
            n_h = counts.get('High',0)
            n_m = counts.get('Medium',0)
            n_l = counts.get('Low',0)

            st.markdown('<div class="sec-head">Hasil Prediksi</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="stat-row">
                <div class="stat-box"><div class="stat-num color-HIGH">{n_h}</div><div class="stat-lbl">High</div></div>
                <div class="stat-box"><div class="stat-num color-MEDIUM">{n_m}</div><div class="stat-lbl">Medium</div></div>
                <div class="stat-box"><div class="stat-num color-LOW">{n_l}</div><div class="stat-lbl">Low</div></div>
            </div>""", unsafe_allow_html=True)

            # Preview table
            st.markdown('<div class="sec-head">Preview</div>', unsafe_allow_html=True)
            show_cols = ['team_home','team_away','match_tournament','match_date_start',
                         'match_priority_level','prob_high','prob_medium','prob_low']
            show_cols = [c for c in show_cols if c in df_result.columns]
            df_show   = df_result[show_cols].head(20)

            # Render table
            rows_html = ""
            for _, r in df_show.iterrows():
                lbl = r.get('match_priority_level','—')
                badge = f'<span class="badge badge-{lbl}">{lbl}</span>' if lbl in ['High','Medium','Low'] else lbl
                dt_str = str(r.get('match_date_start',''))[:16] if pd.notna(r.get('match_date_start')) else '—'
                rows_html += f"""<tr>
                    <td>{r.get('team_home','—')}</td>
                    <td>{r.get('team_away','—')}</td>
                    <td style="font-size:0.72rem;color:var(--text2)">{r.get('match_tournament','—')[:25]}</td>
                    <td style="font-size:0.72rem">{dt_str}</td>
                    <td>{badge}</td>
                    <td style="color:var(--high)">{r.get('prob_high','—')}%</td>
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

            # Download hasil
            st.markdown("<br>", unsafe_allow_html=True)
            csv_out = df_result.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Hasil CSV",
                data=csv_out,
                file_name="bulk_predict_result.csv",
                mime="text/csv",
            )
        else:
            st.markdown("""<div class="empty-state">
                <div class="empty-icon">📂</div>
                <div class="empty-title">Upload CSV</div>
                <div class="empty-sub">Upload file CSV di sebelah kiri<br>lalu klik Jalankan Bulk Predict</div>
            </div>""", unsafe_allow_html=True)
