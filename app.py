import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
from datetime import date, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Manual Target Encoder ──────────────────────────────────────────────────────
class ManualTargetEncoder:
    def __init__(self, cols, smoothing=10):
        self.cols = cols
        self.smoothing = smoothing
        self.stats_ = {}
        self.globals_ = {}

    def fit(self, X, y):
        y_s = pd.Series(y, index=X.index)
        for col in self.cols:
            agg = pd.DataFrame({'y': y_s, 'col': X[col].values}).groupby('col')['y'].agg(['mean', 'count'])
            gm  = y_s.mean()
            agg['s'] = (agg['count'] * agg['mean'] + self.smoothing * gm) / (agg['count'] + self.smoothing)
            self.stats_[col] = agg['s'].to_dict()
            self.globals_[col] = gm
        return self

    def transform(self, X):
        Xo = X.copy()
        for col in self.cols:
            Xo[col] = Xo[col].map(self.stats_[col]).fillna(self.globals_[col])
        return Xo

# ── Manual Preprocessor (bebas versi sklearn) ──────────────────────────────────
class ManualPreprocessor:
    def __init__(self):
        self.numeric_log_cols   = None
        self.numeric_scale_cols = None
        self.binary_cols        = None
        self.ohe_cols           = None
        self.target_enc_cols    = None
        self.medians_           = {}
        self.scale_means_       = {}
        self.scale_stds_        = {}
        self.ohe_categories_    = {}

    def fit(self, X, numeric_log, numeric_scale, binary, ohe, target_enc):
        self.numeric_log_cols   = numeric_log
        self.numeric_scale_cols = numeric_scale
        self.binary_cols        = binary
        self.ohe_cols           = ohe
        self.target_enc_cols    = target_enc

        for col in numeric_log:
            med = float(np.nanmedian(X[col]))
            self.medians_[col] = med
            vals = np.log1p(np.where(np.isnan(X[col].astype(float)), med, X[col].astype(float)))
            self.scale_means_[col] = float(vals.mean())
            self.scale_stds_[col]  = float(vals.std())

        for col in numeric_scale:
            med = float(np.nanmedian(X[col]))
            self.medians_[col] = med
            vals = np.where(np.isnan(X[col].astype(float)), med, X[col].astype(float))
            self.scale_means_[col] = float(vals.mean())
            self.scale_stds_[col]  = float(vals.std())

        for col in ohe:
            self.ohe_categories_[col] = sorted(X[col].dropna().unique().tolist())

        return self

    def transform(self, X):
        parts = []

        for col in self.numeric_log_cols:
            med  = self.medians_[col]
            vals = np.where(pd.isna(X[col]), med, X[col].astype(float))
            vals = np.log1p(vals)
            vals = (vals - self.scale_means_[col]) / (self.scale_stds_[col] + 1e-8)
            parts.append(vals.reshape(-1, 1))

        for col in self.numeric_scale_cols:
            med  = self.medians_[col]
            vals = np.where(pd.isna(X[col]), med, X[col].astype(float))
            vals = (vals - self.scale_means_[col]) / (self.scale_stds_[col] + 1e-8)
            parts.append(vals.reshape(-1, 1))

        for col in self.binary_cols:
            vals = X[col].fillna(0).astype(float).values
            parts.append(vals.reshape(-1, 1))

        for col in self.ohe_cols:
            for cat in self.ohe_categories_[col]:
                vals = (X[col] == cat).astype(float).values
                parts.append(vals.reshape(-1, 1))

        for col in self.target_enc_cols:
            vals = X[col].astype(float).values
            parts.append(vals.reshape(-1, 1))

        return np.hstack(parts)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Match Priority Predictor",
    page_icon="⚽",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f1a2e 50%, #0a0a0f 100%);
    color: #e8e8f0;
}

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }

.title-block { text-align: center; padding: 2rem 0 1rem 0; }
.title-block h1 { font-size: 3.5rem; color: #ffffff; margin: 0; line-height: 1; }
.title-block p { color: #8888aa; font-size: 0.95rem; margin-top: 0.5rem; font-weight: 300; }
.accent { color: #4af0a0; }

.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 3px;
    color: #4af0a0;
    border-bottom: 1px solid #1e2a3a;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
    text-transform: uppercase;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stTimeInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label {
    color: #aaaacc !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

.auto-fill-box {
    background: #0d1f35;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.auto-fill-box .label { font-size: 0.72rem; color: #6677aa; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
.auto-fill-box .value { font-size: 0.92rem; color: #c8d8f0; margin-top: 0.15rem; font-weight: 400; }

.result-card { border-radius: 12px; padding: 2rem; text-align: center; margin-top: 1rem; }
.result-HIGH   { background: linear-gradient(135deg, #1a0a0a, #2d1010); border: 2px solid #ff4444; }
.result-MEDIUM { background: linear-gradient(135deg, #0a1220, #0d2040); border: 2px solid #4488ff; }
.result-LOW    { background: linear-gradient(135deg, #0a1a10, #0d2a15); border: 2px solid #44cc88; }

.result-label { font-family: 'Bebas Neue', sans-serif; font-size: 1rem; letter-spacing: 4px; color: #8888aa; }
.result-value { font-family: 'Bebas Neue', sans-serif; font-size: 4rem; letter-spacing: 4px; line-height: 1.1; margin: 0.3rem 0; }

.color-HIGH   { color: #ff4444; }
.color-MEDIUM { color: #4488ff; }
.color-LOW    { color: #44cc88; }

.prob-bar-wrap { margin: 0.4rem 0; }
.prob-label-row { display: flex; justify-content: space-between; font-size: 0.8rem; color: #8888aa; margin-bottom: 3px; }

.divider { border: none; border-top: 1px solid #1e2a3a; margin: 1.5rem 0; }

.stButton > button {
    background: linear-gradient(135deg, #4af0a0, #00c8ff) !important;
    color: #0a0a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
}

.info-chip {
    display: inline-block;
    background: #1e2a3a;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #8888aa;
    margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model     = joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl'))
    prep      = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocessor.pkl'))
    le        = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))
    tenc      = joblib.load(os.path.join(BASE_DIR, 'models', 'target_encoder.pkl'))
    threshold = joblib.load(os.path.join(BASE_DIR, 'models', 'threshold.pkl'))
    return model, prep, le, tenc, threshold

@st.cache_data
def load_reference():
    xl         = pd.read_excel(os.path.join(BASE_DIR, 'data', 'reference_data.xlsx'), sheet_name=None)
    team_stats = pd.read_csv(os.path.join(BASE_DIR, 'src', 'data', 'team_stats.csv'))
    return xl['tournaments'], xl['leagues'], xl['teams'], team_stats

# ── Helper Functions ───────────────────────────────────────────────────────────
def get_teams_for_tournament(tournament_row, leagues_df, teams_df):
    raw          = str(tournament_row['tournament_league'])
    league_names = [x.strip() for x in raw.split(',') if x.strip()]
    matched_ids  = leagues_df[leagues_df['league_name'].isin(league_names)]['league_id'].tolist()
    matched_teams = teams_df[teams_df['team_league'].isin(matched_ids)]['team_name'].sort_values().tolist()
    return matched_teams

def get_team_hist(team_name, team_stats_df):
    row = team_stats_df[team_stats_df['team_name'] == team_name]
    if len(row) == 0:
        return None
    return row.iloc[0]

def apply_threshold(proba, th_high, th_low, idx_high, idx_low, idx_med):
    preds = []
    for p in proba:
        if p[idx_high] >= th_high:
            preds.append(idx_high)
        elif p[idx_low] >= th_low:
            preds.append(idx_low)
        else:
            preds.append(idx_med)
    return np.array(preds)

# ── Load ───────────────────────────────────────────────────────────────────────
try:
    model, prep, le, tenc, threshold = load_models()
    tournaments_df, leagues_df, teams_df, team_stats_df = load_reference()

    classes  = list(le.classes_)
    idx_high = classes.index('High')
    idx_low  = classes.index('Low')
    idx_med  = classes.index('Medium')

    global_avg_plays    = team_stats_df['avg_plays'].median()
    global_avg_watchers = team_stats_df['avg_watchers'].median()

except Exception as e:
    st.error(f"Gagal load model atau data: {e}")
    st.stop()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>⚽ MATCH PRIORITY <span class="accent">PREDICTOR</span></h1>
    <p>Football · BUSPRO Sport Calendar · Powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

col_form, col_gap, col_result = st.columns([5, 0.5, 4])

with col_form:

    # ── 01: Tournament ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">01 · Tournament</div>', unsafe_allow_html=True)

    tournament_list     = sorted(tournaments_df['tournament_title'].dropna().tolist())
    selected_tournament = st.selectbox("Pilih Tournament", tournament_list, index=0)
    t_row = tournaments_df[tournaments_df['tournament_title'] == selected_tournament].iloc[0]

    channels_raw    = str(t_row['tournament_channel']) if pd.notna(t_row['tournament_channel']) else ''
    channel_options = [x.strip() for x in channels_raw.split(',') if x.strip()]

    # ── 02: Auto-fill ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">02 · Match Details (Auto-filled)</div>', unsafe_allow_html=True)

    af1, af2, af3, af4 = st.columns(4)
    with af1:
        st.markdown(f"""<div class="auto-fill-box">
            <div class="label">Premier Status</div>
            <div class="value">{t_row['tournament_premier'] if pd.notna(t_row['tournament_premier']) else '-'}</div>
        </div>""", unsafe_allow_html=True)
    with af2:
        st.markdown(f"""<div class="auto-fill-box">
            <div class="label">Coverage</div>
            <div class="value">{t_row['tournament_coverage'] if pd.notna(t_row['tournament_coverage']) else '-'}</div>
        </div>""", unsafe_allow_html=True)
    with af3:
        st.markdown(f"""<div class="auto-fill-box">
            <div class="label">Gender</div>
            <div class="value">{t_row['tournament_gender'] if pd.notna(t_row['tournament_gender']) else '-'}</div>
        </div>""", unsafe_allow_html=True)
    with af4:
        st.markdown(f"""<div class="auto-fill-box">
            <div class="label">Organization</div>
            <div class="value" style="font-size:0.78rem">{t_row['tournament_organization'] if pd.notna(t_row['tournament_organization']) else '-'}</div>
        </div>""", unsafe_allow_html=True)

    selected_channel = st.selectbox(
        "Channel Siaran",
        options=channel_options if channel_options else ['Unknown'],
    )

    # ── 03: Teams ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">03 · Teams</div>', unsafe_allow_html=True)

    available_teams = get_teams_for_tournament(t_row, leagues_df, teams_df)

    if not available_teams:
        st.warning("Tidak ada tim yang terdaftar untuk tournament ini.")
        st.stop()

    tc1, tc2 = st.columns(2)
    with tc1:
        selected_home = st.selectbox("Team Home", available_teams, index=0)
    with tc2:
        away_options  = [t for t in available_teams if t != selected_home]
        selected_away = st.selectbox("Team Away", away_options, index=0)

    # ── 04: Schedule ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">04 · Schedule & Duration</div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        match_date = st.date_input("Tanggal Pertandingan", value=date.today())
    with sc2:
        match_time = st.time_input("Jam Kick-off", value=time(20, 0))
    with sc3:
        duration = st.number_input("Durasi (menit)", min_value=30, max_value=600, value=120, step=15)

    # ── 05: Access Flags ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">05 · Access Flags</div>', unsafe_allow_html=True)

    fl1, fl2, fl3 = st.columns(3)
    with fl1:
        match_exclusive    = st.checkbox("Exclusive", value=False)
    with fl2:
        match_login_gating = st.checkbox("Login Gating", value=False)
    with fl3:
        match_drm          = st.checkbox("DRM", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ PREDICT PRIORITY")

# ── Result ─────────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-header">Result</div>', unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("Menghitung prediksi..."):

            home_stats = get_team_hist(selected_home, team_stats_df)
            away_stats = get_team_hist(selected_away, team_stats_df)

            home_avg_plays    = home_stats['avg_plays']    if home_stats is not None else global_avg_plays
            away_avg_plays    = away_stats['avg_plays']    if away_stats is not None else global_avg_plays
            home_avg_watchers = home_stats['avg_watchers'] if home_stats is not None else global_avg_watchers
            away_avg_watchers = away_stats['avg_watchers'] if away_stats is not None else global_avg_watchers
            home_n_past       = home_stats['n_matches']    if home_stats is not None else 0
            away_n_past       = away_stats['n_matches']    if away_stats is not None else 0

            hist_avg_plays    = (home_avg_plays + away_avg_plays) / 2
            hist_max_plays    = max(home_avg_plays, away_avg_plays)
            hist_avg_watchers = (home_avg_watchers + away_avg_watchers) / 2
            is_reliable       = int(home_n_past >= 3 and away_n_past >= 3)

            premier_status = str(t_row['tournament_premier'])      if pd.notna(t_row['tournament_premier'])      else 'FREE'
            coverage       = str(t_row['tournament_coverage'])     if pd.notna(t_row['tournament_coverage'])     else 'INDONESIA'
            gender         = str(t_row['tournament_gender'])       if pd.notna(t_row['tournament_gender'])       else 'Men'
            organization   = str(t_row['tournament_organization']) if pd.notna(t_row['tournament_organization']) else 'Unknown'

            match_dt = datetime.datetime.combine(match_date, match_time)
            hour     = match_dt.hour
            month    = match_dt.month

            row_data = {
                'home_hist_avg_plays':     home_avg_plays,
                'away_hist_avg_plays':     away_avg_plays,
                'match_hist_avg_plays':    hist_avg_plays,
                'match_hist_max_plays':    hist_max_plays,
                'home_hist_avg_watchers':  home_avg_watchers,
                'away_hist_avg_watchers':  away_avg_watchers,
                'match_hist_avg_watchers': hist_avg_watchers,
                'home_n_past':             home_n_past,
                'away_n_past':             away_n_past,
                'is_reliable':             is_reliable,
                'hour':                    hour,
                'month':                   month,
                'duration_minutes':        duration,
                'match_exclusive':         int(match_exclusive),
                'match_login_gating':      int(match_login_gating),
                'match_drm':               int(match_drm),
                'match_gender':            gender,
                'match_coverage':          coverage,
                'match_premier_status':    premier_status,
                'match_tournament':        selected_tournament,
                'match_channel':           selected_channel,
                'match_organization':      organization,
            }

            input_df   = pd.DataFrame([row_data])
            TARGET_ENC = ['match_premier_status', 'match_tournament', 'match_channel', 'match_organization']
            input_df[TARGET_ENC] = tenc.transform(input_df[TARGET_ENC])

            X_input    = prep.transform(input_df)
            proba      = model.predict_proba(X_input)
            th_high    = threshold['th_high']
            th_low     = threshold['th_low']
            pred_idx   = apply_threshold(proba, th_high, th_low, idx_high, idx_low, idx_med)[0]
            pred_label = le.inverse_transform([pred_idx])[0]

            prob_high = proba[0][idx_high]
            prob_low  = proba[0][idx_low]
            prob_med  = proba[0][idx_med]

        icon_map = {'High': '🔴', 'Medium': '🔵', 'Low': '🟢'}

        st.markdown(f"""
        <div class="result-card result-{pred_label}">
            <div class="result-label">MATCH PRIORITY</div>
            <div class="result-value color-{pred_label}">{pred_label.upper()}</div>
            <div style="color:#8888aa; font-size:0.85rem;">
                {icon_map[pred_label]} Confidence: {max(prob_high, prob_low, prob_med)*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Probabilitas per Kelas**")

        for label, prob, color in [
            ('High',   prob_high, '#ff4444'),
            ('Medium', prob_med,  '#4488ff'),
            ('Low',    prob_low,  '#44cc88'),
        ]:
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-label-row">
                    <span style="color:{color}; font-weight:600">{label}</span>
                    <span>{prob*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Input Summary**")
        st.markdown(f"""
        <span class="info-chip">🏆 {selected_tournament}</span>
        <span class="info-chip">🏠 {selected_home}</span>
        <span class="info-chip">✈️ {selected_away}</span>
        <span class="info-chip">📺 {selected_channel}</span>
        <span class="info-chip">📅 {match_date} {match_time.strftime('%H:%M')}</span>
        <span class="info-chip">⏱️ {duration} mnt</span>
        <span class="info-chip">{'🔒 Exclusive' if match_exclusive else '🔓 Non-Exclusive'}</span>
        <span class="info-chip">{'🔑 Login Required' if match_login_gating else '🔓 No Login'}</span>
        <span class="info-chip">Threshold High≥{th_high} Low≥{th_low}</span>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 1rem; color: #444466;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚽</div>
            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.2rem; letter-spacing: 3px;">
                ISI FORM & KLIK PREDICT
            </div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem; color: #333355;">
                Hasil prediksi akan muncul di sini
            </div>
        </div>
        """, unsafe_allow_html=True)
