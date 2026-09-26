"""
Ozone Depletion Severity Predictor — Streamlit app.

Reproduces the layout/flow of the UI mockup (Predict / Model Insights / About
tabs, sliders, gauge, class-probability bars, feature contributions, what-if
simulator, and the 3 accent-color theme buttons) and wires it to the real
Softmax Regression model trained by train_model.py.

Behavior notes:
- Moving a slider/select does NOT recompute the prediction. The result panel
  only updates when the "Predict Severity" button is clicked — the button
  sits at the top of the results panel, above the gauge.
- The 3 theme buttons (Ocean / Aurora / Sunset) swap the accent colors used
  throughout the page.

Run `python train_model.py` once before `streamlit run app.py`.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Ozone Depletion Severity Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# UI Theme
# --------------------------------------------------------------------------
THEME_ACCENTS = {
    "ocean": ("#6ea8ff", "#a78bfa"),
    "aurora": ("#34d399", "#22d3ee"),
    "sunset": ("#fb923c", "#f472b6"),
}
THEME_LABELS = {"ocean": "🌊 Ocean", "aurora": "🌌 Aurora", "sunset": "🌅 Sunset"}

# Theme-specific semantic colors used by the prediction area as well.
# Each palette still reads Low → Moderate → Severe clearly, but its exact
# treatment changes with the selected visual theme.
THEME_STATUS = {
    "ocean": {"low": "#34d399", "mod": "#fbbf24", "sev": "#fb7185"},
    "aurora": {"low": "#22c55e", "mod": "#38bdf8", "sev": "#c084fc"},
    "sunset": {"low": "#a3e635", "mod": "#f59e0b", "sev": "#f43f5e"},
}
UI_MODES = {
    "dark": {"bg":"#0b1120","bg2":"#10182b","panel":"#141b2d","panel2":"#1a2338","card":"rgba(20,27,45,.82)","border":"#2a3550","border_soft":"#34415f","text":"#e6ebf5","muted":"#9aabc9","input":"#111a2d","input_hover":"#17233b","surface":"#10182a","table_header":"#1b2740","shadow":"rgba(0,0,0,.30)"},
    "light": {"bg":"#f5f7fb","bg2":"#eaf0f8","panel":"#ffffff","panel2":"#eef3f9","card":"rgba(255,255,255,.94)","border":"#d6deea","border_soft":"#c3cedd","text":"#172033","muted":"#66758d","input":"#ffffff","input_hover":"#f1f5fa","surface":"#ffffff","table_header":"#edf2f7","shadow":"rgba(25,42,70,.10)"},
}
if "theme_color" not in st.session_state: st.session_state.theme_color = "ocean"
if "ui_mode" not in st.session_state: st.session_state.ui_mode = "dark"
UI = UI_MODES[st.session_state.ui_mode]
STATUS = THEME_STATUS[st.session_state.theme_color]
COLORS = {**UI, **STATUS}
COLORS["accent"], COLORS["accent2"] = THEME_ACCENTS[st.session_state.theme_color]

# Prediction visuals use the active theme's Low/Moderate/Severe palette.
CLASS_COLOR = {"Low": COLORS["low"], "Moderate": COLORS["mod"], "Severe": COLORS["sev"]}
if st.session_state.ui_mode == "dark":
    CLASS_FG = {"Low": "#06271b", "Moderate": "#241800", "Severe": "#2b0710"}
else:
    CLASS_FG = {"Low": "#ffffff", "Moderate": "#251800", "Severe": "#ffffff"}
CLASS_BG_FG = {cls: (CLASS_COLOR[cls], CLASS_FG[cls]) for cls in CLASS_COLOR}

# Full UI skin: all native Streamlit surfaces/widgets use the same mode + accent.
st.markdown(f"""
<style>
:root {{ --bg:{COLORS['bg']}; --bg2:{COLORS['bg2']}; --panel:{COLORS['panel']}; --panel2:{COLORS['panel2']}; --card:{COLORS['card']}; --border:{COLORS['border']}; --border-soft:{COLORS['border_soft']}; --text:{COLORS['text']}; --muted:{COLORS['muted']}; --input:{COLORS['input']}; --input-hover:{COLORS['input_hover']}; --accent:{COLORS['accent']}; --accent2:{COLORS['accent2']}; --low:{COLORS['low']}; --mod:{COLORS['mod']}; --sev:{COLORS['sev']}; --shadow:{COLORS['shadow']}; }}
.stApp {{ background: radial-gradient(1100px 560px at 12% -8%, color-mix(in srgb,var(--accent) 14%,transparent),transparent 62%), radial-gradient(900px 520px at 100% 8%,color-mix(in srgb,var(--accent2) 11%,transparent),transparent 62%), linear-gradient(180deg,var(--bg2),var(--bg) 48%,var(--bg)); color:var(--text); }}
.stAppViewContainer,[data-testid="stAppViewContainer"],.main {{ background:transparent!important; color:var(--text)!important; }}
[data-testid="stHeader"] {{ background:color-mix(in srgb,var(--bg) 88%,transparent)!important; border-bottom:1px solid var(--border); }}
[data-testid="stSidebar"] {{ background:var(--panel)!important; border-right:1px solid var(--border); }}
h1,h2,h3,h4,h5,h6,[data-testid="stMarkdownContainer"] {{ color:var(--text); }}
.muted,.hint,.tl-text {{ color:var(--muted)!important; }}
.section-label {{ color:var(--muted);font-size:11px;text-transform:uppercase;margin:14px 0 8px;letter-spacing:.5px;font-weight:700; }}
.card {{ background:var(--card);border:1px solid var(--border);border-radius:16px;padding:18px;margin-bottom:16px;box-shadow:0 8px 28px var(--shadow);backdrop-filter:blur(12px); }}
.card h3 {{ color:var(--accent); }}
.card-severe {{ border-color:{COLORS['sev']}!important;box-shadow:0 0 0 1px {COLORS['sev']},0 8px 30px color-mix(in srgb,{COLORS['sev']} 24%,transparent); }}
.narrative {{ font-size:13px;line-height:1.6;padding:10px 12px;border-radius:9px;background:var(--panel2);border:1px solid var(--border);border-left:3px solid var(--accent);margin-top:10px;color:var(--text); }}
.tl-item {{ border-left:2px solid var(--border);padding:0 0 16px 16px;margin-left:4px;position:relative; }} .tl-year {{ color:var(--accent2);font-weight:700;font-size:13px; }}
button[data-baseweb="tab"] {{ color:var(--muted)!important;background:transparent!important;font-weight:650;border-radius:9px 9px 0 0; }}
button[data-baseweb="tab"]:hover {{ color:var(--text)!important;background:color-mix(in srgb,var(--accent) 8%,transparent)!important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:var(--accent)!important; }}
div[data-baseweb="tab-highlight"] {{ background:linear-gradient(90deg,var(--accent),var(--accent2))!important; }}
div[data-baseweb="tab-border"] {{ background:var(--border)!important; }}
div.stButton>button,div.stDownloadButton>button {{ min-height:42px;border:1px solid color-mix(in srgb,var(--accent) 45%,var(--border))!important;border-radius:10px!important;background:linear-gradient(100deg,var(--accent),var(--accent2))!important;color:#08111f!important;font-weight:750!important;box-shadow:0 5px 18px color-mix(in srgb,var(--accent) 18%,transparent); }}
div.stButton>button:hover,div.stDownloadButton>button:hover {{ filter:brightness(1.06);transform:translateY(-1px);box-shadow:0 8px 24px color-mix(in srgb,var(--accent) 26%,transparent); }}
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,[data-testid="stSlider"] label,[data-testid="stSelectbox"] label {{ color:var(--text)!important; }}
div[data-baseweb="select"]>div {{ background:var(--input)!important;color:var(--text)!important;border-color:var(--border)!important;border-radius:9px!important; }}
div[data-baseweb="select"] span, div[data-baseweb="select"] svg {{ color:var(--text)!important; fill:var(--text)!important; }}
div[data-baseweb="select"]:focus-within > div {{ border-color:var(--accent)!important; box-shadow:0 0 0 1px var(--accent)!important; }}
div[data-baseweb="select"]>div:hover {{ background:var(--input-hover)!important;border-color:var(--border-soft)!important; }}
div[data-baseweb="popover"],ul[role="listbox"] {{ background:var(--panel)!important;border:1px solid var(--border)!important;box-shadow:0 12px 35px var(--shadow)!important; }}
li[role="option"] {{ color:var(--text)!important;background:var(--panel)!important; }} li[role="option"]:hover {{ background:var(--panel2)!important; }}
input,textarea {{ background:var(--input)!important;color:var(--text)!important;border-color:var(--border)!important; }}
input:focus,textarea:focus {{ border-color:var(--accent)!important;box-shadow:0 0 0 1px var(--accent)!important; }}
/* Slider rail + active track + thumb all follow the selected accent theme. */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background:var(--accent)!important;
    border:2px solid var(--accent2)!important;
    box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent)!important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
    background:var(--border)!important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {{
    background:linear-gradient(90deg,var(--accent),var(--accent2))!important;
}}
[data-testid="stSlider"] [data-testid="stTickBar"] span,
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {{
    color:var(--muted)!important;
}}
[data-testid="stSlider"] input[type="range"]::-webkit-slider-thumb {{
    background:var(--accent)!important;
    border:2px solid var(--accent2)!important;
}}
[data-testid="stSlider"] input[type="range"]::-moz-range-thumb {{
    background:var(--accent)!important;
    border:2px solid var(--accent2)!important;
}}
/* Number/value readouts used by Streamlit slider controls. */
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] [data-testid="stSliderValue"] {{
    color:var(--accent)!important;
}}
[data-testid="stDataFrame"] {{ border:1px solid var(--border)!important;border-radius:10px!important;overflow:hidden;background:var(--panel)!important; }}
/* Plotly iframe/container stays visually tied to the current theme. */
[data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] > div {{
    background:transparent!important;
}}
div[data-testid="stMetricValue"],div[data-testid="stMetricLabel"] {{ color:var(--text)!important; }}
[data-testid="stAlert"],details[data-testid="stExpander"] {{ background:var(--panel)!important;color:var(--text)!important;border:1px solid var(--border)!important; }}
details[data-testid="stExpander"] summary {{ color:var(--text)!important; }}
a {{ color:var(--accent)!important; }}
::-webkit-scrollbar {{ width:9px;height:9px; }} ::-webkit-scrollbar-track {{ background:var(--bg); }} ::-webkit-scrollbar-thumb {{ background:var(--border-soft);border-radius:10px; }} ::-webkit-scrollbar-thumb:hover {{ background:var(--accent); }}
</style>
""", unsafe_allow_html=True)

NUM_COLS = [
    "Stratospheric Temperature (Celsius)",
    "Chlorine Monoxide Concentration (Parts Per Billion)",
    "Bromine Monoxide Concentration (Parts Per Trillion)",
    "CFC-11 Concentration (Parts Per Trillion)",
    "CFC-12 Concentration (Parts Per Trillion)",
    "Nitrous Oxide Concentration (Parts Per Billion)",
    "UV Index",
    "Polar Stratospheric Cloud Index",
    "Wind Speed (Kilometers Per Hour)",
    "Latitude (Degrees)",
    "Methane Concentration (Parts Per Billion)",
]
CAT_COLS = ["Season", "Hemisphere", "Region Type", "Monitoring Station Type"]
LABEL_MAP = {1: "Low", 2: "Moderate", 3: "Severe"}

DEFAULT_RAW = {
    "Stratospheric Temperature (Celsius)": -54.0,
    "Chlorine Monoxide Concentration (Parts Per Billion)": 0.62,
    "Bromine Monoxide Concentration (Parts Per Trillion)": 20.0,
    "CFC-11 Concentration (Parts Per Trillion)": 230,
    "CFC-12 Concentration (Parts Per Trillion)": 500,
    "Nitrous Oxide Concentration (Parts Per Billion)": 320,
    "UV Index": 6.5,
    "Polar Stratospheric Cloud Index": 0.35,
    "Wind Speed (Kilometers Per Hour)": 40,
    "Latitude (Degrees)": -45,
    "Methane Concentration (Parts Per Billion)": 400,
    "Season": "Summer",
    "Hemisphere": "Southern Hemisphere",
    "Region Type": "Polar",
    "Monitoring Station Type": "Satellite Station",
}


# --------------------------------------------------------------------------
# Load trained artifacts (cached so this only runs once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/softmax_model.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
    encoders = joblib.load("artifacts/label_encoders.joblib")
    feature_names = joblib.load("artifacts/feature_names.joblib")
    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)
    with open("artifacts/feature_importance.json") as f:
        importance = json.load(f)
    return model, scaler, encoders, feature_names, metrics, importance


try:
    model, scaler, encoders, feature_names, metrics, importance = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `python train_model.py` once in this "
        "folder before starting the app."
    )
    st.stop()


def build_feature_row(raw: dict) -> pd.DataFrame:
    """Turn the raw slider/select values into the exact scaled feature row
    the model was trained on (same order, same encoders, same scaler)."""
    row = pd.DataFrame([raw])
    for col in CAT_COLS:
        row[col + "_Encoded"] = encoders[col].transform(row[col])
    row = row.drop(columns=CAT_COLS)
    row = row[feature_names]
    row[NUM_COLS] = scaler.transform(row[NUM_COLS])
    return row


def predict(raw: dict):
    row = build_feature_row(raw)
    proba = model.predict_proba(row)[0]  # order follows model.classes_
    class_order = list(model.classes_)  # e.g. [1, 2, 3]
    probs = {LABEL_MAP[c]: p for c, p in zip(class_order, proba)}
    pred_class = class_order[int(np.argmax(proba))]
    label = LABEL_MAP[pred_class]
    confidence = probs[label]
    return label, confidence, probs, row, pred_class


# Seed an initial prediction (from default values) once per session, so the
# results panel isn't empty before the user ever clicks Predict.
if "result" not in st.session_state:
    st.session_state.result = predict(DEFAULT_RAW)
    st.session_state.raw_used = dict(DEFAULT_RAW)

# --------------------------------------------------------------------------
# Header + theme controls
# --------------------------------------------------------------------------
st.markdown("## 🌍 Ozone Depletion Severity Predictor")
st.markdown("<p class='muted'>Softmax Regression classifier · Low / Moderate / Severe · trained on 1,000 station readings</p>", unsafe_allow_html=True)

theme_col1, theme_col2, theme_col3, mode_col, theme_spacer = st.columns([1,1,1,1.35,4.65])
for col, theme_key in zip((theme_col1,theme_col2,theme_col3), THEME_ACCENTS.keys()):
    with col:
        active = st.session_state.theme_color == theme_key
        if st.button(THEME_LABELS[theme_key] + (" ✓" if active else ""), key=f"theme_{theme_key}", use_container_width=True):
            st.session_state.theme_color = theme_key
            st.rerun()
with mode_col:
    mode_label = "☀️ Light mode" if st.session_state.ui_mode == "dark" else "🌙 Dark mode"
    if st.button(mode_label, key="toggle_ui_mode", use_container_width=True):
        st.session_state.ui_mode = "light" if st.session_state.ui_mode == "dark" else "dark"
        st.rerun()

tab_predict, tab_insights, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model Insights", "ℹ️ About"]
)

# ==========================================================================
# TAB 1 — PREDICT
# ==========================================================================
with tab_predict:
    col_inputs, col_results = st.columns([1.1, 1], gap="large")

    # ---- Inputs (moving these does NOT recompute the prediction) ----
    with col_inputs:
        st.markdown("<div class='card'><h3>Atmospheric Inputs</h3>", unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Chemistry</div>", unsafe_allow_html=True)
        clo = st.slider("Chlorine Monoxide (ppb)", 0.10, 1.30, DEFAULT_RAW["Chlorine Monoxide Concentration (Parts Per Billion)"], 0.01)
        cfc11 = st.slider("CFC-11 (ppt)", 180, 280, DEFAULT_RAW["CFC-11 Concentration (Parts Per Trillion)"], 1)
        cfc12 = st.slider("CFC-12 (ppt)", 450, 580, DEFAULT_RAW["CFC-12 Concentration (Parts Per Trillion)"], 1)
        bro = st.slider("Bromine Monoxide (ppt)", 10.0, 30.0, DEFAULT_RAW["Bromine Monoxide Concentration (Parts Per Trillion)"], 0.1)
        n2o = st.slider("Nitrous Oxide (ppb)", 300, 340, DEFAULT_RAW["Nitrous Oxide Concentration (Parts Per Billion)"], 1)
        ch4 = st.slider("Methane (ppb)", 50, 800, DEFAULT_RAW["Methane Concentration (Parts Per Billion)"], 1)

        st.markdown("<div class='section-label'>Physical Conditions</div>", unsafe_allow_html=True)
        temp = st.slider("Stratospheric Temp (°C)", -70.0, -45.0, DEFAULT_RAW["Stratospheric Temperature (Celsius)"], 0.1)
        psc = st.slider("Polar Stratospheric Cloud Index", 0.05, 0.90, DEFAULT_RAW["Polar Stratospheric Cloud Index"], 0.01)
        uv = st.slider("UV Index", 2.0, 10.0, DEFAULT_RAW["UV Index"], 0.1)
        wind = st.slider("Wind Speed (km/h)", 5, 70, DEFAULT_RAW["Wind Speed (Kilometers Per Hour)"], 1)
        lat = st.slider("Latitude (°)", -90, 90, DEFAULT_RAW["Latitude (Degrees)"], 1)

        st.markdown("<div class='section-label'>Location &amp; Station Context</div>", unsafe_allow_html=True)
        season = st.selectbox("Season", ["Summer", "Winter", "Autumn", "Spring"])
        hemi = st.selectbox("Hemisphere", ["Northern Hemisphere", "Southern Hemisphere"], index=1)
        region = st.selectbox("Region Type", ["Rural", "Polar", "Industrial", "Urban"], index=1)
        station = st.selectbox(
            "Monitoring Station Type",
            ["Satellite Station", "Ground Based Station", "Weather Balloon Station"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Current (live) values from the widgets — NOT applied until Predict is clicked
    raw = {
        "Stratospheric Temperature (Celsius)": temp,
        "Chlorine Monoxide Concentration (Parts Per Billion)": clo,
        "Bromine Monoxide Concentration (Parts Per Trillion)": bro,
        "CFC-11 Concentration (Parts Per Trillion)": cfc11,
        "CFC-12 Concentration (Parts Per Trillion)": cfc12,
        "Nitrous Oxide Concentration (Parts Per Billion)": n2o,
        "UV Index": uv,
        "Polar Stratospheric Cloud Index": psc,
        "Wind Speed (Kilometers Per Hour)": wind,
        "Latitude (Degrees)": lat,
        "Methane Concentration (Parts Per Billion)": ch4,
        "Season": season,
        "Hemisphere": hemi,
        "Region Type": region,
        "Monitoring Station Type": station,
    }

    # ---- Results (only refreshed when the button below is clicked) ----
    with col_results:
        st.markdown("<div class='hint'>Adjust inputs on the left, then click below to update the prediction.</div>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡ Predict Severity", type="primary", use_container_width=True)
        if predict_clicked:
            st.session_state.result = predict(raw)
            st.session_state.raw_used = dict(raw)

        label, confidence, probs, scaled_row, pred_class = st.session_state.result
        raw_used = st.session_state.raw_used
        top_pct = confidence * 100
        bg, fg = CLASS_BG_FG[label]
        card_class = "card card-severe" if label == "Severe" else "card"

        st.markdown(f"<div class='{card_class}'><h3>Prediction Result</h3>", unsafe_allow_html=True)

        gc1, gc2 = st.columns(2)
        with gc1:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=top_pct,
                    number={"suffix": "%", "font": {"color": COLORS["text"], "size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                        "bar": {"color": CLASS_COLOR[label]},
                        "bgcolor": COLORS["panel2"],
                        "borderwidth": 0,
                    },
                )
            )
            fig.update_layout(
                height=180,
                margin=dict(l=10, r=10, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["text"]},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"<div style='text-align:center'>Confidence — "
                f"<span class='badge' style='background:{bg};color:{fg}'>{label.upper()}</span></div>",
                unsafe_allow_html=True,
            )
        with gc2:
            hole_r = 22 + (probs["Severe"]) * 55
            svg = f"""
            <svg width="140" height="140" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="55" fill="none" stroke="{COLORS['border']}" stroke-width="2"/>
              <circle cx="60" cy="60" r="{hole_r:.0f}" fill="{CLASS_COLOR[label]}" opacity="0.35"/>
              <circle cx="60" cy="60" r="{hole_r:.0f}" fill="none" stroke="{CLASS_COLOR[label]}" stroke-width="2"/>
              <text x="60" y="112" text-anchor="middle" font-size="9" fill="{COLORS['muted']}">Ozone hole (relative size)</text>
            </svg>
            """
            st.markdown(f"<div style='text-align:center'>{svg}</div>", unsafe_allow_html=True)

        verdict = (
            "requires immediate attention"
            if label == "Severe"
            else "is within a stable, low-risk range"
            if label == "Low"
            else "shows early warning signs worth monitoring"
        )
        st.markdown(
            f"<div class='narrative'>For a {raw_used['Region Type'].lower()} {raw_used['Hemisphere'].lower()} "
            f"station in {raw_used['Season']}, current atmospheric readings point to {label.lower()} depletion "
            f"severity ({top_pct:.0f}% confidence) — this {verdict}.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-label'>Class Probabilities</div>", unsafe_allow_html=True)
        for cls in ["Low", "Moderate", "Severe"]:
            pct = probs[cls] * 100
            st.markdown(
                f"""
                <div style='display:flex;align-items:center;gap:8px;font-size:12px;margin-bottom:6px'>
                  <div style='width:70px;color:{COLORS['muted']}'>{cls}</div>
                  <div style='flex:1;background:{COLORS['panel2']};border-radius:6px;height:14px;overflow:hidden'>
                    <div style='height:100%;border-radius:6px;width:{pct:.0f}%;background:{CLASS_COLOR[cls]}'></div>
                  </div>
                  <div style='width:42px;text-align:right;font-weight:700'>{pct:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Feature contribution for this prediction: coef (for predicted class)
        # x scaled feature value — how much each feature pushed the winning class.
        st.markdown("<div class='section-label'>Feature Contribution (this prediction)</div>", unsafe_allow_html=True)
        class_idx = list(model.classes_).index(pred_class)
        coefs = model.coef_[class_idx]
        contrib = coefs * scaled_row.iloc[0].values
        top_feats = ["Chlorine Monoxide Concentration (Parts Per Billion)", "Polar Stratospheric Cloud Index",
                     "Stratospheric Temperature (Celsius)", "CFC-12 Concentration (Parts Per Trillion)"]
        short_names = {
            "Chlorine Monoxide Concentration (Parts Per Billion)": "Chlorine Monoxide",
            "Polar Stratospheric Cloud Index": "PSC Index",
            "Stratospheric Temperature (Celsius)": "Strat. Temp",
            "CFC-12 Concentration (Parts Per Trillion)": "CFC-12",
        }
        max_abs = max(abs(contrib[feature_names.index(f)]) for f in top_feats) or 1.0
        for f in top_feats:
            val = contrib[feature_names.index(f)]
            width_pct = min(48, abs(val) / max_abs * 48)
            color = COLORS["sev"] if val > 0 else COLORS["low"]
            left = 50 if val > 0 else 50 - width_pct
            st.markdown(
                f"""
                <div style='display:flex;align-items:center;gap:8px;font-size:11.5px;margin-bottom:6px'>
                  <div style='width:110px;color:{COLORS['muted']};text-align:right'>{short_names[f]}</div>
                  <div style='flex:1;height:10px;background:{COLORS['panel2']};border-radius:5px;position:relative'>
                    <div style='position:absolute;top:0;bottom:0;left:{left:.0f}%;width:{width_pct:.0f}%;background:{color};border-radius:5px'></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            "<p class='hint'>Bars right of center push toward Severe; left of center push toward Low.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-label'>What-if Simulator — vary Chlorine Monoxide</div>", unsafe_allow_html=True)
        sim_x = np.linspace(0.10, 1.30, 20)
        sev_vals = []
        for v in sim_x:
            r = dict(raw_used)
            r["Chlorine Monoxide Concentration (Parts Per Billion)"] = v
            row = build_feature_row(r)
            p = model.predict_proba(row)[0]
            sev_vals.append(p[list(model.classes_).index(3)] * 100)
        bar_colors = [
            COLORS["sev"] if v > 45 else COLORS["mod"] if v > 20 else COLORS["low"] for v in sev_vals
        ]
        sim_fig = go.Figure(go.Bar(x=list(range(len(sim_x))), y=sev_vals, marker_color=bar_colors))
        sim_fig.update_layout(
            height=110,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={"visible": False},
            yaxis={"visible": False, "range": [0, 100]},
            showlegend=False,
        )
        st.plotly_chart(sim_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            "<p class='hint'>Predicted Severe-class probability as Chlorine Monoxide moves across its "
            "full range, other inputs held at the last predicted values.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================================
# TAB 2 — MODEL INSIGHTS
# ==========================================================================
with tab_insights:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("<div class='card'><h3>Model Comparison</h3>", unsafe_allow_html=True)
        m = metrics
        comp_df = pd.DataFrame(
            {
                "Metric": ["Test Accuracy", "Train Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
                "Perceptron": [
                    f"{m['perceptron']['test_accuracy']:.2%}",
                    f"{m['perceptron']['train_accuracy']:.2%}",
                    f"{m['perceptron']['precision_macro']:.2%}",
                    f"{m['perceptron']['recall_macro']:.2%}",
                    f"{m['perceptron']['f1_macro']:.2%}",
                ],
                "Softmax (selected)": [
                    f"{m['softmax']['test_accuracy']:.2%}",
                    f"{m['softmax']['train_accuracy']:.2%}",
                    f"{m['softmax']['precision_macro']:.2%}",
                    f"{m['softmax']['recall_macro']:.2%}",
                    f"{m['softmax']['f1_macro']:.2%}",
                ],
            }
        )
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.markdown(
            "<p class='muted'>Softmax Regression selected for higher accuracy, fewer Low/Moderate "
            "boundary errors, and interpretable class probabilities.</p></div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("<div class='card'><h3>Confusion Matrix — Softmax</h3>", unsafe_allow_html=True)
        cm = np.array(metrics["softmax"]["confusion_matrix"])
        labels = metrics["class_labels"]
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Pred {l}" for l in labels],
                y=[f"Actual {l}" for l in labels],
                colorscale=[[0, COLORS["panel2"]], [1, COLORS["accent"]]],
                text=cm,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        cm_fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text"]},
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            "<p class='muted'>Most confusion occurs between Low and Moderate — the two closest "
            "classes in feature space.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='card'><h3>Feature Importance Ranking</h3>", unsafe_allow_html=True)
    medals = ["🥇", "🥈", "🥉"] + [str(i) for i in range(4, len(importance) + 1)]
    max_imp = importance[0][1]
    for (name, val), medal in zip(importance, medals):
        pct = val / max_imp * 100
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:12.5px'>
              <div style='width:22px;text-align:center'>{medal}</div>
              <div style='width:230px'>{name}</div>
              <div style='flex:1;height:10px;background:{COLORS['panel2']};border-radius:5px'>
                <div style='height:100%;border-radius:5px;width:{pct:.0f}%;background:linear-gradient(90deg,{COLORS['accent']},{COLORS['accent2']})'></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        "<p class='hint'>Based on mean |coefficient| magnitude across the one-vs-rest rows of the "
        "trained Softmax model.</p></div>",
        unsafe_allow_html=True,
    )

# ==========================================================================
# TAB 3 — ABOUT
# ==========================================================================
with tab_about:
    st.markdown(
        f"""
        <div class='card'>
          <h3>About This Model</h3>
          <p class='muted'>The ozone layer plays a critical role in protecting life on Earth from
          harmful UV radiation. This model predicts depletion severity — Low, Moderate, or Severe —
          from atmospheric measurements including chlorine monoxide, CFC concentrations,
          stratospheric temperature, and polar stratospheric cloud index.</p>
          <p class='muted'>Built with Softmax Regression (multinomial logistic regression), chosen
          over a Perceptron baseline for its higher accuracy and interpretable probability outputs —
          valuable in an environmental monitoring context where prediction confidence matters.</p>
        </div>
        <div class='card'>
          <h3>Ozone Depletion — Key Milestones</h3>
          <div class='tl-item'><div class='tl-year'>1974</div><div class='tl-text'>CFCs identified as
          ozone-depleting chemicals (Molina &amp; Rowland).</div></div>
          <div class='tl-item'><div class='tl-year'>1985</div><div class='tl-text'>Antarctic
          "ozone hole" discovered by British Antarctic Survey.</div></div>
          <div class='tl-item'><div class='tl-year'>1987</div><div class='tl-text'>Montreal Protocol
          signed — global phase-out of ozone-depleting substances begins.</div></div>
          <div class='tl-item'><div class='tl-year'>2016</div><div class='tl-text'>First clear signs
          of ozone layer recovery reported.</div></div>
          <div class='tl-item'><div class='tl-year'>~2066</div><div class='tl-text'>Projected full
          recovery of the Antarctic ozone hole (NOAA/NASA estimate).</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
