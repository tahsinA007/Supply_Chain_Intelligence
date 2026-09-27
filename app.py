import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SupplyChain AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH = os.path.join(BASE_DIR, "data", "supply_chain_intelligence_Dataset.csv")

# -----------------------------------------------------------------------------
# Theme definitions
# -----------------------------------------------------------------------------
THEMES = {
    "Ocean Blue": {
        "primary": "#2563EB", "secondary": "#06B6D4", "accent": "#7C3AED",
        "soft": "#EFF6FF", "gradient": "linear-gradient(135deg,#2563EB 0%,#06B6D4 100%)",
    },
    "Aurora Purple": {
        "primary": "#7C3AED", "secondary": "#4F46E5", "accent": "#EC4899",
        "soft": "#F5F3FF", "gradient": "linear-gradient(135deg,#7C3AED 0%,#4F46E5 55%,#EC4899 100%)",
    },
    "Emerald Mint": {
        "primary": "#059669", "secondary": "#0EA5A4", "accent": "#2563EB",
        "soft": "#ECFDF5", "gradient": "linear-gradient(135deg,#059669 0%,#0EA5A4 55%,#2563EB 100%)",
    },
    "Sunset Coral": {
        "primary": "#EA580C", "secondary": "#E11D48", "accent": "#7C3AED",
        "soft": "#FFF7ED", "gradient": "linear-gradient(135deg,#EA580C 0%,#E11D48 55%,#7C3AED 100%)",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "Aurora Purple"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -----------------------------------------------------------------------------
# Model/data loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return {
        "si_num": joblib.load(os.path.join(ARTIFACT_DIR, "si_num.joblib")),
        "si_cat": joblib.load(os.path.join(ARTIFACT_DIR, "si_cat.joblib")),
        "ohe": joblib.load(os.path.join(ARTIFACT_DIR, "ohe.joblib")),
        "scaler": joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib")),
        "svr": joblib.load(os.path.join(ARTIFACT_DIR, "svr_model.joblib")),
        "xgb": joblib.load(os.path.join(ARTIFACT_DIR, "xgb_model.joblib")),
        "le": joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib")),
        "scaler_cluster": joblib.load(os.path.join(ARTIFACT_DIR, "scaler_cluster.joblib")),
        "kmeans": joblib.load(os.path.join(ARTIFACT_DIR, "kmeans_model.joblib")),
        "meta": joblib.load(os.path.join(ARTIFACT_DIR, "meta.joblib")),
    }


@st.cache_data

def load_dataset():
    return pd.read_csv(DATA_PATH)


art = load_artifacts()
meta = art["meta"]
df = load_dataset()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def apply_theme():
    palette = THEMES[st.session_state.theme]
    if st.session_state.dark_mode:
        bg = "#0B1020"
        surface = "#111827"
        surface2 = "#172033"
        text = "#F8FAFC"
        muted = "#A7B0C0"
        border = "#27344A"
        shadow = "0 10px 30px rgba(0,0,0,.28)"
    else:
        bg = "#F7F8FC"
        surface = "#FFFFFF"
        surface2 = "#F8FAFF"
        text = "#172554"
        muted = "#64748B"
        border = "#E2E8F0"
        shadow = "0 10px 28px rgba(30,41,59,.08)"

    st.markdown(
        f"""
        <style>
        :root {{
            --primary:{palette['primary']}; --secondary:{palette['secondary']};
            --accent:{palette['accent']}; --soft:{palette['soft']};
            --bg:{bg}; --surface:{surface}; --surface2:{surface2};
            --text:{text}; --muted:{muted}; --border:{border}; --shadow:{shadow};
        }}
        .stApp {{ background:var(--bg); color:var(--text); }}
        [data-testid="stHeader"] {{ background:transparent; }}
        [data-testid="stSidebar"] {{
            background: var(--gradient, {palette['gradient']});
            border-right:0;
        }}
        [data-testid="stSidebar"] * {{ color:white !important; }}
        [data-testid="stSidebarContent"] {{ padding: 1.2rem .9rem; }}
        .brand {{ font-size:1.25rem; font-weight:800; letter-spacing:-.02em; margin:.3rem .4rem 1.4rem; }}
        .brand-icon {{ display:inline-flex; width:36px; height:36px; border-radius:11px; align-items:center; justify-content:center;
            background:rgba(255,255,255,.18); margin-right:8px; }}
        .page-title {{ font-size:2.25rem; font-weight:800; color:var(--text); letter-spacing:-.035em; margin:0; }}
        .page-subtitle {{ color:var(--muted); margin:.25rem 0 1.2rem; font-size:1rem; }}
        .hero {{ border-radius:26px; padding:2.2rem; background:{palette['gradient']}; color:white;
            box-shadow:var(--shadow); position:relative; overflow:hidden; }}
        .hero:after {{ content:""; position:absolute; width:260px; height:260px; border-radius:50%;
            background:rgba(255,255,255,.10); right:-80px; top:-100px; }}
        .hero h1 {{ font-size:2.7rem; margin:0 0 .55rem; letter-spacing:-.04em; }}
        .hero p {{ max-width:620px; color:rgba(255,255,255,.88); font-size:1.02rem; line-height:1.65; }}
        .card {{ background:var(--surface); border:1px solid var(--border); border-radius:20px; padding:1.1rem;
            box-shadow:var(--shadow); }}
        .section {{ font-weight:800; font-size:1.05rem; color:var(--text); margin:.25rem 0 .7rem; }}
        .muted {{ color:var(--muted); }}
        .kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:1rem 1.05rem; box-shadow:var(--shadow); }}
        .kpi .icon {{ width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
            background:var(--soft); color:var(--primary); font-size:1.15rem; margin-bottom:.55rem; }}
        .kpi .value {{ font-size:1.55rem; font-weight:800; color:var(--text); }}
        .kpi .label {{ font-size:.82rem; color:var(--muted); margin-top:.1rem; }}
        .profile {{ border-radius:18px; padding:1.15rem; background:#FFFFFF !important; border:1px solid #D9DEE8; box-shadow:0 8px 22px rgba(15,23,42,.08); }}
        .profile-name {{ font-size:1.3rem; font-weight:850; color:var(--primary); }}
        .profile-desc {{ color:#111827 !important; font-weight:750 !important; line-height:1.6; margin-top:.5rem; background:#FFFFFF !important; }}
        .chip {{ display:inline-block; padding:.35rem .7rem; border-radius:999px; background:var(--soft); color:var(--primary); font-weight:700; font-size:.78rem; }}
        .meter-wrap {{ background:var(--surface); border:1px solid var(--border); border-radius:20px; padding:1.15rem; box-shadow:var(--shadow); }}
        .meter {{ width:210px; height:105px; margin:.4rem auto 0; border-radius:210px 210px 0 0; position:relative; overflow:hidden;
            background:conic-gradient(from 270deg, #EF4444 0deg 45deg, #F59E0B 45deg 95deg, #10B981 95deg 180deg, transparent 180deg); }}
        .meter:after {{ content:""; position:absolute; width:150px; height:75px; left:30px; bottom:0; background:var(--surface); border-radius:150px 150px 0 0; }}
        .meter-value {{ position:absolute; z-index:2; left:0; right:0; bottom:12px; text-align:center; font-size:1.55rem; font-weight:800; color:var(--text); }}
        .status-pill {{ display:inline-block; padding:.42rem .8rem; border-radius:999px; font-weight:800; }}
        .status-early {{ background:#DCFCE7; color:#166534; }} .status-on {{ background:#D1FAE5; color:#047857; }}
        .status-delay {{ background:#FEF3C7; color:#B45309; }} .status-severe {{ background:#FEE2E2; color:#B91C1C; }}
        .factor {{ padding:.7rem .8rem; border:1px solid var(--border); border-radius:14px; background:var(--surface2); }}
        .factor strong {{ color:var(--text); }}
        .footer-note {{ text-align:center; color:var(--muted); font-size:.78rem; padding:1.2rem 0 .5rem; }}
        div.stButton > button, div.stFormSubmitButton > button {{
            border:0 !important; color:white !important; background:{palette['gradient']} !important;
            border-radius:12px !important; font-weight:800 !important; padding:.62rem 1rem !important;
            box-shadow:0 7px 18px rgba(79,70,229,.22) !important;
        }}
        div[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:.7rem; }}
        .stSelectbox label, .stNumberInput label, .stSlider label {{ color:var(--text) !important; font-weight:650 !important; }}
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
            background:var(--surface) !important; color:var(--text) !important; border-color:var(--border) !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap:.3rem; }}
        .stTabs [data-baseweb="tab"] {{ color:var(--muted); }}
        .stTabs [aria-selected="true"] {{ color:var(--primary) !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def go(page):
    st.session_state.page = page
    st.rerun()


def kpi(icon, label, value):
    st.markdown(f"""
    <div class="kpi"><div class="icon">{icon}</div><div class="value">{value}</div><div class="label">{label}</div></div>
    """, unsafe_allow_html=True)


def status_class(status):
    return {
        "Early": "status-early",
        "On Time": "status-on",
        "Delayed": "status-delay",
        "Severely Delayed": "status-severe",
    }.get(status, "status-delay")


def status_probability(prediction):
    probs = prediction.get("status_probs", {})
    return float(probs.get(prediction["status"], 0.0) * 100)


PROFILE_INFO = {
    0: ("Balanced Standard", "Balanced shipment size and moderate operating conditions. This profile typically represents regular day-to-day supply-chain movements."),
    1: ("Long-Haul Care", "Longer-distance, higher lead-time shipments with relatively stronger reliability. Extra transit planning is useful for these movements."),
    2: ("High-Volume Strategic", "Large-volume shipments with high inventory, demand and transportation cost. These movements can have a bigger operational and financial impact."),
    3: ("High-Volume Core", "Large shipments with substantial demand and inventory requirements. Capacity and stock planning are especially important for this profile."),
    4: ("Fast & Reliable", "Shorter-distance shipments with shorter lead times and stronger supplier reliability. This profile is suited to faster-moving operations."),
}


def cluster_profile(cluster_id):
    return PROFILE_INFO.get(int(cluster_id), (f"Shipment Profile {cluster_id}", "A model-defined shipment group based on the shipment's numeric operating characteristics."))


def make_prediction(raw):
    X = raw.copy()
    X[meta["log_col"]] = np.log1p(X[meta["log_col"]])
    X_cat = pd.DataFrame(
        art["ohe"].transform(X[meta["ohe_col"]]),
        columns=art["ohe"].get_feature_names_out(meta["ohe_col"]),
    )
    X = pd.concat([X.drop(columns=meta["ohe_col"]).reset_index(drop=True), X_cat], axis=1)
    X[meta["num_col"]] = art["scaler"].transform(X[meta["num_col"]])
    X = X[meta["feature_columns"]]

    pred_days = float(art["svr"].predict(X)[0])
    pred_status_enc = art["xgb"].predict(X)[0]
    pred_status = art["le"].inverse_transform([pred_status_enc])[0]
    probs_raw = art["xgb"].predict_proba(X)[0]
    status_probs = {str(label): float(prob) for label, prob in zip(art["le"].classes_, probs_raw)}

    cluster_input = raw[meta["cluster_cols"]]
    cluster_scaled = art["scaler_cluster"].transform(cluster_input)
    cluster_id = int(art["kmeans"].predict(cluster_scaled)[0])

    return {
        "days": max(0.0, pred_days),
        "status": str(pred_status),
        "status_probs": status_probs,
        "cluster_id": cluster_id,
        "raw": raw.iloc[0].to_dict(),
    }


def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-icon">📦</span>SupplyChain AI</div>', unsafe_allow_html=True)
        st.caption("AI-powered supply-chain intelligence")
        st.write("")

        pages = ["Home", "Prediction", "Insights"]
        for p in pages:
            icon = {"Home": "🏠", "Prediction": "✨", "Insights": "📊"}[p]
            if st.button(f"{icon}  {p}", key=f"nav_{p}", use_container_width=True):
                go(p)

        st.divider()
        st.markdown("**Appearance**")
        theme_names = list(THEMES.keys())
        selected = st.selectbox(
            "Color theme",
            theme_names,
            index=theme_names.index(st.session_state.theme),
            key="theme_selector",
            label_visibility="collapsed",
        )
        if selected != st.session_state.theme:
            st.session_state.theme = selected
            st.rerun()

        dark = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='muted' style='color:rgba(255,255,255,.78)!important;font-size:.8rem'>Smarter insights.<br>Stronger supply chains.</div>", unsafe_allow_html=True)


def header(title, subtitle):
    c1, c2 = st.columns([5, 1.4])
    with c1:
        st.markdown(f'<div class="page-title">{title}</div><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    with c2:
        mode_label = "Dark" if st.session_state.dark_mode else "Light"
        st.markdown(f'<div style="text-align:right"><span class="chip">{st.session_state.theme}</span><br><span class="muted" style="font-size:.78rem">{mode_label} mode</span></div>', unsafe_allow_html=True)


def home_page():
    header("Welcome to SupplyChain AI", "A modern machine-learning workspace for shipment prediction and supply-chain intelligence.")

    st.markdown("""
    <div class="hero">
      <div style="position:relative;z-index:1;max-width:760px">
        <div class="chip" style="background:rgba(255,255,255,.16);color:white">AI-POWERED SUPPLY CHAIN INTELLIGENCE</div>
        <h1 style="margin-top:.8rem">Smarter predictions.<br>Better decisions.</h1>
        <p>Enter shipment characteristics, generate an ML prediction, and automatically explore the delivery result and shipment profile on the Insights page.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><div style="font-size:1.5rem">✨</div><h3>Prediction</h3><p class="muted">Estimate delivery time, delivery status and shipment profile from 12 shipment inputs.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div style="font-size:1.5rem">📦</div><h3>Shipment Profiles</h3><p class="muted">Understand the operational characteristics of the shipment cluster identified by the model.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><div style="font-size:1.5rem">📊</div><h3>Insights</h3><p class="muted">See the prediction, status meter, input factors and supporting dataset context in one place.</p></div>', unsafe_allow_html=True)

    st.markdown("<div class='section' style='margin-top:1.5rem'>Project at a glance</div>", unsafe_allow_html=True)
    cols = st.columns(6)
    values = [
        ("📦", f"{len(df):,}", "Shipment records"),
        ("🛍️", str(df["Product_Category"].nunique()), "Product categories"),
        ("🌍", str(df["Supplier_Region"].nunique()), "Supplier regions"),
        ("🚚", str(df["Shipping_Mode"].nunique()), "Shipping modes"),
        ("🏭", str(df["Warehouse_Type"].nunique()), "Warehouse types"),
        ("🤖", "3", "ML outputs"),
    ]
    for col, (icon, value, label) in zip(cols, values):
        with col:
            kpi(icon, label, value)

    st.write("")
    left, right = st.columns([3, 1])
    with left:
        st.markdown("""
        <div class="card">
          <div class="section">How it works</div>
          <div class="factor"><strong>01 · Enter shipment data</strong><br><span class="muted">Use the Prediction page to provide all 12 model inputs.</span></div><br>
          <div class="factor"><strong>02 · Run the model</strong><br><span class="muted">The existing SVR, XGBoost and K-Means artifacts generate three outputs.</span></div><br>
          <div class="factor"><strong>03 · Explore the result</strong><br><span class="muted">The app automatically opens Insights with the personalized prediction and supporting analysis.</span></div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        if st.button("🚀  Get Started", use_container_width=True):
            go("Prediction")

    st.markdown('<div class="footer-note">SupplyChain AI · Machine-learning shipment intelligence</div>', unsafe_allow_html=True)


def prediction_page():
    header("Shipment Prediction", "Enter all 12 shipment inputs. After prediction, you will be taken automatically to Insights.")

    with st.form("shipment_prediction_form", clear_on_submit=False):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">📦 Shipment information</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            product_category = st.selectbox("Product Category", ["Apparel", "Automotive Parts", "Electronics", "Furniture", "Industrial Equipment", "Perishable Food", "Pharmaceuticals"])
            shipping_mode = st.selectbox("Shipping Mode", ["Air", "Rail", "Road", "Sea"])
        with c2:
            supplier_region = st.selectbox("Supplier Region", ["East Asia", "Europe", "Latin America", "North America", "South Asia", "Southeast Asia"])
            warehouse_type = st.selectbox("Warehouse Type", ["Automated", "Cross-Dock", "Hybrid", "Manual"])
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">📐 Order & logistics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            order_quantity = st.number_input("Order Quantity", min_value=1.0, value=500.0, step=1.0)
            lead_time = st.number_input("Supplier Lead Time (days)", min_value=0.0, value=7.0, step=0.5)
        with c2:
            distance_km = st.number_input("Distance (km)", min_value=0.0, value=800.0, step=10.0)
            transport_cost = st.number_input("Transportation Cost", min_value=0.0, value=2000.0, step=100.0)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">📊 Inventory & demand</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inventory_level = st.number_input("Inventory Level", min_value=0.0, value=300.0, step=10.0)
        with c2:
            demand_forecast = st.number_input("Demand Forecast", min_value=0.0, value=450.0, step=10.0)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">🛡️ Supplier performance</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            reliability = st.slider("Supplier Reliability Score", 0.0, 100.0, 85.0, 0.1)
        with c2:
            delay_rate = st.slider("Historical Delay Rate", 0.0, 1.0, 0.10, 0.01, format="%.2f")
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

        submitted = st.form_submit_button("✨  Predict & View Insights →", use_container_width=True)

    if submitted:
        raw = pd.DataFrame([{
            "Product_Category": product_category,
            "Supplier_Region": supplier_region,
            "Shipping_Mode": shipping_mode,
            "Warehouse_Type": warehouse_type,
            "Order_Quantity": order_quantity,
            "Distance_km": distance_km,
            "Supplier_Lead_Time_days": lead_time,
            "Transportation_Cost": transport_cost,
            "Inventory_Level": inventory_level,
            "Demand_Forecast": demand_forecast,
            "Supplier_Reliability_Score": reliability,
            "Historical_Delay_Rate": delay_rate,
        }])
        st.session_state.prediction = make_prediction(raw)
        st.session_state.page = "Insights"
        st.rerun()


def status_meter(status, probability):
    st.markdown(f"""
    <div class="meter-wrap">
      <div class="section">🎯 Delivery Status Meter</div>
      <div class="muted">Model probability for the predicted status</div>
      <div class="meter"><div class="meter-value">{probability:.0f}%</div></div>
      <div style="text-align:center;margin-top:.25rem"><span class="status-pill {status_class(status)}">{status}</span></div>
    </div>
    """, unsafe_allow_html=True)


def insights_page():
    if not st.session_state.prediction:
        header("Insights", "Run a prediction first to unlock shipment-specific insights.")
        st.markdown('<div class="card" style="text-align:center;padding:3rem"><div style="font-size:3rem">🔮</div><h2>No prediction yet</h2><p class="muted">Go to Prediction, enter the shipment information and click Predict.</p></div>', unsafe_allow_html=True)
        if st.button("Go to Prediction →", use_container_width=True):
            go("Prediction")
        return

    pred = st.session_state.prediction
    profile_name, profile_desc = cluster_profile(pred["cluster_id"])
    prob = status_probability(pred)
    raw = pred["raw"]

    header("Prediction Insights", "Your shipment prediction, profile explanation and supporting data context.")
    st.markdown('<div class="chip">Based on your latest prediction</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1: kpi("⏱️", "Estimated Delivery Time", f"{pred['days']:.1f} days")
    with c2: kpi("✓", "Delivery Status", pred["status"])
    with c3: kpi("📦", "Shipment Profile", profile_name)

    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        status_meter(pred["status"], prob)
        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">📦 Shipment profile</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="profile"><div class="profile-name">{profile_name}</div><div class="profile-desc">{profile_desc}</div></div>', unsafe_allow_html=True)
        st.write("")
        st.markdown("**What this profile means**")
        cluster_centers = art["scaler_cluster"].inverse_transform(art["kmeans"].cluster_centers_)
        center = pd.Series(cluster_centers[pred["cluster_id"]], index=meta["cluster_cols"])
        comparisons = [
            ("📦 Order volume", raw["Order_Quantity"], center["Order_Quantity"], "higher" if raw["Order_Quantity"] >= center["Order_Quantity"] else "lower"),
            ("🌍 Distance", raw["Distance_km"], center["Distance_km"], "higher" if raw["Distance_km"] >= center["Distance_km"] else "lower"),
            ("⏳ Lead time", raw["Supplier_Lead_Time_days"], center["Supplier_Lead_Time_days"], "higher" if raw["Supplier_Lead_Time_days"] >= center["Supplier_Lead_Time_days"] else "lower"),
            ("🛡️ Reliability", raw["Supplier_Reliability_Score"], center["Supplier_Reliability_Score"], "higher" if raw["Supplier_Reliability_Score"] >= center["Supplier_Reliability_Score"] else "lower"),
        ]
        for label, value, typical, direction in comparisons:
            st.markdown(f'<div class="factor"><strong>{label}</strong><br><span class="muted">Your value: {value:.1f} · Profile typical: {typical:.1f} · {direction} than this profile\'s center.</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">📊 Status probabilities</div>', unsafe_allow_html=True)
        probs = pred["status_probs"]
        prob_df = pd.DataFrame({"Probability": [probs.get(s, 0) for s in ["Early", "On Time", "Delayed", "Severely Delayed"]]}, index=["Early", "On Time", "Delayed", "Severely Delayed"])
        st.bar_chart(prob_df, height=235, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">🔎 Your input snapshot</div>', unsafe_allow_html=True)
        display_map = {
            "Product_Category":"Product", "Supplier_Region":"Region", "Shipping_Mode":"Shipping Mode", "Warehouse_Type":"Warehouse",
            "Order_Quantity":"Order Quantity", "Distance_km":"Distance (km)", "Supplier_Lead_Time_days":"Lead Time (days)",
            "Transportation_Cost":"Transportation Cost", "Inventory_Level":"Inventory Level", "Demand_Forecast":"Demand Forecast",
            "Supplier_Reliability_Score":"Reliability Score", "Historical_Delay_Rate":"Historical Delay Rate"
        }
        rows = []
        for key, label in display_map.items():
            value = raw[key]
            if key == "Historical_Delay_Rate": value = f"{value:.0%}"
            elif isinstance(value, (float, np.floating)): value = f"{value:,.1f}"
            rows.append((label, value))
        snap = pd.DataFrame(rows, columns=["Input", "Value"])
        st.dataframe(snap, hide_index=True, use_container_width=True, height=320)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section">📈 Supporting dataset insights</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        kpi("📦", "Dataset Records", f"{len(df):,}")
    with b:
        median_days = df["Delivery_Time_days"].median()
        kpi("📏", "Dataset Median Delivery", f"{median_days:.1f} days")
    with c:
        same_mode = df[df["Shipping_Mode"] == raw["Shipping_Mode"]]
        mode_days = same_mode["Delivery_Time_days"].mean() if len(same_mode) else df["Delivery_Time_days"].mean()
        kpi("🚚", f"Avg. Delivery · {raw['Shipping_Mode']}", f"{mode_days:.1f} days")

    t1, t2, t3 = st.tabs(["Delivery context", "Shipping mode", "Product category"])
    with t1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">📈 Historical delivery-time distribution</div>', unsafe_allow_html=True)
        hist = df["Delivery_Time_days"].round().value_counts().sort_index()
        hist = hist[hist.index <= hist.index.max()]
        st.bar_chart(hist, height=280, use_container_width=True)
        st.markdown(
            '<div class="factor" style="margin-top:.7rem;background:var(--surface2);">'
            '<strong>What this graph tells you</strong><br>'
            '<span class="muted">Each bar represents the number of historical shipments in the training dataset that took approximately that many days to deliver. Taller bars mean that delivery time occurred more often. It gives context for your predicted delivery time above; it does <strong>not</strong> show the prediction itself.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with t2:
        mode_summary = df.groupby("Shipping_Mode", dropna=True).agg(
            Shipments=("Shipping_Mode", "size"),
            Avg_Delivery_Days=("Delivery_Time_days", "mean"),
        ).sort_values("Shipments", ascending=False)
        st.dataframe(mode_summary.round(2), use_container_width=True)
        st.bar_chart(mode_summary[["Shipments"]], height=260, use_container_width=True)
    with t3:
        cat_summary = df.groupby("Product_Category", dropna=True).agg(
            Shipments=("Product_Category", "size"),
            Avg_Delivery_Days=("Delivery_Time_days", "mean"),
        ).sort_values("Shipments", ascending=False)
        st.dataframe(cat_summary.round(2), use_container_width=True)
        st.bar_chart(cat_summary[["Shipments"]], height=260, use_container_width=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section">💡 What to pay attention to</div>', unsafe_allow_html=True)
    tips = []
    if raw["Supplier_Reliability_Score"] < df["Supplier_Reliability_Score"].median():
        tips.append("Supplier reliability is below the dataset median; reliability improvement may reduce operational risk.")
    else:
        tips.append("Supplier reliability is at or above the dataset median, which is a positive operational signal.")
    if raw["Historical_Delay_Rate"] > df["Historical_Delay_Rate"].median():
        tips.append("Historical delay rate is above the dataset median; monitor the shipment more closely.")
    else:
        tips.append("Historical delay rate is at or below the dataset median.")
    if raw["Distance_km"] > df["Distance_km"].median():
        tips.append("This shipment travels farther than the typical dataset shipment, so transit exposure is higher.")
    else:
        tips.append("This shipment distance is below the dataset median.")
    for tip in tips:
        st.markdown(f'<div class="factor">• {tip}</div>', unsafe_allow_html=True)
        st.write("")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("🔄 Make another prediction", use_container_width=True):
        go("Prediction")


# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------
apply_theme()
sidebar()

if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Prediction":
    prediction_page()
else:
    insights_page()
