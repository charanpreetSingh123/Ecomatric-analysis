import os
import io
import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans


# =====================
# PAGE CONFIGURATION
# =====================
st.set_page_config(
    page_title="EcoTrack AI – Carbon Footprint Intelligence",
    page_icon="🌍",
    layout="wide",
)


# =================
# CUSTOM STYLING
# =================
st.markdown(
    """
<style>
/* ── Base ── */
.stApp {
    background: linear-gradient(135deg, #071b16 0%, #0d2a23 40%, #12382d 100%);
    color: #EAF7F0;
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* ── Typography – scope tightly to avoid breaking Streamlit internals ── */
h1, h2, h3, h4, h5, h6 { color: #EAF7F0 !important; font-weight: 700 !important; }
.stMarkdown p { color: #DCEFE4 !important; }

/* ── Hero ── */
.hero-box {
    background: linear-gradient(135deg, rgba(19,78,64,0.95), rgba(15,51,43,0.95));
    border: 1px solid rgba(96,196,151,0.22);
    border-radius: 24px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 28px rgba(0,0,0,0.32);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #F3FFF8;
    margin-bottom: 0.4rem;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 0.97rem;
    color: #CBE8D7;
    line-height: 1.65;
    max-width: 860px;
}

/* ── Cards ── */
.section-card {
    background: rgba(17,44,36,0.88);
    border: 1px solid rgba(96,196,151,0.18);
    border-radius: 22px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.28);
    backdrop-filter: blur(8px);
}

/* ── KPI cards – flex column so content stacks cleanly ── */
.kpi-card {
    background: linear-gradient(145deg, rgba(24,65,53,0.95), rgba(13,38,31,0.95));
    border: 1px solid rgba(96,196,151,0.22);
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.28);
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    height: 100%;
}
.kpi-label {
    font-size: 0.82rem;
    color: #A8D5BC;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.kpi-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: #F5FFF9;
    line-height: 1.2;
    word-break: break-word;
}
.kpi-delta {
    font-size: 0.8rem;
    color: #8BE0B1;
    font-weight: 500;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B221C 0%, #102E25 100%);
    border-right: 1px solid rgba(96,196,151,0.15);
}

/* ── Inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: #EAF7F0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2D8A61, #4FBF83);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.55rem 1.1rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Alerts ── */
.stAlert { border-radius: 14px; border: 1px solid rgba(96,196,151,0.16); }

/* ── Divider ── */
hr {
    border: none;
    height: 1px;
    background: rgba(96,196,151,0.12);
    margin: 1rem 0;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    color: #A8D5BC !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #F3FFF8 !important;
    border-bottom: 2px solid #4FBF83 !important;
}

/* ── Footer ── */
.eco-footer {
    text-align: center;
    font-size: 0.78rem;
    color: #6BAD8F;
    padding: 1.2rem 0 0.5rem;
    border-top: 1px solid rgba(96,196,151,0.1);
    margin-top: 1.5rem;
    line-height: 1.7;
}
</style>
""",
    unsafe_allow_html=True,
)


# ===============
# DATA LOADING
# ================
@st.cache_data
def load_emission_factors(path="emission_factors.csv"):
    if not os.path.exists(path):
        st.warning("emission_factors.csv not found — using built-in fallback factors.")
        return pd.DataFrame(
            {
                "Category": [
                    "Transport","Transport","Transport","Transport",
                    "Energy",
                    "Food","Food","Food","Food","Food","Food",
                    "Shopping","Shopping",
                ],
                "Item": [
                    "Car (Petrol)","Car (Diesel)","Bus","Electric Vehicle",
                    "Electricity",
                    "Vegan Meal","Vegetarian Meal","Non-Veg Meal",
                    "Beef","Chicken","Veggies",
                    "Clothes","Electronics",
                ],
                "Unit": [
                    "km","km","km","km",
                    "kWh",
                    "meal","meal","meal","kg","kg","kg",
                    "item","item",
                ],
                "EmissionFactor": [
                    0.12, 0.14, 0.05, 0.03,
                    0.82,
                    2.0, 3.0, 5.0, 27.0, 6.9, 2.0,
                    10.0, 50.0,
                ],
            }
        )
    return pd.read_csv(path)


@st.cache_data
def load_synthetic(path="synthetic_lifestyles.csv"):
    if not os.path.exists(path):
        st.warning("synthetic_lifestyles.csv not found — ML features will be limited.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_real_dataset(path="Carbon_Emission.csv"):
    if not os.path.exists(path):
        st.warning("Carbon_Emission.csv not found — real-world analytics unavailable.")
        return pd.DataFrame()
    return pd.read_csv(path)


ef = load_emission_factors()
syn = load_synthetic()
real_df = load_real_dataset()

ef_map = ef.set_index("Item")["EmissionFactor"].to_dict()


# ==================
# HELPER FUNCTIONS
# ==================
def safe_get_factor(item_name: str, default_value: float) -> float:
    return ef_map.get(item_name, default_value)


def classify_impact(value: float) -> str:
    if value < 8:
        return "Low Impact 🌱"
    elif value < 16:
        return "Moderate Impact 🌍"
    return "High Impact 🔥"


def compute_sustainability_score(total_emission: float) -> float:
    score = 100 - (total_emission * 4.0)
    return round(max(0.0, min(100.0, score)), 1)


def score_band(score: float) -> str:
    if score >= 80:
        return "Excellent 🌿"
    elif score >= 60:
        return "Good ✅"
    elif score >= 40:
        return "Average ⚖️"
    return "Needs Improvement ⚠️"


def try_parse_list(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def get_real_world_average(real_data: pd.DataFrame):
    if real_data.empty or "CarbonEmission" not in real_data.columns:
        return None
    return float(real_data["CarbonEmission"].mean())


def create_download_report(user_summary_df: pd.DataFrame, meta_info: dict) -> bytes:
    buffer = io.StringIO()
    buffer.write("EcoTrack AI — Carbon Footprint Summary\n")
    buffer.write("=" * 45 + "\n\n")
    for key, value in meta_info.items():
        buffer.write(f"{key},{value}\n")
    buffer.write("\nCategory Summary\n")
    user_summary_df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def plotly_dark_layout() -> dict:
    """Shared Plotly layout settings for the dark eco theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAF7F0", size=13),
        title_font=dict(size=17, color="#EAF7F0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#EAF7F0")),
    )


# ========================
# RULE-BASED CALCULATION
# ========================
def calc_emissions(
    transport_mode, transport_km, electricity_kwh,
    diet_type, meals_per_day,
    clothes_per_month, electronics_per_year,
    beef, chicken, veggies,
):
    transport = safe_get_factor(transport_mode, 0.10) * transport_km
    energy = safe_get_factor("Electricity", 0.82) * electricity_kwh
    food_meal = safe_get_factor(diet_type, 3.0) * meals_per_day
    food_kg = (
        safe_get_factor("Beef", 27.0) * beef
        + safe_get_factor("Chicken", 6.9) * chicken
        + safe_get_factor("Veggies", 2.0) * veggies
    )
    shopping = (
        safe_get_factor("Clothes", 10.0) * (clothes_per_month / 30.0)
        + safe_get_factor("Electronics", 50.0) * (electronics_per_year / 365.0)
    )
    total = transport + energy + food_meal + food_kg + shopping
    return transport, energy, (food_meal + food_kg), shopping, total


# ============================================================
# APP HEADER
# ============================================================
st.markdown(
    """
<div class="hero-box">
    <div class="hero-title">🌍 EcoTrack AI — Hybrid Carbon Footprint Intelligence System</div>
    <div class="hero-subtitle">
        A smart sustainability dashboard that combines <strong>rule-based estimation</strong>,
        <strong>machine learning prediction</strong>, <strong>lifestyle clustering</strong>,
        <strong>scenario simulation</strong>, and <strong>real-world carbon behavior analytics</strong>.
        Uses both a synthetic lifestyle dataset and a real carbon emission dataset for hybrid intelligence.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ===================
# SIDEBAR INPUTS
# ===================
with st.sidebar:
    st.header("🧾 Daily Lifestyle Inputs")
    st.caption("Enter your daily habits to estimate your carbon footprint.")

    transport_options = (
        sorted(ef[ef["Category"] == "Transport"]["Item"].unique().tolist())
        if "Category" in ef.columns
        else ["Car (Petrol)", "Car (Diesel)", "Bus", "Electric Vehicle"]
    )
    if not transport_options:
        transport_options = ["Car (Petrol)", "Bus"]

    transport_mode = st.selectbox("Mode of Transport", transport_options)
    transport_km = st.number_input("Daily Distance (km)", min_value=0.0, value=10.0, step=0.5)
    electricity_kwh = st.number_input("Electricity Consumption (kWh/day)", min_value=0.0, value=6.0, step=0.5)
    diet_type = st.selectbox("Diet Type", ["Vegan Meal", "Vegetarian Meal", "Non-Veg Meal"])
    meals_per_day = st.slider("Meals per Day", 1, 5, 3)
    clothes_per_month = st.slider("Clothes Purchased / Month", 0, 20, 2)
    electronics_per_year = st.slider("Electronics Purchased / Year", 0, 10, 1)

    st.markdown("---")
    st.subheader("🍽️ Detailed Food Inputs")
    beef_kg_day = st.number_input("Beef (kg/day)", min_value=0.0, value=0.05, step=0.01)
    chicken_kg_day = st.number_input("Chicken (kg/day)", min_value=0.0, value=0.15, step=0.01)
    veggies_kg_day = st.number_input("Vegetables (kg/day)", min_value=0.0, value=0.35, step=0.01)

    st.markdown("---")
    st.subheader("🧠 Lifestyle Profile")
    body_type = st.selectbox("Body Type", ["underweight", "normal", "overweight", "obese"])
    sex = st.selectbox("Sex", ["male", "female"])
    air_travel = st.selectbox(
        "Air Travel Frequency", ["never", "rarely", "frequently", "very frequently"]
    )
    energy_eff = st.selectbox("Energy Efficiency Habit", ["No", "Sometimes", "Yes"])


# ==========================
# CURRENT USER EMISSIONS
# ==========================
em_transport, em_energy, em_food, em_shopping, em_total = calc_emissions(
    transport_mode, transport_km, electricity_kwh,
    diet_type, meals_per_day,
    clothes_per_month, electronics_per_year,
    beef_kg_day, chicken_kg_day, veggies_kg_day,
)

impact_label_rule = classify_impact(em_total)
sustainability_score = compute_sustainability_score(em_total)
sustainability_band = score_band(sustainability_score)


# ====================
# ML TRAINING 
# ====================
syn_model = None
best_model_name = None
syn_r2 = None
syn_mae = None
feature_importance_df = None
segment_label = None
ml_pred = np.nan
class_pred = None
class_acc = None

EXPECTED_FEATURES = [
    "transport_mode", "transport_km_day", "electricity_kwh_day",
    "diet_type", "meals_per_day", "clothes_per_month",
    "electronics_per_year", "beef_kg_day", "chicken_kg_day", "veggies_kg_day",
]
TARGET_COL = "em_total"
CAT_COLS = ["transport_mode", "diet_type"]
NUM_COLS = [c for c in EXPECTED_FEATURES if c not in CAT_COLS]

current_df = pd.DataFrame([{
    "transport_mode": transport_mode,
    "transport_km_day": transport_km,
    "electricity_kwh_day": electricity_kwh,
    "diet_type": diet_type,
    "meals_per_day": meals_per_day,
    "clothes_per_month": clothes_per_month,
    "electronics_per_year": electronics_per_year,
    "beef_kg_day": beef_kg_day,
    "chicken_kg_day": chicken_kg_day,
    "veggies_kg_day": veggies_kg_day,
}])

if not syn.empty and all(c in syn.columns for c in EXPECTED_FEATURES) and TARGET_COL in syn.columns:
    with st.spinner("Training ML models on synthetic dataset…"):
        X = syn[EXPECTED_FEATURES].copy()
        y = syn[TARGET_COL].copy()

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        candidates = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

        best_mae_val = float("inf")
        for name, reg in candidates.items():
            pipe = Pipeline([("pre", pre), ("model", reg)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            cur_mae = mean_absolute_error(y_test, preds)
            cur_r2 = r2_score(y_test, preds)
            if cur_mae < best_mae_val:
                best_mae_val = cur_mae
                syn_model = pipe
                best_model_name = name
                syn_r2 = cur_r2
                syn_mae = cur_mae

        if syn_model is not None:
            ml_pred = float(syn_model.predict(current_df)[0])

        # Classification
        y_class = y.apply(classify_impact)
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
        clf = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=200, random_state=42))])
        clf.fit(Xc_train, yc_train)
        class_acc = accuracy_score(yc_test, clf.predict(Xc_test))
        class_pred = clf.predict(current_df)[0]

        # Clustering
        cluster_pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ])
        cluster_X = cluster_pre.fit_transform(X)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(cluster_X)
        current_cluster = int(km.predict(cluster_pre.transform(current_df))[0])
        syn_clusters = km.predict(cluster_X)

        cluster_order = (
            pd.DataFrame({"cluster": syn_clusters, "em": y})
            .groupby("cluster")["em"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        cluster_names = {
            cluster_order[0]: "Eco Conscious 🌱",
            cluster_order[1]: "Balanced Lifestyle 🌍",
            cluster_order[2]: "Carbon Intensive 🔥",
        }
        segment_label = cluster_names.get(current_cluster, f"Cluster {current_cluster}")

        try:
            if best_model_name in ("Random Forest", "Gradient Boosting"):
                model_obj = syn_model.named_steps["model"]
                if hasattr(model_obj, "feature_importances_"):
                    ohe = syn_model.named_steps["pre"].named_transformers_["cat"]
                    raw_cat = ohe.get_feature_names_out(CAT_COLS)
                    clean_cat = [n.split("_", 2)[-1] if "_" in n else n for n in raw_cat]
                    all_features = list(clean_cat) + NUM_COLS
                    importances = model_obj.feature_importances_
                    top_idx = np.argsort(importances)[-10:][::-1]
                    feature_importance_df = pd.DataFrame({
                        "Feature": [all_features[i] for i in top_idx],
                        "Importance": [round(importances[i], 4) for i in top_idx],
                    })
        except Exception:
            feature_importance_df = None


# =============================
# REAL DATASET BENCHMARKING
# ===========================================================
real_avg = get_real_world_average(real_df)
relative_position = (em_total / real_avg * 100) if real_avg else None


# =====================
# MAIN TABS
# =====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🧠 ML & Insights",
    "🧪 Scenario Optimizer",
    "📂 Dataset Analytics",
])


# =========================
# TAB 1 — MAIN DASHBOARD
# ==========================
with tab1:

    # ── KPI Cards ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")

    ml_text = f"{ml_pred:.2f} kg CO₂/day" if not np.isnan(ml_pred) else "—"
    ml_sub = f"Best model: {best_model_name}" if best_model_name else "Dataset unavailable"
    lifestyle_label = segment_label if segment_label else impact_label_rule

    kpi_data = [
        ("Rule-Based Footprint", f"{em_total:.2f} kg CO₂/day", "Factor-based transparent estimate", c1),
        ("ML Predicted Footprint", ml_text, ml_sub, c2),
        (f"Sustainability Score", f"{sustainability_score} / 100", sustainability_band, c3),
        ("Lifestyle Segment", lifestyle_label, "Hybrid behavioural clustering", c4),
    ]

    for label, value, delta, col in kpi_data:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-delta">{delta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Benchmark ──────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📌 Real-World Benchmark")

        b1, b2, b3 = st.columns(3)
        b1.metric("Your Estimated Footprint", f"{em_total:.2f} kg CO₂/day")
        b2.metric(
            "Real Dataset Average",
            f"{real_avg:.1f}" if real_avg is not None else "Unavailable",
        )
        b3.metric(
            "Your Level vs Average",
            f"{relative_position:.1f}%" if relative_position is not None else "Unavailable",
            help="100% = exactly at average. <100% = below average (better). >100% = above average.",
        )
        st.caption(
            "ℹ️ The real-world dataset target (CarbonEmission) may use a different unit scale "
            "than the rule-based daily estimate. It is used here for behavioural comparison only."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Footprint Breakdown ────────────────────────────────
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Carbon Footprint Breakdown")

        col_pie, col_bar = st.columns([1, 1], gap="medium")

        labels = ["Transport", "Energy", "Food", "Shopping"]
        values = [em_transport, em_energy, em_food, em_shopping]

        with col_pie:
            fig_pie = px.pie(
                names=labels, values=values,
                hole=0.55,
                title="Daily Emission Distribution",
                color_discrete_sequence=px.colors.sequential.Teal,
            )
            fig_pie.update_traces(textinfo="percent+label", pull=[0.04] * 4)
            fig_pie.update_layout(**plotly_dark_layout())
            st.plotly_chart(fig_pie, width="stretch")

        with col_bar:
            fig_bar = px.bar(
                x=labels, y=values,
                labels={"x": "Category", "y": "kg CO₂/day"},
                title="Emission by Category",
                color=values,
                color_continuous_scale="Teal",
            )
            fig_bar.update_coloraxes(showscale=False)
            fig_bar.update_layout(**plotly_dark_layout())
            st.plotly_chart(fig_bar, width="stretch")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Category Summary Table ─────────────────────────────
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📋 Category-wise Emission Summary")

        summary_df = pd.DataFrame({
            "Category": ["Transport", "Energy", "Food", "Shopping", "Total"],
            "Estimated Emissions (kg CO₂/day)": [
                round(em_transport, 3), round(em_energy, 3),
                round(em_food, 3), round(em_shopping, 3), round(em_total, 3),
            ],
            "Share (%)": [
                f"{em_transport/em_total*100:.1f}%" if em_total > 0 else "—",
                f"{em_energy/em_total*100:.1f}%" if em_total > 0 else "—",
                f"{em_food/em_total*100:.1f}%" if em_total > 0 else "—",
                f"{em_shopping/em_total*100:.1f}%" if em_total > 0 else "—",
                "100%",
            ],
        })
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Personalised Suggestions ───────────────────────────
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("💡 Personalised Sustainability Suggestions")

        tips = []
        if em_transport > 5:
            tips.append("🚌 **Transport** — Replace some trips with bus, metro, carpooling, or cycling.")
        if em_energy > 4:
            tips.append("⚡ **Energy** — Reduce standby loads, switch to LED lighting, and optimise appliance usage.")
        if "Non-Veg" in diet_type and em_food > 6:
            tips.append("🥗 **Diet** — Reducing high-emission meals and adding plant-based options can significantly lower your footprint.")
        if beef_kg_day > 0.08:
            tips.append("🥩 **Beef** — Beef has one of the highest per-kg emission impacts. Even small reductions help noticeably.")
        if em_shopping > 2:
            tips.append("🛍️ **Shopping** — Reducing fast fashion and extending electronics lifespan lowers this category.")
        if air_travel in ("frequently", "very frequently"):
            tips.append("✈️ **Air Travel** — Frequent flying strongly increases real-world carbon profiles. Consider rail or fewer trips.")
        if energy_eff == "No":
            tips.append("🏠 **Energy Habits** — Improving efficiency habits (e.g. insulation, smart appliances) reduces both direct and indirect emissions.")
        if not tips:
            tips.append("✅ Great job! Your current habits reflect a relatively balanced lifestyle. Keep tracking weekly for further optimisation.")

        for tip in tips:
            st.markdown(f"- {tip}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Download ───────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("⬇️ Download Your Summary")

        report_meta = {
            "Rule-Based Footprint (kg CO2/day)": round(em_total, 3),
            "ML Predicted Footprint (kg CO2/day)": round(ml_pred, 3) if not np.isnan(ml_pred) else "Unavailable",
            "Sustainability Score": sustainability_score,
            "Impact Class": class_pred if class_pred else impact_label_rule,
            "Lifestyle Segment": segment_label if segment_label else "Unavailable",
            "Best Regression Model": best_model_name if best_model_name else "Unavailable",
        }

        st.download_button(
            label="📥 Download Carbon Footprint Report (CSV)",
            data=create_download_report(summary_df, report_meta),
            file_name="ecotrack_ai_summary.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =============================
# TAB 2 — ML & INSIGHTS
# ============================================================
with tab2:

    # ── Model Performance ──────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📈 Model Performance — Synthetic Dataset")

    if syn_model is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best Model", best_model_name)
        m2.metric("R² Score", f"{syn_r2:.4f}")
        m3.metric("MAE", f"{syn_mae:.3f} kg CO₂/day")
        m4.metric("Classification Accuracy", f"{class_acc:.3f}" if class_acc else "—")

        if class_pred:
            st.info(f"🔍 Predicted Impact Class for your inputs: **{class_pred}**")
    else:
        st.info("Synthetic dataset unavailable or required columns are missing.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature Importance ─────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧠 Feature Importance — Top Emission Drivers")

    if feature_importance_df is not None and not feature_importance_df.empty:
        fig_imp = px.bar(
            feature_importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top 10 Features ({best_model_name})",
            color="Importance",
            color_continuous_scale="Teal",
        )
        fig_imp.update_layout(**plotly_dark_layout(), yaxis={"categoryorder": "total ascending"})
        fig_imp.update_coloraxes(showscale=False)
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info("Feature importance is available when a tree-based model wins the comparison.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sustainability Gauge ───────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🎯 Sustainability Score Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sustainability_score,
        delta={"reference": 60, "increasing": {"color": "#FF6B6B"}, "decreasing": {"color": "#4FBF83"}},
        title={"text": f"Sustainability Score<br><span style='font-size:0.85rem;color:#8BE0B1'>{sustainability_band}</span>"},
        domain={"x": [0.1, 0.9], "y": [0.1, 0.9]},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#EAF7F0"},
            "bar": {"color": "#4FBF83", "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(96,196,151,0.2)",
            "steps": [
                {"range": [0, 40], "color": "rgba(255,99,71,0.35)"},
                {"range": [40, 70], "color": "rgba(255,215,0,0.25)"},
                {"range": [70, 100], "color": "rgba(78,191,131,0.3)"},
            ],
            "threshold": {
                "line": {"color": "#EAF7F0", "width": 2},
                "thickness": 0.75,
                "value": 60,
            },
        },
    ))
    gauge.update_layout(height=320, **plotly_dark_layout())
    st.plotly_chart(gauge, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Rule vs ML comparison ──────────────────────────────
    if not np.isnan(ml_pred):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("⚖️ Rule-Based vs ML Prediction")

        diff = ml_pred - em_total
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rule-Based", f"{em_total:.3f} kg CO₂/day")
        col_b.metric("ML Predicted", f"{ml_pred:.3f} kg CO₂/day")
        col_c.metric("Difference", f"{diff:+.3f} kg CO₂/day",
                     delta_color="inverse" if diff > 0 else "normal")

        st.caption(
            "A positive difference means the ML model predicts a higher footprint than the rule-based estimate. "
            "This can reflect non-linear interactions between lifestyle features that simple factor multiplication misses."
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ==================================
# TAB 3 — SCENARIO OPTIMIZER
# ============================================================
with tab3:

    # ── What-if ───────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧪 What-If Scenario Analysis")
    st.caption("Simulate greener choices and compare them against your current lifestyle.")

    s1, s2 = st.columns(2, gap="medium")

    with s1:
        new_transport_mode = st.selectbox(
            "Scenario Transport Mode",
            transport_options,
            index=transport_options.index(transport_mode) if transport_mode in transport_options else 0,
        )
        # Guard against transport_km == 0
        max_km = max(float(transport_km) * 2, 50.0)
        new_km = st.slider("Scenario Daily Distance (km)", 0.0, max_km, float(transport_km), step=0.5)

    with s2:
        max_elec = max(float(electricity_kwh) * 2, 20.0)
        new_elec = st.slider("Scenario Electricity (kWh/day)", 0.0, max_elec, float(electricity_kwh), step=0.5)
        new_diet = st.selectbox(
            "Scenario Diet Type",
            ["Vegan Meal", "Vegetarian Meal", "Non-Veg Meal"],
            index=["Vegan Meal", "Vegetarian Meal", "Non-Veg Meal"].index(diet_type),
        )

    em_t2, em_e2, em_f2, em_s2, em_total2 = calc_emissions(
        new_transport_mode, new_km, new_elec,
        new_diet, meals_per_day,
        clothes_per_month, electronics_per_year,
        beef_kg_day, chicken_kg_day, veggies_kg_day,
    )

    fig_scenario = go.Figure()
    categories = ["Transport", "Energy", "Food", "Shopping"]
    fig_scenario.add_trace(go.Bar(
        name="Current",
        x=categories,
        y=[em_transport, em_energy, em_food, em_shopping],
        marker_color="rgba(78,191,131,0.7)",
    ))
    fig_scenario.add_trace(go.Bar(
        name="Scenario",
        x=categories,
        y=[em_t2, em_e2, em_f2, em_s2],
        marker_color="rgba(255,165,0,0.7)",
    ))
    fig_scenario.update_layout(
        barmode="group",
        title="Current vs Scenario — Emission Comparison",
        xaxis_title="Category",
        yaxis_title="kg CO₂ / day",
        **plotly_dark_layout(),
    )
    st.plotly_chart(fig_scenario, width="stretch")

    delta = em_total2 - em_total
    if delta < 0:
        st.success(f"✅ Scenario Footprint: **{em_total2:.2f} kg CO₂/day** — saves **{abs(delta):.2f} kg CO₂/day** vs current.")
    elif delta > 0:
        st.warning(f"⚠️ Scenario Footprint: **{em_total2:.2f} kg CO₂/day** — **{delta:.2f} kg CO₂/day higher** than current.")
    else:
        st.info(f"ℹ️ Scenario Footprint: **{em_total2:.2f} kg CO₂/day** — no change from current.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Auto Recommendations ───────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🤖 Automatic Greener Alternatives")

    recommendations = []

    for mode in transport_options:
        _, _, _, _, test_total = calc_emissions(
            mode, transport_km, electricity_kwh,
            diet_type, meals_per_day,
            clothes_per_month, electronics_per_year,
            beef_kg_day, chicken_kg_day, veggies_kg_day,
        )
        recommendations.append({
            "Alternative": f"Switch transport → {mode}",
            "Estimated Total (kg CO₂/day)": round(test_total, 3),
            "Savings vs Current (kg CO₂/day)": round(em_total - test_total, 3),
        })

    for pct in [10, 20, 30]:
        _, _, _, _, test_total = calc_emissions(
            transport_mode, transport_km, electricity_kwh * (1 - pct / 100),
            diet_type, meals_per_day,
            clothes_per_month, electronics_per_year,
            beef_kg_day, chicken_kg_day, veggies_kg_day,
        )
        recommendations.append({
            "Alternative": f"Reduce electricity by {pct}%",
            "Estimated Total (kg CO₂/day)": round(test_total, 3),
            "Savings vs Current (kg CO₂/day)": round(em_total - test_total, 3),
        })

    for diet_opt in ["Vegan Meal", "Vegetarian Meal", "Non-Veg Meal"]:
        if diet_opt != diet_type:
            _, _, _, _, test_total = calc_emissions(
                transport_mode, transport_km, electricity_kwh,
                diet_opt, meals_per_day,
                clothes_per_month, electronics_per_year,
                beef_kg_day, chicken_kg_day, veggies_kg_day,
            )
            recommendations.append({
                "Alternative": f"Switch diet → {diet_opt}",
                "Estimated Total (kg CO₂/day)": round(test_total, 3),
                "Savings vs Current (kg CO₂/day)": round(em_total - test_total, 3),
            })

    rec_df = (
        pd.DataFrame(recommendations)
        .sort_values("Savings vs Current (kg CO₂/day)", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(rec_df.head(10), width="stretch", hide_index=True)

    if not rec_df.empty and rec_df.iloc[0]["Savings vs Current (kg CO₂/day)"] > 0:
        best_alt = rec_df.iloc[0]
        st.success(
            f"🏆 Best quick win: **{best_alt['Alternative']}** "
            f"can save ~**{best_alt['Savings vs Current (kg CO₂/day)']:.2f} kg CO₂/day**."
        )
    else:
        st.info("Your current choices are already relatively optimised under the tested alternatives.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 4 — DATASET ANALYTICS
# ============================================================
with tab4:

    # ── Real Dataset Overview ──────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📂 Real Dataset Overview — Carbon_Emission.csv")

    if real_df.empty:
        st.info("Carbon_Emission.csv is not available.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Rows", f"{len(real_df):,}")
        d2.metric("Columns", len(real_df.columns))
        d3.metric(
            "Avg CarbonEmission",
            f"{real_df['CarbonEmission'].mean():.1f}" if "CarbonEmission" in real_df.columns else "—",
        )
        st.dataframe(real_df.head(15), width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Distribution ───────────────────────────────────────
    if not real_df.empty and "CarbonEmission" in real_df.columns:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📈 CarbonEmission Distribution")

        fig_hist = px.histogram(
            real_df, x="CarbonEmission", nbins=35,
            title="Distribution of CarbonEmission in Real Dataset",
            color_discrete_sequence=["#4FBF83"],
        )
        fig_hist.update_layout(**plotly_dark_layout())
        st.plotly_chart(fig_hist, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Diet & Transport side by side ──────────────────────
    has_diet = not real_df.empty and all(c in real_df.columns for c in ["Diet", "CarbonEmission"])
    has_transport = not real_df.empty and all(c in real_df.columns for c in ["Transport", "CarbonEmission"])

    if has_diet or has_transport:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🥗 Diet & 🚌 Transport vs CarbonEmission")

        cols = st.columns(2, gap="medium")

        if has_diet:
            diet_avg = real_df.groupby("Diet", as_index=False)["CarbonEmission"].mean()
            fig_diet = px.bar(
                diet_avg, x="Diet", y="CarbonEmission",
                title="Avg Emission by Diet",
                color="CarbonEmission", color_continuous_scale="Teal",
            )
            fig_diet.update_coloraxes(showscale=False)
            fig_diet.update_layout(**plotly_dark_layout())
            cols[0].plotly_chart(fig_diet, width="stretch")

        if has_transport:
            transport_avg = real_df.groupby("Transport", as_index=False)["CarbonEmission"].mean()
            fig_tr = px.bar(
                transport_avg, x="Transport", y="CarbonEmission",
                title="Avg Emission by Transport",
                color="CarbonEmission", color_continuous_scale="Teal",
            )
            fig_tr.update_coloraxes(showscale=False)
            fig_tr.update_layout(**plotly_dark_layout())
            cols[1].plotly_chart(fig_tr, width="stretch")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Recycling ──────────────────────────────────────────
    if not real_df.empty and "Recycling" in real_df.columns:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("♻️ Recycling Behaviour Insights")

        recycle_counts = [len(try_parse_list(v)) for v in real_df["Recycling"]]
        fig_rec = px.histogram(
            pd.DataFrame({"Recycled Categories Count": recycle_counts}),
            x="Recycled Categories Count",
            nbins=8,
            title="How Many Recycling Categories People Typically Use",
            color_discrete_sequence=["#4FBF83"],
        )
        fig_rec.update_layout(**plotly_dark_layout())
        st.plotly_chart(fig_rec, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Synthetic Preview ──────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧪 Synthetic Dataset Preview")

    if syn.empty:
        st.info("synthetic_lifestyles.csv is not available.")
    else:
        s1c, s2c, s3c = st.columns(3)
        s1c.metric("Rows", f"{len(syn):,}")
        s2c.metric("Columns", len(syn.columns))
        s3c.metric(
            "Avg em_total",
            f"{syn['em_total'].mean():.3f} kg CO₂/day" if "em_total" in syn.columns else "—",
        )
        st.dataframe(syn.head(15), width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
<div class="eco-footer">
    🌍 <strong>EcoTrack AI</strong> — Hybrid Carbon Footprint Intelligence System<br>
    Emission factors are approximate and intended for educational use.<br>
    Synthetic dataset powers direct user-aligned ML prediction · Real dataset (Carbon_Emission.csv) powers benchmarking & analytics.<br>
    <span style="color:#4a8c6a;">Built with Python · Streamlit · Scikit-learn · Plotly</span>
</div>
""",
    unsafe_allow_html=True,
)
