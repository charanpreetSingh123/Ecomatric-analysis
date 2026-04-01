# ============================================================
# Environmental Impact Estimator (ML-Powered)
# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Environmental Impact Estimator (ML)",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING (Professional Dark Eco Theme)
# ============================================================
st.markdown("""
<style>
/* ---------- Global App Background ---------- */
.stApp {
    background: linear-gradient(135deg, #071b16 0%, #0d2a23 40%, #12382d 100%);
    color: #EAF7F0;
}

/* ---------- Main Container ---------- */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ---------- Typography ---------- */
h1, h2, h3, h4, h5, h6 {
    color: #EAF7F0 !important;
    font-weight: 700 !important;
}

p, label, div, span {
    color: #DCEFE4 !important;
}

/* ---------- Section Cards ---------- */
.section-card {
    background: rgba(17, 44, 36, 0.88);
    border: 1px solid rgba(96, 196, 151, 0.18);
    border-radius: 22px;
    padding: 1.25rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
    backdrop-filter: blur(8px);
}

/* ---------- Hero Banner ---------- */
.hero-box {
    background: linear-gradient(135deg, rgba(19, 78, 64, 0.95), rgba(15, 51, 43, 0.95));
    border: 1px solid rgba(96, 196, 151, 0.22);
    border-radius: 24px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.32);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #F3FFF8;
    margin-bottom: 0.35rem;
}

.hero-subtitle {
    font-size: 1rem;
    color: #CBE8D7;
    line-height: 1.6;
}

/* ---------- KPI Cards ---------- */
.kpi-card {
    background: linear-gradient(145deg, rgba(24, 65, 53, 0.95), rgba(13, 38, 31, 0.95));
    border: 1px solid rgba(96, 196, 151, 0.22);
    border-radius: 22px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.28);
    min-height: 135px;
}

.kpi-label {
    font-size: 0.95rem;
    color: #A8D5BC;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #F5FFF9;
    margin-bottom: 0.25rem;
}

.kpi-delta {
    font-size: 0.9rem;
    color: #8BE0B1;
    font-weight: 600;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B221C 0%, #102E25 100%);
    border-right: 1px solid rgba(96, 196, 151, 0.15);
}

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="slider"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border-radius: 12px !important;
    color: #EAF7F0 !important;
}

/* ---------- Metric Component Cleanup ---------- */
[data-testid="metric-container"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ---------- Alerts ---------- */
.stAlert {
    border-radius: 16px;
    border: 1px solid rgba(96, 196, 151, 0.16);
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #2D8A61, #4FBF83);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.55rem 1rem;
}

.stButton > button:hover {
    filter: brightness(1.05);
}

/* ---------- Divider ---------- */
hr {
    border: none;
    height: 1px;
    background: rgba(96, 196, 151, 0.12);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================
@st.cache_data
def load_emission_factors(path="emission_factors.csv"):
    """
    Load emission factors dataset.
    Falls back to a default minimal dataset if the file is missing.
    """
    if not os.path.exists(path):
        st.warning("emission_factors.csv not found — using default fallback factors.")
        return pd.DataFrame({
            "Category": ["Transport", "Energy", "Food", "Shopping"],
            "Item": ["Car (Petrol)", "Electricity", "Non-Veg Meal", "Clothes"],
            "Unit": ["km", "kWh", "meal", "item"],
            "EmissionFactor": [0.12, 0.82, 5.0, 10.0]
        })
    return pd.read_csv(path)


@st.cache_data
def load_synthetic(path="synthetic_lifestyles.csv"):
    """
    Load synthetic lifestyle dataset used for ML training and clustering.
    Returns empty DataFrame if file is missing.
    """
    if not os.path.exists(path):
        st.warning("synthetic_lifestyles.csv not found — ML features will be unavailable.")
        return pd.DataFrame()
    return pd.read_csv(path)


# Load datasets
ef = load_emission_factors()
syn = load_synthetic()

# Create a quick lookup dictionary for emission factors
ef_map = ef.set_index("Item")["EmissionFactor"].to_dict()


# ============================================================
# APP HEADER / HERO SECTION
# ============================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🌍 Environmental Impact Estimator — ML Powered</div>
    <div class="hero-subtitle">
        Estimate your daily carbon footprint using emission factors and machine learning.
        Explore category-wise emissions, simulate greener scenarios, discover your lifestyle segment,
        and get personalized sustainability recommendations.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — USER INPUTS
# ============================================================
with st.sidebar:
    st.header("🧾 Daily Lifestyle Inputs")
    st.caption("Provide your daily habits to estimate your carbon footprint.")

    # Transport input
    transport_options = sorted(ef[ef["Category"] == "Transport"]["Item"].unique())
    transport_mode = st.selectbox("Mode of Transport", transport_options)
    transport_km = st.number_input("Daily Distance (km)", min_value=0.0, value=10.0, step=0.5)

    # Energy input
    electricity_kwh = st.number_input("Electricity Consumption (kWh/day)", min_value=0.0, value=6.0, step=0.5)

    # Food input
    diet_type = st.selectbox("Diet Type", ["Vegan Meal", "Vegetarian Meal", "Non-Veg Meal"])
    meals_per_day = st.slider("Meals per Day", 1, 5, 3)

    # Shopping input
    clothes_per_month = st.slider("Clothes Purchased / Month", 0, 20, 2)
    electronics_per_year = st.slider("Electronics Purchased / Year", 0, 10, 1)

    # Optional food item details
    st.markdown("---")
    st.subheader("🍽️ Detailed Food Inputs")
    beef_kg_day = st.number_input("Beef (kg/day)", min_value=0.0, value=0.05, step=0.01)
    chicken_kg_day = st.number_input("Chicken (kg/day)", min_value=0.0, value=0.15, step=0.01)
    veggies_kg_day = st.number_input("Vegetables (kg/day)", min_value=0.0, value=0.35, step=0.01)


# ============================================================
# RULE-BASED EMISSION CALCULATION
# ============================================================
def calc_emissions(
    transport_mode,
    transport_km,
    electricity_kwh,
    diet_type,
    meals_per_day,
    clothes_per_month,
    electronics_per_year,
    beef,
    chicken,
    veggies
):
    """
    Calculate emissions using rule-based emission factors.

    Returns:
        transport_emissions
        energy_emissions
        food_emissions
        shopping_emissions
        total_emissions
    """
    transport = ef_map.get(transport_mode, 0) * transport_km
    energy = ef_map.get("Electricity", 0.82) * electricity_kwh

    # Meal-based food emissions
    food_meal = ef_map.get(diet_type, 0) * meals_per_day

    # Ingredient-based food emissions
    food_kg = (
        ef_map.get("Beef", 27.0) * beef
        + ef_map.get("Chicken", 6.9) * chicken
        + ef_map.get("Veggies", 2.0) * veggies
    )

    # Shopping emissions converted into daily estimate
    shopping = (
        ef_map.get("Clothes", 10.0) * (clothes_per_month / 30.0)
        + ef_map.get("Electronics", 50.0) * (electronics_per_year / 365.0)
    )

    total = transport + energy + food_meal + food_kg + shopping
    return transport, energy, (food_meal + food_kg), shopping, total


# Compute rule-based emissions
em_transport, em_energy, em_food, em_shopping, em_total = calc_emissions(
    transport_mode,
    transport_km,
    electricity_kwh,
    diet_type,
    meals_per_day,
    clothes_per_month,
    electronics_per_year,
    beef_kg_day,
    chicken_kg_day,
    veggies_kg_day
)


# ============================================================
# MACHINE LEARNING MODEL TRAINING
# ============================================================
ml_ready = syn.copy()

if not syn.empty:
    # Feature columns and target column
    features = [
        "transport_mode",
        "transport_km_day",
        "electricity_kwh_day",
        "diet_type",
        "meals_per_day",
        "clothes_per_month",
        "electronics_per_year",
        "beef_kg_day",
        "chicken_kg_day",
        "veggies_kg_day"
    ]
    target = "em_total"

    X = ml_ready[features]
    y = ml_ready[target]

    # Categorical and numerical features
    cat_cols = ["transport_mode", "diet_type"]
    num_cols = [c for c in features if c not in cat_cols]

    # Preprocessing pipeline
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    # Random Forest regression pipeline
    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Simple train-test split using index slicing
    idx = int(0.8 * len(ml_ready))
    X_tr, X_te = X.iloc[:idx], X.iloc[idx:]
    y_tr, y_te = y.iloc[:idx], y.iloc[idx:]

    # Train model
    model.fit(X_tr, y_tr)

    # Evaluate model
    if len(X_te) > 0:
        y_pred = model.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
    else:
        r2 = None
        mae = None
else:
    model = None
    r2 = None
    mae = None


# ============================================================
# CURRENT USER DATAFRAME FOR ML PREDICTION
# ============================================================
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
    "veggies_kg_day": veggies_kg_day
}])

# ML prediction for current user
if model is not None:
    ml_pred = float(model.predict(current_df)[0])
else:
    ml_pred = np.nan


# ============================================================
# LIFESTYLE CLUSTERING (K-MEANS SEGMENTATION)
# ============================================================
segment_label = None

if not syn.empty:
    k_pipe = Pipeline([
        ("pre", pre),
        ("km", KMeans(n_clusters=3, random_state=42, n_init=10))
    ])

    # Fit clustering pipeline
    k_pipe.fit(X)

    # Predict current user segment
    seg = int(k_pipe.named_steps["km"].predict(
        k_pipe.named_steps["pre"].transform(current_df)
    )[0])

    # Assign human-readable labels based on average emissions
    syn_segs = k_pipe.named_steps["km"].predict(
        k_pipe.named_steps["pre"].transform(X)
    )

    syn_seg_em = (
        pd.DataFrame({"seg": syn_segs, "em": y})
        .groupby("seg")["em"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    ordered = {
        syn_seg_em[0]: "Low Impact 🌱",
        syn_seg_em[1]: "Moderate Impact 🌍",
        syn_seg_em[2]: "High Impact 🔥"
    }

    segment_label = ordered.get(seg, f"Segment {seg}")


# ============================================================
# TOP KPI CARDS
# ============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Rule-based Footprint</div>
        <div class="kpi-value">{em_total:.2f} kg CO₂/day</div>
        <div class="kpi-delta">Calculated using emission factors</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if not np.isnan(ml_pred):
        delta_val = ml_pred - em_total
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ML Predicted Footprint</div>
            <div class="kpi-value">{ml_pred:.2f} kg CO₂/day</div>
            <div class="kpi-delta">Difference vs Rule-based: {delta_val:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">ML Predicted Footprint</div>
            <div class="kpi-value">—</div>
            <div class="kpi-delta">Synthetic dataset not available</div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Lifestyle Segment</div>
        <div class="kpi-value">{segment_label if segment_label else "—"}</div>
        <div class="kpi-delta">Cluster-based lifestyle classification</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================
if model is not None and (r2 is not None) and (mae is not None):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📈 Model Performance")
    p1, p2 = st.columns(2)
    with p1:
        st.metric("R² Score", f"{r2:.3f}")
    with p2:
        st.metric("Mean Absolute Error", f"{mae:.3f} kg CO₂/day")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# CARBON FOOTPRINT BREAKDOWN (DONUT CHART)
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📊 Carbon Footprint Breakdown")

labels = ["Transport", "Energy", "Food", "Shopping"]
values = [em_transport, em_energy, em_food, em_shopping]

fig = px.pie(
    names=labels,
    values=values,
    hole=0.50,
    title="Daily Carbon Emission Distribution"
)

fig.update_traces(
    textinfo="percent+label",
    pull=[0.03, 0.03, 0.03, 0.03]
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#EAF7F0"),
    title_font=dict(size=20)
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# DETAILED CATEGORY TABLE
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📋 Category-wise Emission Summary")

summary_df = pd.DataFrame({
    "Category": ["Transport", "Energy", "Food", "Shopping", "Total"],
    "Estimated Emissions (kg CO₂/day)": [
        round(em_transport, 3),
        round(em_energy, 3),
        round(em_food, 3),
        round(em_shopping, 3),
        round(em_total, 3)
    ]
})

st.dataframe(summary_df, use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🧪 What-if Scenario Analysis")
st.caption("Simulate alternative choices and compare their impact against your current lifestyle.")

c1, c2 = st.columns(2)

with c1:
    new_transport_mode = st.selectbox(
        "Try a Different Transport Mode",
        transport_options,
        index=transport_options.index(transport_mode)
    )

    new_km = st.slider(
        "Daily Distance (km) — Scenario",
        0.0,
        max(float(transport_km * 2), 50.0),
        float(transport_km),
        step=0.5
    )

with c2:
    new_elec = st.slider(
        "Electricity (kWh/day) — Scenario",
        0.0,
        max(float(electricity_kwh * 2), 20.0),
        float(electricity_kwh),
        step=0.5
    )

# Scenario emissions
em_t2, em_e2, em_f2, em_s2, em_total2 = calc_emissions(
    new_transport_mode,
    new_km,
    new_elec,
    diet_type,
    meals_per_day,
    clothes_per_month,
    electronics_per_year,
    beef_kg_day,
    chicken_kg_day,
    veggies_kg_day
)

# Comparison chart
fig2 = go.Figure()

fig2.add_trace(go.Bar(
    name="Current",
    x=labels,
    y=[em_transport, em_energy, em_food, em_shopping]
))

fig2.add_trace(go.Bar(
    name="Scenario",
    x=labels,
    y=[em_t2, em_e2, em_f2, em_s2]
))

fig2.update_layout(
    barmode="group",
    title="Current vs Scenario Emissions",
    xaxis_title="Category",
    yaxis_title="kg CO₂/day",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#EAF7F0")
)

st.plotly_chart(fig2, use_container_width=True)

delta = em_total2 - em_total
if delta < 0:
    st.success(f"Scenario Footprint: {em_total2:.2f} kg CO₂/day ({delta:.2f} improvement vs current)")
elif delta > 0:
    st.warning(f"Scenario Footprint: {em_total2:.2f} kg CO₂/day ({delta:+.2f} higher than current)")
else:
    st.info(f"Scenario Footprint: {em_total2:.2f} kg CO₂/day (No change from current)")

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# ML FEATURE IMPORTANCE
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🧠 ML Model Insights")

if model is not None:
    try:
        rf = model.named_steps["rf"]

        # Extract one-hot encoded categorical feature names
        ohe = model.named_steps["pre"].named_transformers_["cat"]
        ohe_features = list(ohe.get_feature_names_out(["transport_mode", "diet_type"]))

        # Final feature list after preprocessing
        final_features = ohe_features + [
            "transport_km_day",
            "electricity_kwh_day",
            "meals_per_day",
            "clothes_per_month",
            "electronics_per_year",
            "beef_kg_day",
            "chicken_kg_day",
            "veggies_kg_day"
        ]

        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]

        imp_df = pd.DataFrame({
            "Feature": [final_features[i] for i in top_idx],
            "Importance": [importances[i] for i in top_idx]
        })

        fig_imp = px.bar(
            imp_df,
            x="Feature",
            y="Importance",
            title="Top 10 Most Influential Features"
        )

        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAF7F0")
        )

        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.error(f"Could not compute feature importances: {e}")
else:
    st.info("Upload or provide synthetic_lifestyles.csv to enable ML model insights.")

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PERSONALIZED SUSTAINABILITY SUGGESTIONS
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("💡 Personalized Sustainability Suggestions")

tips = []

# Transport suggestion
if em_transport > 5:
    tips.append("Consider shifting some daily trips to public transport, cycling, or carpooling to reduce transport-related emissions.")

# Energy suggestion
if em_energy > 4:
    tips.append("Your electricity footprint is relatively high. Reduce standby loads, switch to LED lighting, and explore energy-efficient appliances.")

# Food suggestion
if "Non-Veg" in diet_type and em_food > 6:
    tips.append("Incorporating more plant-based meals during the week can significantly lower food-related carbon emissions.")

# Shopping suggestion
if em_shopping > 2:
    tips.append("Reducing fast-fashion purchases and extending the life of electronic devices can meaningfully lower shopping emissions.")

# Positive message if already low impact
if not tips:
    tips.append("Great job! Your current habits indicate a relatively balanced footprint. Continue monitoring weekly to identify further improvements.")

for tip in tips:
    st.markdown(f"- {tip}")

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# OPTIONAL RAW DATA PREVIEW
# ============================================================
with st.expander("📂 Dataset Preview (Optional)"):
    st.markdown("### Emission Factors Dataset")
    st.dataframe(ef, use_container_width=True)

    st.markdown("### Synthetic Lifestyle Dataset")
    if syn.empty:
        st.info("synthetic_lifestyles.csv not available.")
    else:
        st.dataframe(syn.head(20), use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.caption(
    "Note: Emission factors are approximations and outputs are intended for educational and analytical use only. "
    "Machine learning predictions depend on the quality and representativeness of the synthetic lifestyle dataset."
)
