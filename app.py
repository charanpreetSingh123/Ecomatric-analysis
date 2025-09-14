import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Environmental Impact Estimator (ML)", layout="wide")

st.title("Environmental Impact Estimator — with Machine Learning")
st.caption("Estimate your carbon footprint, see category breakdowns, and get AI-driven insights. Fast, visual, and extensible.")

# ---------- Data loading (with auto-generate fallback) ----------
@st.cache_data
def load_emission_factors(path="emission_factors.csv"):
    if not os.path.exists(path):
        st.warning("emission_factors.csv not found — using default factors.")
        return pd.DataFrame({
            "Category":["Transport","Energy","Food","Shopping"],
            "Item":["Car (Petrol)","Electricity","Non-Veg Meal","Clothes"],
            "Unit":["km","kWh","meal","item"],
            "EmissionFactor":[0.12,0.82,5.0,10.0]
        })
    return pd.read_csv(path)

@st.cache_data
def load_synthetic(path="synthetic_lifestyles.csv"):
    if not os.path.exists(path):
        st.warning("synthetic_lifestyles.csv not found — starting with an empty dataset.")
        return pd.DataFrame()
    return pd.read_csv(path)

ef = load_emission_factors()
syn = load_synthetic()

# Quick lookup
ef_map = ef.set_index("Item")["EmissionFactor"].to_dict()

# ---------- Sidebar: user inputs ----------
with st.sidebar:
    st.header("🧾 Your Daily Habits")
    transport_mode = st.selectbox("Mode of Transport", sorted(ef[ef["Category"]=="Transport"]["Item"].unique()))
    transport_km = st.number_input("Daily Distance (km)", min_value=0.0, value=10.0, step=0.5)
    electricity_kwh = st.number_input("Electricity (kWh/day)", min_value=0.0, value=6.0, step=0.5)
    diet_type = st.selectbox("Diet Type", ["Vegan Meal","Vegetarian Meal","Non-Veg Meal"])
    meals_per_day = st.slider("Meals per day", 1, 5, 3)
    clothes_per_month = st.slider("Clothes / month", 0, 20, 2)
    electronics_per_year = st.slider("Electronics / year", 0, 10, 1)
    beef_kg_day = st.number_input("Beef (kg/day, optional)", min_value=0.0, value=0.05, step=0.01)
    chicken_kg_day = st.number_input("Chicken (kg/day, optional)", min_value=0.0, value=0.15, step=0.01)
    veggies_kg_day = st.number_input("Veggies (kg/day, optional)", min_value=0.0, value=0.35, step=0.01)

# ---------- Rule-based calculation ----------
def calc_emissions(transport_mode, transport_km, electricity_kwh, diet_type, meals_per_day, clothes_per_month, electronics_per_year, beef, chicken, veggies):
    transport = ef_map.get(transport_mode, 0) * transport_km
    energy = ef_map.get("Electricity", 0.82) * electricity_kwh
    food_meal = ef_map.get(diet_type, 0) * meals_per_day
    food_kg = ef_map.get("Beef", 27.0)*beef + ef_map.get("Chicken", 6.9)*chicken + ef_map.get("Veggies", 2.0)*veggies
    shopping = ef_map.get("Clothes", 10.0) * (clothes_per_month/30.0) + ef_map.get("Electronics", 50.0) * (electronics_per_year/365.0)
    total = transport + energy + food_meal + food_kg + shopping
    return transport, energy, (food_meal + food_kg), shopping, total

em_transport, em_energy, em_food, em_shopping, em_total = calc_emissions(
    transport_mode, transport_km, electricity_kwh, diet_type, meals_per_day, clothes_per_month, electronics_per_year, beef_kg_day, chicken_kg_day, veggies_kg_day
)

# ---------- Machine Learning: train on synthetic dataset ----------
ml_ready = syn.copy()
if not syn.empty:
    features = ["transport_mode","transport_km_day","electricity_kwh_day","diet_type","meals_per_day","clothes_per_month","electronics_per_year","beef_kg_day","chicken_kg_day","veggies_kg_day"]
    target = "em_total"

    X = ml_ready[features]
    y = ml_ready[target]

    # Preprocess: one-hot for categoricals
    cat_cols = ["transport_mode","diet_type"]
    num_cols = [c for c in features if c not in cat_cols]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    model.fit(X, y)

    # Evaluate on holdout (simple split)
    idx = int(0.8*len(ml_ready))
    X_tr, X_te = X.iloc[:idx], X.iloc[idx:]
    y_tr, y_te = y.iloc[:idx], y.iloc[idx:]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
else:
    model = None
    r2 = None
    mae = None

# Predict ML estimate for current user
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

if model is not None:
    ml_pred = float(model.predict(current_df)[0])
else:
    ml_pred = np.nan

# ---------- Clustering (lifestyle segments) ----------
segment_label = None
if not syn.empty:
    # Fit on numeric features only (with OHE for categorials via pipeline)
    k_pipe = Pipeline([("pre", pre), ("km", KMeans(n_clusters=3, random_state=42, n_init=10))])
    k_pipe.fit(X)
    seg = int(k_pipe.named_steps["km"].predict(k_pipe.named_steps["pre"].transform(current_df))[0])
    # Name segments by average emissions
    labels_map = {}
    syn_segs = k_pipe.named_steps["km"].predict(k_pipe.named_steps["pre"].transform(X))
    syn_seg_em = pd.DataFrame({"seg": syn_segs, "em": y}).groupby("seg")["em"].mean().sort_values().index.tolist()
    # Lowest mean -> "Low Impact", mid -> "Moderate", highest -> "High"
    ordered = {syn_seg_em[0]:"Low Impact 🌱", syn_seg_em[1]:"Moderate Impact 🌍", syn_seg_em[2]:"High Impact 🔥"}
    segment_label = ordered.get(seg, f"Segment {seg}")

# ---------- Top KPIs ----------
col1, col2, col3 = st.columns(3)
col1.metric("Rule-based Footprint", f"{em_total:.2f} kg CO₂/day")
if not np.isnan(ml_pred):
    col2.metric("ML Predicted Footprint", f"{ml_pred:.2f} kg CO₂/day", delta=f"{ml_pred - em_total:+.2f}")
else:
    col2.metric("ML Predicted Footprint", "—")
if segment_label:
    col3.metric("Lifestyle Segment", segment_label)
else:
    col3.metric("Lifestyle Segment", "—")

# ---------- Breakdown donut (Plotly) ----------
st.subheader("📊 Carbon Footprint Breakdown")
labels = ["Transport","Energy","Food","Shopping"]
values = [em_transport, em_energy, em_food, em_shopping]
fig = px.pie(
    names=labels, values=values, hole=0.45, title="Daily Carbon Breakdown",
)
fig.update_traces(textinfo="percent+label", pull=[0.05]*4)
st.plotly_chart(fig, use_container_width=True)

# ---------- What-if analysis (slider-based) ----------
st.subheader("🧪 What-if Analysis")
c1, c2 = st.columns(2)
with c1:
    new_transport_mode = st.selectbox("Try a different transport mode", sorted(ef[ef["Category"]=="Transport"]["Item"].unique()), index=list(sorted(ef[ef["Category"]=="Transport"]["Item"].unique())).index(transport_mode))
    new_km = st.slider("Daily Distance (km) — scenario", 0.0, max(transport_km*2, 50.0), float(transport_km), step=0.5)
with c2:
    new_elec = st.slider("Electricity (kWh/day) — scenario", 0.0, max(electricity_kwh*2, 20.0), float(electricity_kwh), step=0.5)

em_t2, em_e2, em_f2, em_s2, em_total2 = calc_emissions(
    new_transport_mode, new_km, new_elec, diet_type, meals_per_day, clothes_per_month, electronics_per_year, beef_kg_day, chicken_kg_day, veggies_kg_day
)

fig2 = go.Figure()
fig2.add_trace(go.Bar(name="Current", x=labels, y=[em_transport, em_energy, em_food, em_shopping]))
fig2.add_trace(go.Bar(name="Scenario", x=labels, y=[em_t2, em_e2, em_f2, em_s2]))
fig2.update_layout(barmode='group', title="Category Comparison: Current vs Scenario", xaxis_title="", yaxis_title="kg CO₂/day")
st.plotly_chart(fig2, use_container_width=True)

delta = em_total2 - em_total
st.info(f"**Scenario impact:** {em_total2:.2f} kg CO₂/day ({delta:+.2f} vs current)")

# ---------- Feature importance (if ML model available) ----------
st.subheader("🧠 ML Model Insight")
if model is not None:
    try:
        rf = model.named_steps["rf"]
        # Extract feature names after preprocessing
        ohe = model.named_steps["pre"].named_transformers_["cat"]
        ohe_features = list(ohe.get_feature_names_out(["transport_mode","diet_type"]))
        final_features = ohe_features + ["transport_km_day","electricity_kwh_day","meals_per_day","clothes_per_month","electronics_per_year","beef_kg_day","chicken_kg_day","veggies_kg_day"]
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        imp_df = pd.DataFrame({
            "feature": [final_features[i] for i in top_idx],
            "importance": [importances[i] for i in top_idx]
        })
        fig_imp = px.bar(imp_df, x="feature", y="importance", title="Top Feature Importances")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.write("Could not compute feature importances.", e)
else:
    st.write("Upload or generate a dataset to train the ML model.")

# ---------- Tips (rule + data-driven) ----------
st.subheader("💡 Personalized Suggestions")
tips = []
if em_transport > 5:
    tips.append("Consider switching some trips to public transport or carpool — big gains in transport emissions.")
if em_energy > 4:
    tips.append("Reduce electricity use: turn off standby loads, adopt LEDs, consider solar plans.")
if "Non-Veg" in diet_type and em_food > 6:
    tips.append("Try adding more plant-based meals — even 3 per week can lower emissions significantly.")
if em_shopping > 2:
    tips.append("Buy fewer fast-fashion items and extend device lifecycles (repair > replace).")
if not tips:
    tips.append("You're doing great! Keep tracking weekly to spot trends and improvements.")
for t in tips:
    st.markdown(f"- {t}")

st.caption("Note: Emission factors are approximations; results are estimates for educational use.")
