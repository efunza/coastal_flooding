from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Kenya Coastal Flood Early Warning", layout="wide")

DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features.csv"
MODEL_PATH = DATA_DIR / "model.joblib"


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_resource
def load_model(path: Path):
    payload = joblib.load(path)
    # payload expected: {"model": fitted_model, "features": [..]}
    return payload["model"], payload["features"]


def compute_probs(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback
    scores = model.predict(X)
    scores = scores.astype(float)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


def risk_label(p: float, high_threshold: float) -> str:
    if p >= high_threshold:
        return "HIGH"
    if p >= high_threshold * 0.6:
        return "MEDIUM"
    return "LOW"


def color_rgb(p: float, high_threshold: float) -> list[int]:
    # green / orange / red
    if p >= high_threshold:
        return [220, 40, 40]
    if p >= high_threshold * 0.6:
        return [245, 160, 40]
    return [40, 180, 90]


# -----------------------------
# UI Header
# -----------------------------
st.title("🛰️ Kenya Coastal Flooding Early-Warning Dashboard")
st.caption(
    "Streamlit demo for a satellite-based coastal flood risk model. "
    "Uses precomputed satellite/terrain features + a trained ML model to generate risk maps and alerts."
)

# -----------------------------
# Load data/model
# -----------------------------
if not FEATURES_PATH.exists():
    st.error(
        "Missing `data/features.csv`.\n\n"
        "Create it using `python make_sample_data.py` (demo) or replace it with your real exported features."
    )
    st.stop()

df = load_features(FEATURES_PATH)

if not MODEL_PATH.exists():
    st.warning(
        "Missing `data/model.joblib`. The app can still show the data, but predictions won't run.\n\n"
        "Create it using `python train_model.py` (demo) or replace it with your real trained model."
    )
    model = None
    feature_cols = []
else:
    model, feature_cols = load_model(MODEL_PATH)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

if "location" in df.columns:
    locations = sorted([x for x in df["location"].dropna().unique()])
    location = st.sidebar.selectbox("Location", locations)
    d = df[df["location"] == location].copy()
else:
    st.sidebar.info("No `location` column found. Showing all points.")
    d = df.copy()
    location = "All"

if "date" in d.columns and d["date"].notna().any():
    min_d = d["date"].min().date()
    max_d = d["date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_d, max_d))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        d = d[(d["date"] >= start) & (d["date"] <= end)]
else:
    st.sidebar.info("No usable `date` column found. Skipping date filter.")

high_threshold = st.sidebar.slider("High-risk threshold", 0.10, 0.90, 0.60, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Demo alert message")
st.sidebar.write("If any populated zone has HIGH risk, issue an alert (simulated).")

# -----------------------------
# Predictions
# -----------------------------
required_geo = {"lat", "lon"}
if not required_geo.issubset(set(d.columns)):
    st.error("Your CSV must include `lat` and `lon` columns for mapping.")
    st.stop()

if d.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

if model is not None:
    missing = [c for c in feature_cols if c not in d.columns]
    if missing:
        st.error(f"Model expects missing feature columns: {missing}")
        st.stop()

    X = d[feature_cols]
    d["flood_prob"] = compute_probs(model, X)
else:
    # If no model, show a simple heuristic score (so the app still demonstrates)
    needed = ["rain_3day", "rain_7day", "wave_h", "elevation", "dist_coast", "slope"]
    if all(c in d.columns for c in needed):
        score = (
            0.015 * d["rain_3day"]
            + 0.010 * d["rain_7day"]
            + 0.40 * d["wave_h"]
            - 0.030 * d["elevation"]
            - 0.020 * d["dist_coast"]
            - 0.015 * d["slope"]
        )
        d["flood_prob"] = 1 / (1 + np.exp(-score))
        st.info("Using heuristic risk score because `data/model.joblib` is missing.")
    else:
        st.error("No model found and not enough columns to compute a heuristic risk score.")
        st.stop()

# Risk summaries
max_prob = float(d["flood_prob"].max())
mean_prob = float(d["flood_prob"].mean())
overall = risk_label(max_prob, high_threshold)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall risk (max)", overall)
c2.metric("Max probability", f"{max_prob:.2f}")
c3.metric("Average probability", f"{mean_prob:.2f}")
c4.metric("Points evaluated", f"{len(d):,}")

# Alert box
if overall == "HIGH":
    st.error(f"🚨 ALERT: HIGH coastal flooding risk detected in {location}. Prepare response actions.")
elif overall == "MEDIUM":
    st.warning(f"⚠️ WATCH: MEDIUM coastal flooding risk in {location}. Monitor conditions and stay ready.")
else:
    st.success(f"✅ LOW risk in {location} for selected period.")

# -----------------------------
# Map (PyDeck)
# -----------------------------
st.subheader("🗺️ Flood Risk Map")

# Add colors and sizes
d = d.copy()
d["color"] = d["flood_prob"].apply(lambda p: color_rgb(float(p), high_threshold))
d["radius"] = d["flood_prob"].apply(lambda p: 40 + 120 * float(p))  # meters-ish

center_lat = float(d["lat"].mean())
center_lon = float(d["lon"].mean())

layer = pdk.Layer(
    "ScatterplotLayer",
    data=d,
    get_position=["lon", "lat"],
    get_fill_color="color",
    get_radius="radius",
    pickable=True,
    auto_highlight=True,
    opacity=0.75,
)

tooltip_fields = []
for col in ["location", "date"] + [c for c in d.columns if c in ("rain_3day", "rain_7day", "wave_h", "elevation", "slope", "dist_coast")]:
    if col in d.columns:
        tooltip_fields.append(f"<b>{col}:</b> {{{col}}}<br/>")
tooltip_fields.append("<b>flood_prob:</b> {flood_prob}")

tooltip = {"html": "".join(tooltip_fields), "style": {"backgroundColor": "white", "color": "black"}}

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.5, pitch=0),
    layers=[layer],
    tooltip=tooltip,
)

st.pydeck_chart(deck, use_container_width=True)

# -----------------------------
# Data table + Download
# -----------------------------
st.subheader("📄 Predictions Table")
show_cols = [c for c in ["location", "date", "lat", "lon"] if c in d.columns] + \
            [c for c in ["rain_3day", "rain_7day", "wave_h", "elevation", "slope", "dist_coast"] if c in d.columns] + \
            ["flood_prob"]

st.dataframe(d[show_cols].sort_values("flood_prob", ascending=False).head(500), use_container_width=True)

csv_bytes = d[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download predictions (CSV)", csv_bytes, file_name="flood_predictions.csv", mime="text/csv")

st.markdown("---")
st.caption(
    "Replace `data/features.csv` with real satellite-derived features (e.g., GPM IMERG rainfall, Sentinel-1 flood labels, DEM elevation). "
    "Then retrain the model and redeploy."
)
