import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

np.random.seed(7)

OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)

# Rough bounding boxes (not perfect—just for demo)
AREAS = {
    "Mombasa":  {"lat": (-4.20, -3.90), "lon": (39.55, 39.80)},
    "Kilifi":   {"lat": (-3.95, -3.45), "lon": (39.55, 39.95)},
    "Malindi":  {"lat": (-3.40, -2.95), "lon": (39.90, 40.15)},
    "Lamu":     {"lat": (-2.45, -1.95), "lon": (40.75, 41.10)},
}

def synthesize_points(location: str, n: int = 500):
    box = AREAS[location]
    lat = np.random.uniform(*box["lat"], size=n)
    lon = np.random.uniform(*box["lon"], size=n)

    # Fake "satellite-derived" indicators (you will replace with real GPM/S1/ocean features)
    rain_3day = np.random.gamma(shape=2.0, scale=15.0, size=n)       # mm
    rain_7day = rain_3day + np.random.gamma(shape=2.0, scale=12.0, size=n)
    wave_h = np.random.uniform(0.5, 3.0, size=n)                     # meters

    # Terrain / vulnerability-like features
    elevation = np.clip(np.random.normal(loc=18, scale=12, size=n), 0, 120)  # meters
    slope = np.clip(np.random.normal(loc=2.5, scale=2.0, size=n), 0, 25)     # degrees
    dist_coast = np.clip(np.random.exponential(scale=2.0, size=n), 0, 25)    # km

    # Create dates (last 120 days)
    start = datetime.now().date() - timedelta(days=120)
    dates = [start + timedelta(days=int(x)) for x in np.random.uniform(0, 120, size=n)]

    df = pd.DataFrame({
        "location": location,
        "date": dates,
        "lat": lat,
        "lon": lon,
        "rain_3day": np.round(rain_3day, 2),
        "rain_7day": np.round(rain_7day, 2),
        "wave_h": np.round(wave_h, 2),
        "elevation": np.round(elevation, 2),
        "slope": np.round(slope, 2),
        "dist_coast": np.round(dist_coast, 2),
    })

    # Synthetic label for training demo:
    # higher risk if high rain + high waves + low elevation + close to coast
    score = (
        0.015 * df["rain_3day"]
        + 0.010 * df["rain_7day"]
        + 0.40  * df["wave_h"]
        - 0.030 * df["elevation"]
        - 0.020 * df["dist_coast"]
        - 0.015 * df["slope"]
    )
    prob = 1 / (1 + np.exp(-score))
    df["flooded"] = (np.random.rand(n) < prob).astype(int)

    return df

all_df = pd.concat([synthesize_points(k, n=600) for k in AREAS.keys()], ignore_index=True)
all_df.to_csv(OUT / "features.csv", index=False)
print("✅ Created data/features.csv with demo features + flooded labels.")
