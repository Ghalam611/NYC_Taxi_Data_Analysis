import os
import io
import base64

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN


app = FastAPI(title="🚕 The Future Of Taxis")

# ======================
# FILE PATHS
# ======================
DATA_CSV = "yellow_tripdata_2016-03.csv"          
KMEANS_PATH = "kmeans_pickups.pkl"
FORECAST_PATH = "demand_forecasting_best_model.pkl"
FARE_MODEL_PATH = "fare_prediction_model.pkl"


FEATURES = ["pickup_hour", "day_of_week", "is_weekend"]

# ======================
# LOAD MODELS SAFELY
# ======================
KMEANS_OK = False
KMEANS_ERROR = ""
kmeans_model = None

try:
    if not os.path.exists(KMEANS_PATH):
        raise FileNotFoundError(f"KMeans model not found: {KMEANS_PATH}")
    kmeans_model = joblib.load(KMEANS_PATH)
    KMEANS_OK = True
except Exception as e:
    KMEANS_OK = False
    KMEANS_ERROR = str(e)

FORECAST_OK = False
FORECAST_ERROR = ""
best_model = None
scaler = None

try:
    if not os.path.exists(FORECAST_PATH):
        raise FileNotFoundError(f"Forecast model not found: {FORECAST_PATH}")
    model_data = joblib.load(FORECAST_PATH)
    best_model = model_data["model"]
    scaler = model_data["scaler"]
    FORECAST_OK = True
except Exception as e:
    FORECAST_OK = False
    FORECAST_ERROR = str(e)

# ======================
# LOAD FARE PREDICTION MODEL
# ======================
FARE_OK = False
FARE_ERROR = ""
fare_model = None
fare_scaler = None

try:
    if not os.path.exists(FARE_MODEL_PATH):
        raise FileNotFoundError(f"Fare model not found: {FARE_MODEL_PATH}")

    data = joblib.load(FARE_MODEL_PATH)

    if isinstance(data, dict):
        fare_model = data["model"]
        fare_scaler = data.get("scaler", None)
    else:
        fare_model = data
        fare_scaler = None

    FARE_OK = True
except Exception as e:
    FARE_OK = False
    FARE_ERROR = str(e)

# ======================
# CSS
# ======================
CSS_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }

body { font-family: 'Poppins', sans-serif; background: #ffffff; color: #000; min-height: 100vh; }

nav {
  background: #FFD700;
  padding: 15px 0;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  position: sticky; top: 0; z-index: 1000;
}

nav .nav-content {
  max-width: 1200px;
  margin: auto;
  display: flex; justify-content: space-between; align-items: center;
  padding: 0 20px;
}

nav a { color: #000; text-decoration: none; font-weight: 800; margin-left: 15px; }

.container { max-width: 1100px; margin: 40px auto; padding: 0 20px; }

.hero-card {
  background: #f2f2f2;
  padding: 42px;
  border-radius: 20px;
  text-align: center;
  margin-bottom: 30px;
}

.card {
  background: #f2f2f2;
  padding: 30px;
  border-radius: 20px;
  margin-bottom: 30px;
}

h1, h2 { margin-bottom: 15px; }

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 20px;
}

.feature-item {
  background: #fff;
  padding: 22px;
  border-radius: 15px;
  text-align: center;
  transition: 0.2s;
  border: 2px solid transparent;
}

.feature-item:hover { border-color: #FFD700; }

.feature-item a {
  color: inherit;
  text-decoration: none;
  font-weight: 800;
  display: inline-block;
  width: 100%;
}

.form-group { max-width: 420px; margin-bottom: 18px; }

input[type="date"] {
  width: 100%;
  padding: 12px;
  border-radius: 12px;
  border: 1px solid #ccc;
  background: #fff;
}

button {
  background: #FFD700;
  border: none;
  padding: 12px 30px;
  border-radius: 30px;
  font-weight: 900;
  cursor: pointer;
}

.notice {
  background: #fff3cd;
  border: 1px solid #ffe69c;
  padding: 14px;
  border-radius: 12px;
}

.error {
  background: #f8d7da;
  border: 1px solid #f1aeb5;
  padding: 14px;
  border-radius: 12px;
}

.stats-row {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-top: 18px;
}

.stat-box {
  flex: 1;
  min-width: 180px;
  background: #fff;
  padding: 18px;
  border-radius: 15px;
  text-align: center;
}

.stat-label { font-weight: 800; opacity: 0.85; }
.stat-value { font-size: 26px; font-weight: 900; margin-top: 6px; }
"""

def render_page(title: str, content: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title} | 🚕 The Future Of Taxis</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>{CSS_STYLE}</style>
</head>
<body>

<nav>
  <div class="nav-content">
    <a href="/">🚕 The Future Of Taxis</a>
    <div>
      <a href="/">Home</a>
    </div>
  </div>
</nav>

<div class="container">
{content}
</div>

</body>
</html>"""


# ======================
# HELPER: CLEAN DATA FOR TRIP CLUSTERS
# ======================
def clean_for_trip_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
   
    """
    df = df.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # duration بين 1 دقيقة و 12 ساعة
    df = df[(df["duration_min"] > 1) & (df["duration_min"] < 720)]

    # مسافة منطقية
    df = df[(df["trip_distance"] > 1) & (df["trip_distance"] < 23)]

    # السرعة (ميل/ساعة)
    df["Speed"] = df["trip_distance"] / (df["duration_min"] / 60.0)
    df = df[(df["Speed"] >= 0.5) & (df["Speed"] <= 80)]

    # المبلغ الكلي
    df = df[(df["total_amount"] > 1) & (df["total_amount"] < 200)]

    return df


# ======================
# HOME
# ======================
@app.get("/", response_class=HTMLResponse)
async def home():
    banners = ""

    if not KMEANS_OK:
        banners += f"""
        <div class="card"><div class="error">
          <b>KMeans model not loaded.</b><br><small>{KMEANS_ERROR}</small>
        </div></div>
        """

    if not FORECAST_OK:
        banners += f"""
        <div class="card"><div class="error">
          <b>Forecast model not loaded.</b><br><small>{FORECAST_ERROR}</small>
        </div></div>
        """

    content = f"""
    <div class="hero-card">
      <h1>🚕 Welcome to the Future of Taxis</h1>
      <p>AI-powered demand forecasting & intelligent trip clustering</p>
    </div>

    {banners}

    <div class="card">
      <h2>Features</h2>
      <div class="features-grid">
        <div class="feature-item"><a href="/dashboard">🗺️ Pickup & Trip Clusters Dashboard</a></div>
        <div class="feature-item"><a href="/predict">⏰ Predict Taxi Demand</a></div>
        <div class="feature-item">
  <a href="/fare">💵 Fare Prediction</a>
</div>

      </div>
    </div>
    """
    return HTMLResponse(render_page("Home", content))


# ======================
# DASHBOARD (PICKUP + TRIP CLUSTERS)
# ======================
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        if not os.path.exists(DATA_CSV):
            raise FileNotFoundError(f"CSV not found: {DATA_CSV}")

        if not os.path.exists(KMEANS_PATH):
            raise FileNotFoundError(f"KMeans pkl not found: {KMEANS_PATH}")

        # نقرأ الأعمدة المهمة مرّة وحدة
        df = pd.read_csv(
            DATA_CSV,
            usecols=[
                "pickup_latitude",
                "pickup_longitude",
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "trip_distance",
                "fare_amount",
                "total_amount",
            ],
        ).dropna()

        # فلترة داخل حدود نيويورك
        df = df[
            df["pickup_latitude"].between(40.4774, 40.9176)
            & df["pickup_longitude"].between(-74.2591, -73.7004)
        ]

        if len(df) == 0:
            raise RuntimeError("No points after NYC filtering (check bounds or CSV values).")

        # ================================
        # 1) Pickup Location Clusters – K-Means
        # ================================
        MAP_POINTS = 20000
        df_map = df.sample(n=min(len(df), MAP_POINTS), random_state=42)

        coords = df_map[["pickup_latitude", "pickup_longitude"]].astype(float).to_numpy()

        kmeans_pickup = joblib.load(KMEANS_PATH)
        df_map["cluster_kmeans"] = kmeans_pickup.predict(coords)

        center_lat = float(df_map["pickup_latitude"].mean())
        center_lon = float(df_map["pickup_longitude"].mean())

        fig_map_k = px.scatter_mapbox(
            df_map,
            lat="pickup_latitude",
            lon="pickup_longitude",
            color="cluster_kmeans",
            zoom=9,
            height=650,
            center={"lat": center_lat, "lon": center_lon},
            title="NYC Pickup Location Clusters – K-Means",
        )
        fig_map_k.update_layout(mapbox_style="open-street-map")

        map_kmeans_html = fig_map_k.to_html(full_html=False, include_plotlyjs="cdn")

        # ================================
        # 2) Pickup Location Clusters – DBSCAN
        # ================================
        # نستخدم نفس العينة df_map عشان يقارنون نفس النقاط
        coords_rad = np.radians(
            df_map[["pickup_latitude", "pickup_longitude"]].to_numpy()
        )

        db = DBSCAN(
            eps=0.5 / 6371,      
            min_samples=50,
            metric="haversine",
            algorithm="ball_tree",
        )

        labels_db = db.fit_predict(coords_rad)
        df_map["cluster_dbscan"] = labels_db  # -1 = noise

        fig_map_db = px.scatter_mapbox(
            df_map,
            lat="pickup_latitude",
            lon="pickup_longitude",
            color="cluster_dbscan",
            zoom=9,
            height=650,
            center={"lat": center_lat, "lon": center_lon},
            title="NYC Pickup Location Clusters – DBSCAN",
        )
        fig_map_db.update_layout(mapbox_style="open-street-map")

        map_dbscan_html = fig_map_db.to_html(full_html=False, include_plotlyjs=False)

        # ================================
        # 3) Trip Clusters (PCA) 
        # ================================
        df_trip = clean_for_trip_clusters(df)

        TRIP_POINTS = 200_000
        df_trip_sample = df_trip.sample(
            n=min(len(df_trip), TRIP_POINTS), random_state=42
        )

        X = df_trip_sample[["trip_distance", "fare_amount", "duration_min", "Speed"]]

        scaler_tc = StandardScaler()
        X_scaled = scaler_tc.fit_transform(X)

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X_scaled)

        df_trip_sample["PCA1"] = pca_data[:, 0]
        df_trip_sample["PCA2"] = pca_data[:, 1]

        optimal_k = 8
        kmeans_trip = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df_trip_sample["cluster_trip"] = kmeans_trip.fit_predict(pca_data)

        cluster_summary = df_trip_sample.groupby("cluster_trip")[
            ["trip_distance", "fare_amount", "duration_min", "Speed"]
        ].mean()

        cluster_summary["count"] = df_trip_sample.groupby("cluster_trip").size()
        cluster_summary["count_pct"] = (
            cluster_summary["count"] / len(df_trip_sample) * 100
        )

        cluster_summary = cluster_summary.reset_index().rename(
            columns={"cluster_trip": "cluster"}
        )

        df_vis = df_trip_sample.sample(
            n=min(len(df_trip_sample), 20_000), random_state=1
        )

        fig_pca = px.scatter(
            df_vis,
            x="PCA1",
            y="PCA2",
            color="cluster_trip",
            title=f"Trip Clusters in PCA Space – K-Means (K={optimal_k})",
            hover_data=["trip_distance", "fare_amount", "duration_min", "Speed"],
        )

        melted = cluster_summary.melt(
            id_vars="cluster",
            value_vars=["trip_distance", "fare_amount", "duration_min", "Speed"],
            var_name="Feature",
            value_name="Average",
        )

        fig_features = px.bar(
            melted,
            x="cluster",
            y="Average",
            color="Feature",
            barmode="group",
            title="Average Trip Features per Cluster (K-Means)",
        )

        trip_pca_html = fig_pca.to_html(full_html=False, include_plotlyjs=False)
        trip_features_html = fig_features.to_html(full_html=False, include_plotlyjs=False)

        # ================================
        
        # ================================
        content = f"""
        <div class="hero-card">
            <h1>🗺️ Pickup & Trip Clusters</h1>
            <p>Compare K-Means and DBSCAN on pickup locations, plus trip behavior clusters.</p>
        </div>

        <div class="card">
            <h2>Pickup Location Clusters – K-Means</h2>
            {map_kmeans_html}
            <p style="margin-top:10px;"><b>Showing:</b> {len(df_map):,} pickup points.</p>
        </div>

        <div class="card">
            <h2>Pickup Location Clusters – DBSCAN</h2>
            <p style="margin-bottom:8px;">DBSCAN detects dense regions and marks noise as cluster -1.</p>
            {map_dbscan_html}
        </div>

        <div class="card">
            <h2>Trip Clusters in PCA Space – K-Means</h2>
            <p>Each point is a trip projected using PCA (distance, fare, duration, speed).</p>
            {trip_pca_html}
        </div>

        <div class="card">
            <h2>Trip Cluster Characteristics – K-Means</h2>
            <p>Average trip distance, fare, duration, and speed for each cluster.</p>
            {trip_features_html}
        </div>
        """
        return HTMLResponse(render_page("Dashboard", content))

    except Exception as e:
        content = f"""
        <div class="hero-card"><h1>⚠️ Dashboard Error</h1></div>
        <div class="card">
            <div style="background:#f8d7da;border:1px solid #f1aeb5;padding:14px;border-radius:12px;">
                <b>Exception:</b>
                <pre style="white-space:pre-wrap;">{repr(e)}</pre>
            </div>
        </div>
        """
        return HTMLResponse(render_page("Dashboard", content))


# ======================
# PREDICT (GET)
# ======================
@app.get("/predict", response_class=HTMLResponse)
async def predict_get():
    if not FORECAST_OK:
        content = f"""
        <div class="hero-card"><h1>📅 Predict Taxi Demand</h1></div>
        <div class="card"><div class="error">
          <b>Cannot predict because forecast model isn't loaded.</b><br>
          <small>{FORECAST_ERROR}</small>
        </div></div>
        """
        return HTMLResponse(render_page("Predict", content))

    content = """
    <div class="hero-card"><h1>📅 Predict Taxi Demand</h1></div>
    <div class="card">
      <form method="post">
        <div class="form-group">
          <label>Select Date</label>
          <input type="date" name="date" required>
        </div>
        <button type="submit">Generate Prediction</button>
      </form>
    </div>
    """
    return HTMLResponse(render_page("Predict", content))

# ======================
# PREDICT (POST)
# ======================
@app.post("/predict", response_class=HTMLResponse)
async def predict_post(date: str = Form(...)):
    if not FORECAST_OK:
        content = f"""
        <div class="hero-card"><h1>📅 Predict Taxi Demand</h1></div>
        <div class="card"><div class="error">
          <b>Cannot predict because forecast model isn't loaded.</b><br>
          <small>{FORECAST_ERROR}</small>
        </div></div>
        """
        return HTMLResponse(render_page("Predict", content))

    try:
        selected_date = pd.to_datetime(date)

        df_pred = pd.DataFrame({
            "pickup_hour": list(range(24)),
            "day_of_week": [selected_date.dayofweek] * 24,
            "is_weekend": [1 if selected_date.dayofweek in [5, 6] else 0] * 24
        })

        X_scaled = scaler.transform(df_pred[FEATURES])
        preds = np.maximum(best_model.predict(X_scaled), 0)

        max_pred = float(np.max(preds))
        avg_pred = float(np.mean(preds))
        min_pred = float(np.min(preds))
        peak_hours = [i for i, v in enumerate(preds) if float(v) == max_pred]
        peak_text = ", ".join(f"{h:02d}:00-{h:02d}:59" for h in peak_hours)

        colors = ["#D9D9D9"] * 24
        for h in peak_hours:
            colors[h] = "#FFD700"

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(24), preds, color=colors, edgecolor="#333")
        ax.set_title("Predicted Taxi Demand per Hour")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Demand")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode("utf-8")

        content = f"""
        <div class="hero-card"><h1>Prediction Result</h1></div>

        <div class="card">
          <div class="notice">
            <b>Date:</b> {selected_date.date()}<br>
            <b>Peak hours:</b> {peak_text}
          </div>

          <div class="stats-row">
            <div class="stat-box">
              <div class="stat-label">Peak Demand</div>
              <div class="stat-value">{int(max_pred)}</div>
            </div>
            <div class="stat-box">
              <div class="stat-label">Average Demand</div>
              <div class="stat-value">{int(avg_pred)}</div>
            </div>
            <div class="stat-box">
              <div class="stat-label">Lowest Demand</div>
              <div class="stat-value">{int(min_pred)}</div>
            </div>
          </div>
        </div>

        <div class="card">
          <img src="data:image/png;base64,{img}" style="width:100%; border-radius:12px;">
        </div>
        """
        return HTMLResponse(render_page("Result", content))

    except Exception as e:
        content = f"""
        <div class="hero-card"><h1>⚠️ Error</h1></div>
        <div class="card"><div class="error">
          <b>Exception:</b><br><small>{str(e)}</small>
        </div></div>
        """
        return HTMLResponse(render_page("Error", content))


# ======================
# FARE PREDICTION (GET)
# ======================
@app.get("/fare", response_class=HTMLResponse)
async def fare_get():
    if not FARE_OK:
        content = f"""
        <div class="card error">
          <b>Fare model not loaded.</b><br>
          <small>{FARE_ERROR}</small>
        </div>
        """
        return HTMLResponse(render_page("Fare Prediction", content))

    content = """
    <div class="hero-card">
      <h1>💵 Fare Prediction</h1>
      <p>Predict taxi fare using distance + date & time</p>
    </div>

    <div class="card">
      <form method="post">
        <div class="form-group">
          <label>Trip Distance (miles)</label>
          <input name="trip_distance" type="number" step="0.01" min="0" required>
        </div>

        <div class="form-group">
          <label>Pickup Date & Time</label>
          <input type="datetime-local" name="pickup_datetime" required>
        </div>

        <button type="submit">Predict Fare</button>
      </form>
    </div>
    """
    return HTMLResponse(render_page("Fare Prediction", content))



# ======================
# FARE PREDICTION (POST)
# ======================
@app.post("/fare", response_class=HTMLResponse)
async def fare_post(
    trip_distance: float = Form(...),
    pickup_datetime: str = Form(...)
):
    try:
        dt = pd.to_datetime(pickup_datetime)

        
        is_peak_hour = 1 if (7 <= dt.hour <= 9) or (16 <= dt.hour <= 19) else 0

      
        DEFAULT_PICKUP_LAT = 40.7580
        DEFAULT_PICKUP_LON = -73.9855
        DEFAULT_DROPOFF_LAT = 40.7681
        DEFAULT_DROPOFF_LON = -73.9819

        
        X = pd.DataFrame([{
            "trip_distance": trip_distance,            
            "pickup_longitude": DEFAULT_PICKUP_LON,
            "pickup_latitude": DEFAULT_PICKUP_LAT,
            "dropoff_longitude": DEFAULT_DROPOFF_LON,
            "dropoff_latitude": DEFAULT_DROPOFF_LAT,
            "pickup_hour": dt.hour,
            "pickup_day": dt.day,
            "pickup_month": dt.month,
            "pickup_dayofweek": dt.dayofweek,
            "is_peak_hour": is_peak_hour               
        }])

        
        if fare_scaler is not None:
            X = fare_scaler.transform(X)

        fare = float(fare_model.predict(X)[0])

        content = f"""
        <div class="hero-card"><h1>💵 Fare Prediction Result</h1></div>

        <div class="card">
          <div class="stat-box">
            <div class="stat-label">Predicted Fare</div>
            <div class="stat-value">${fare:.2f}</div>
          </div>

          <div class="notice" style="margin-top:15px;">
            <b>Date:</b> {dt.date()}<br>
            <b>Time:</b> {dt.strftime("%H:%M")}<br>
            <b>Peak hour?</b> {"Yes" if is_peak_hour else "No"}<br>
            <b>Distance unit:</b> miles
          </div>
        </div>
        """
        return HTMLResponse(render_page("Fare Result", content))

    except Exception as e:
        content = f"""
        <div class="card error">
          <b>Error:</b><br>
          <small>{str(e)}</small>
        </div>
        """
        return HTMLResponse(render_page("Error", content))
