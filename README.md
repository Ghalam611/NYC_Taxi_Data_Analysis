# NYC Taxi Data Analysis

## Overview

A data science project analyzing NYC Yellow Taxi trip records to extract operational insights and build predictive models for demand and fares. The work covers data cleaning, geospatial clustering, trip pattern discovery, and machine learning.

## Objectives

- Clean and preprocess TLC trip data (outliers, time & location features).
- Identify pickup/drop-off **hotspots** using geospatial clustering.
- Discover common **trip patterns and routes** via trip clustering.
- Forecast taxi **demand** using time series modeling.
- Predict **fares** using supervised machine learning.
- Provide an **interactive dashboard** for visualization and predictions.

## Repository Structure

| File | Description |
| --- | --- |
| `nyc_taxi_clean.py` | Data ingestion, cleaning, and feature engineering |
| `dae.ipynb` | Exploratory Data Analysis (EDA) |
| `cluster.py` | Geospatial clustering logic |
| `dbscan_pickups.pkl` | Serialized pickup clustering results |
| `trip-c.ipynb` | Trip clustering and route pattern analysis |
| `demand_forecasting.py` | Demand forecasting model training/evaluation |
| `demand_forecasting_best_model.pkl` | Best demand forecasting model |
| `fare_prediction.py` | Fare prediction model training/evaluation |
| `app.py` | Interactive application/dashboard |

## Key Insights

- Taxi demand shows clear **temporal patterns**, with peaks during rush hours and weekends.
- **Geospatial clustering** reveals consistent pickup/drop-off hotspots in central Manhattan and major transit areas.
- Trip clustering identifies **recurring travel patterns**, distinguishing short urban rides from long cross-zone trips.
- Fare prediction models demonstrate a strong relationship between **distance, duration, and time of day**, achieving reliable estimation performance.

## Data Source

The project uses publicly available Yellow Taxi trip record data provided by the **NYC Taxi & Limousine Commission (TLC).**

**Kaggle Dataset:** [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data)

## Technologies

- **Python**: Pandas, NumPy
- **ML**: Scikit-learn, time series libraries
- **Geospatial**: GeoPandas, Folium
- **Visualization**: Matplotlib, Seaborn, Plotly
- **App**: FastAPI

## Contributors

- [Ghalam611](https://github.com/Ghalam611)
- [Dhay-Alharbi](https://github.com/Dhay-Alharbi)
- [Raghad-Alfarsi](https://github.com/Raghad-Alfarsi)
- [Almutlak-dev](https://github.com/Almutlak-dev)
- [Ali-Arishi](https://github.com/Ali-Arishi)
