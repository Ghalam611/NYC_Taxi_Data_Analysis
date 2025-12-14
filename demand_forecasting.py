import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def save_model(model, scaler, filename="demand_forecasting_best_model.pkl"):
    """Save the trained model and scaler."""
    joblib.dump({"model": model, "scaler": scaler}, filename)
    print(f"Model saved as '{filename}'")


def create_features(df):
    """
    Preprocess dataset:
    - Combine pickup date and time into datetime
    - Extract features: pickup_hour, day_of_week, is_weekend
    - Aggregate hourly demand per day
    """
    # Combine date and time
    df['pickup_datetime'] = pd.to_datetime(
        df['pickup_date'].astype(str) + ' ' + df['pickup_time'].astype(str)
    )

    # Extract features
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Aggregate hourly demand
    demand_df = df.groupby(['pickup_date', 'pickup_hour']).size().reset_index(name='demand')

    # Add day_of_week and is_weekend to aggregated data
    demand_df['day_of_week'] = pd.to_datetime(demand_df['pickup_date']).dt.dayofweek
    demand_df['is_weekend'] = demand_df['day_of_week'].isin([5, 6]).astype(int)

    return demand_df


def plot_model_comparison(results, demand_df):
    """Bar plots for MAE and R² with percentage error."""
    models = list(results.keys())
    mae_values = [results[m]["MAE"] for m in models]
    r2_values = [results[m]["R2"] for m in models]
    avg_demand = demand_df['demand'].mean()
    percentage_errors = [(mae / avg_demand) * 100 for mae in mae_values]

    plt.figure(figsize=(12, 5))

    # MAE Plot
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, mae_values, color='steelblue')
    plt.title("MAE Comparison")
    plt.ylabel("MAE")
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{bar.get_height():.0f}\n({percentage_errors[i]:.2f}%)",
                 ha='center', va='bottom', fontsize=8)

    # R² Plot
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, r2_values, color='steelblue')
    plt.title("R² Comparison")
    plt.ylabel("R² Score")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300)
    plt.show()


def plot_actual_vs_pred(y_true, y_pred, title, filename):
    """Plot actual vs predicted values."""
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(14, 6))
    plt.plot(y_true[:300], label="Actual", linewidth=2)
    plt.plot(y_pred[:300], label="Predicted", linewidth=2)
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("Samples", fontsize=13)
    plt.ylabel("Demand", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()



if __name__ == "__main__":
    # Load Dataset
    df = pd.read_csv(
        "cleaned_data/yellow_tripdata_2016-03_cleaned.csv",
        parse_dates=['pickup_date', 'dropoff_date']
    )

    # Preprocess and aggregate
    demand_df = create_features(df)

    print("-" * 50)
    print(demand_df.head(10))
    print("-" * 50)

    # Features and target
    features = ['pickup_hour', 'day_of_week', 'is_weekend']
    X = demand_df[features]
    y = demand_df["demand"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Define models
    models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest": RandomForestRegressor(n_estimators=120, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42)
    }

    results = {}
    trained_models = {}

    # Train & evaluate
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = np.maximum(model.predict(X_test_s), 0)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "R2": r2}
        trained_models[name] = model

    # Best model selection
    best_model_name = min(results, key=lambda x: results[x]["MAE"])
    best_model = trained_models[best_model_name]

    avg_demand = demand_df['demand'].mean()
    best_mae = results[best_model_name]["MAE"]
    best_r2 = results[best_model_name]["R2"]
    percentage_error = (best_mae / avg_demand) * 100

    print(f"\nBest Model: {best_model_name}")
    print(f"MAE: {best_mae:.2f}")
    print(f"R²: {best_r2:.4f}")
    print(f"Average hourly demand: {avg_demand:.2f}")
    print(f"Approximate percentage error: {percentage_error:.2f}%")

    save_model(best_model, scaler)

    # Plot comparisons
    plot_model_comparison(results, demand_df)

    # Actual vs Predicted plots
    train_pred = np.maximum(best_model.predict(X_train_s), 0)
    test_pred = np.maximum(best_model.predict(X_test_s), 0)

    plot_actual_vs_pred(y_train.values, train_pred, "Training Set: Actual vs Predicted Demand", "train_actual_vs_pred.png")
    plot_actual_vs_pred(y_test.values, test_pred, "Test Set: Actual vs Predicted Demand", "test_actual_vs_pred.png")


