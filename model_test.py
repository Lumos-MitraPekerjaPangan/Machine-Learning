import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Import train_save_model from main.py
from main import train_save_model

# Helper to load JSON data for a commodity and metric
def load_json_data(commodity, metric):
    filepath = f"{commodity}/{metric}.json"
    with open(filepath, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data["data"])
    df["ds"] = pd.to_datetime(df["date"])
    df["y"] = df["value"]
    return df[["ds", "y"]]


# Test: train then predict for each commodity and metric
def compare_predictions(commodity, metric):
    print(f"\n=== {commodity.upper()} - {metric.upper()} ===")
    df = load_json_data(commodity, metric)
    # Always train before predict
    model, forecast = train_save_model(df, metric)
    # Use all but last 6 months for training, last 6 for test
    train_df = df.iloc[:-6]
    test_df = df.iloc[-6:]
    # Predict for the test period
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)
    forecast_test = forecast.tail(6).reset_index(drop=True)
    test_actuals = test_df["y"].reset_index(drop=True)
    test_preds = forecast_test["yhat"]

    # Compare with the same months from the previous year (year-over-year)
    if len(df) >= 18:
        last_year_df = df.iloc[-18:-12].reset_index(drop=True)
        print("\nComparison with previous year's same months:")
        print("Month     | Last Year | This Year | Predicted")
        for i in range(6):
            month = test_df["ds"].iloc[i].strftime("%Y-%m")
            last_year_val = last_year_df["y"].iloc[i] if i < len(last_year_df) else np.nan
            print(f"{month} | {last_year_val:9.2f} | {test_actuals[i]:9.2f} | {test_preds.iloc[i]:9.2f}")
    else:
        print("\nNot enough data for year-over-year comparison.")

    # Metrics
    mae = mean_absolute_error(test_actuals, test_preds)
    rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    print(f"\nMAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(df["ds"], df["y"], label="Actual (All Years)", color="blue")
    plt.plot(forecast["ds"], forecast["yhat"], label="Model Forecast", color="green")
    plt.scatter(test_df["ds"], test_actuals, color="red", label="Test Actuals")
    plt.scatter(test_df["ds"], test_preds, color="orange", label="Test Predictions")
    if len(df) >= 18:
        plt.scatter(last_year_df["ds"], last_year_df["y"], color="purple", label="Last Year Actuals")
    plt.title(f"{commodity.capitalize()} - {metric.replace('_',' ').capitalize()} Prediction vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    # Save plot to a neat folder
    plot_dir = os.path.join("plots", commodity)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{metric}_prediction_vs_actual.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    commodities = ["apel", "beras", "ikan"]
    metrics = ["supply", "demand", "price", "bulk_price"]
    for commodity in commodities:
        for metric in metrics:
            compare_predictions(commodity, metric)
