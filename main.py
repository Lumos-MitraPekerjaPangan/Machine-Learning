import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from prophet import Prophet
import seaborn as sns
import joblib  # For model serialization
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Move import to the top

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Function to generate dummy time series data
def generate_dummy_data(start_date='2018-01-01', periods=60):
    """
    Generate dummy monthly data with trend and seasonality
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq='M')
    
    # Create trend component (increasing trend)
    trend = np.linspace(100, 200, periods)
    
    # Create seasonality component (yearly pattern)
    seasonality = 30 * np.sin(np.linspace(0, 2 * np.pi * (periods/12), periods))
    
    # Add some noise
    noise = np.random.normal(0, 10, periods)
    
    # Combine components
    values = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    return df

# Function to generate and save market data as separate JSON files
def generate_market_data(start_date='2018-01-01', periods=60):
    """
    Generate dummy monthly data for supply, demand, and average price
    and save them as separate JSON files
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq='M')
    dates_str = dates.strftime('%Y-%m-%d').tolist()
    
    # Supply data - base with increasing trend and summer seasonality
    supply_base = np.linspace(1000, 1500, periods)
    supply_seasonality = 200 * np.sin(np.linspace(0, 2 * np.pi * (periods/12), periods))
    supply_noise = np.random.normal(0, 50, periods)
    supply_values = np.round(supply_base + supply_seasonality + supply_noise, 2)
    
    # Demand data - steeper trend, winter seasonality
    demand_base = np.linspace(900, 1600, periods)
    demand_seasonality = 300 * np.sin(np.linspace(np.pi, 3 * np.pi * (periods/12), periods))
    demand_noise = np.random.normal(0, 70, periods)
    demand_values = np.round(demand_base + demand_seasonality + demand_noise, 2)
    
    # Price data - based on supply-demand dynamics (in Rupiah)
    base_price = 750000  # 50 USD ≈ 750,000 IDR
    price_sensitivity = 225000  # 15 USD ≈ 225,000 IDR
    price_noise = np.random.normal(0, 75000, periods)  # 5 USD ≈ 75,000 IDR
    price_values = np.round(base_price + price_sensitivity * (demand_values/supply_values - 1) + price_noise, 2)
    
    # Create and save supply.json
    supply_data = {
        "metric": "supply",
        "unit": "units",
        "data": [{"date": date, "value": float(value)} for date, value in zip(dates_str, supply_values)]
    }
    
    with open('supply.json', 'w') as f:
        json.dump(supply_data, f, indent=4)
    
    # Create and save demand.json
    demand_data = {
        "metric": "demand",
        "unit": "units",
        "data": [{"date": date, "value": float(value)} for date, value in zip(dates_str, demand_values)]
    }
    
    with open('demand.json', 'w') as f:
        json.dump(demand_data, f, indent=4)
    
    # Create and save average-price.json
    price_data = {
        "metric": "average price",
        "unit": "IDR",  # Changed from USD to IDR
        "data": [{"date": date, "value": float(value)} for date, value in zip(dates_str, price_values)]
    }
    
    with open('average-price.json', 'w') as f:
        json.dump(price_data, f, indent=4)
    
    print(f"Market data files created: supply.json, demand.json, average-price.json")
    
    # Return the dataframes for model training
    supply_df = pd.DataFrame({
        'ds': dates,
        'y': supply_values
    })
    
    demand_df = pd.DataFrame({
        'ds': dates,
        'y': demand_values
    })
    
    price_df = pd.DataFrame({
        'ds': dates,
        'y': price_values
    })
    
    return {
        'supply': supply_df,
        'demand': demand_df,
        'price': price_df
    }

# Generate dummy data
data = generate_dummy_data(start_date='2018-01-01', periods=60)

# Generate and save market data, and get dataframes for training
market_data = generate_market_data(start_date='2018-01-01', periods=60)

# Train models for each dataset
print("\nTraining specialized models...")

# Function to train and save a Prophet model
def train_save_model(df, name):
    # Split into train/test
    train_df = df.iloc[:-6].copy()
    test_df = df.iloc[-6:].copy()
    
    # Create and train model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        interval_width=0.95
    )
    model.fit(train_df)
    
    # Save model
    model_path = f'../Backend/prophet_model_{name}.joblib'
    joblib.dump(model, model_path)
    print(f"Model for {name} saved to {model_path}")
    
    # Evaluate model
    future = model.make_future_dataframe(periods=6, freq='ME')
    forecast = model.predict(future)
    
    # Get test predictions
    test_preds = forecast.iloc[-6:][['yhat']].reset_index(drop=True)
    test_actuals = test_df[['y']].reset_index(drop=True)
    
    mae = mean_absolute_error(test_actuals, test_preds)
    rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    
    print(f"Model for {name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Visualize
    plt.figure(figsize=(14, 8))
    plt.plot(df['ds'], df['y'], 'o-', color='blue', label=f'Historical {name} Data')
    plt.plot(forecast['ds'], forecast['yhat'], 'r--', label='Forecast')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='red', alpha=0.2,
        label='95% Confidence Interval'
    )
    plt.title(f'{name.capitalize()} Forecast Model', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(f'{name.capitalize()} Value', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{name}_model_forecast.png')
    
    return model, forecast

# Train and save individual models
print("\nTraining supply model...")
supply_model, supply_forecast = train_save_model(market_data['supply'], 'supply')

print("\nTraining demand model...")
demand_model, demand_forecast = train_save_model(market_data['demand'], 'demand')

print("\nTraining price model...")
price_model, price_forecast = train_save_model(market_data['price'], 'price')

print("\nAll models trained and saved successfully!")

# Display summary of all models
print("\n======= Model Evaluation Summary =======")
for data_type in ['supply', 'demand', 'price']:
    # Load the data for this model
    df = market_data[data_type]
    train_df = df.iloc[:-6]
    test_df = df.iloc[-6:]
    
    # Load forecasts for this model
    forecast_var = locals()[f"{data_type}_forecast"]
    
    # Get test predictions
    test_preds = forecast_var.iloc[-6:][['yhat']].reset_index(drop=True)
    test_actuals = test_df[['y']].reset_index(drop=True)
    
    # Calculate metrics
    mae = mean_absolute_error(test_actuals, test_preds)
    rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    
    print(f"{data_type.capitalize()} Model:")
    print(f"  - MAE: {mae:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - Model saved to: ../Backend/prophet_model_{data_type}.joblib")
    print()
