import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from prophet import Prophet
import seaborn as sns
import joblib
import json
import tempfile
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase with your credentials
cred = credentials.Certificate('hackfest-2025-7f1a4-firebase-adminsdk-fbsvc-a86f273b3e.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'hackfest-2025-7f1a4.firebasestorage.app'
})
bucket = storage.bucket()
print("Firebase storage initialized successfully")

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Function to generate and save market data as separate JSON files
def generate_market_data(start_date='2020-01-01', end_date=None):
    """
    Generate dummy monthly data for supply, demand, regular price, and bulk price
    and save them as separate JSON files
    """
    # Set end_date to current month if not provided
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # Convert to timestamps and calculate periods
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    # Calculate number of months between dates
    num_months = (end_ts.year - start_ts.year) * 12 + (end_ts.month - start_ts.month) + 1
    
    # Create date range
    dates = pd.date_range(start=start_ts, periods=num_months, freq='M')
    dates_str = dates.strftime('%Y-%m-%d').tolist()
    
    # Supply data with clear seasonality
    supply_base = np.linspace(1000, 1500, num_months)
    supply_seasonality = 200 * np.sin(np.linspace(0, 2 * np.pi * (num_months/12), num_months))
    supply_noise = np.random.normal(0, 20, num_months)
    supply_values = np.round(supply_base + supply_seasonality + supply_noise, 2)
    
    # Demand data - more regularized pattern with clearer seasonality
    demand_trend = np.linspace(800, 1800, num_months)
    demand_seasonality = 200 * np.sin(np.linspace(0, 2 * np.pi * (num_months/12), num_months))
    demand_noise = np.random.normal(0, 15, num_months)
    demand_values = np.round(demand_trend + demand_seasonality + demand_noise, 2)
    
    # Generate normal price data
    price_trend = np.linspace(700000, 800000, num_months)
    price_seasonality = 30000 * np.sin(np.linspace(0, 2 * np.pi * (num_months/12), num_months))
    price_noise = np.random.normal(0, 10000, num_months)
    price_values = np.round(price_trend + price_seasonality + price_noise, 2)
    
    # Generate bulk price data (with discount compared to normal price)
    bulk_price_base = price_values * 0.85
    bulk_price_seasonality = 15000 * np.sin(np.linspace(0.5, 2.5 * np.pi * (num_months/12), num_months))
    bulk_price_noise = np.random.normal(0, 5000, num_months)
    bulk_price_values = np.round(bulk_price_base + bulk_price_seasonality + bulk_price_noise, 2)
    
    # Create and save JSON files to Firebase
    for name, values in [
        ('supply', supply_values), 
        ('demand', demand_values), 
        ('price', price_values), 
        ('bulk_price', bulk_price_values)
    ]:
        # Create JSON structure
        unit = "IDR" if "price" in name else "units"
        metric = "normal price" if name == "price" else "bulk price" if name == "bulk_price" else name
        
        json_data = {
            "metric": metric,
            "unit": unit,
            "data": [{"date": date, "value": float(value)} for date, value in zip(dates_str, values)]
        }
        
        # Save locally for reference
        filename = f"{name.replace('_', '-')}.json"
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=4)
            
        # Save to Firebase
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            temp_path = tmp.name
            with open(temp_path, 'w') as f:
                json.dump(json_data, f)
            
            # Upload to Firebase Storage
            firebase_path = f'data/{filename}'
            blob = bucket.blob(firebase_path)
            blob.upload_from_filename(temp_path)
        
        # Remove temp file
        os.unlink(temp_path)
        
    print(f"Market data files created and uploaded to Firebase")
    
    # Return the dataframes for model training
    return {
        'supply': pd.DataFrame({'ds': dates, 'y': supply_values}),
        'demand': pd.DataFrame({'ds': dates, 'y': demand_values}),
        'price': pd.DataFrame({'ds': dates, 'y': price_values}),
        'bulk_price': pd.DataFrame({'ds': dates, 'y': bulk_price_values})
    }

# Function to train and save a Prophet model
def train_save_model(df, name):
    # Split into train/test
    train_df = df.iloc[:-6].copy()
    test_df = df.iloc[-6:].copy()
    
    # Create model with optimized parameters for each type
    if name == 'supply':
        model = Prophet(
            yearly_seasonality=10,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95,
            mcmc_samples=0,
            uncertainty_samples=100
        )
    elif name == 'demand':
        model = Prophet(
            yearly_seasonality=10,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            mcmc_samples=0,
            uncertainty_samples=100
        )
    else:  # price or bulk_price model
        model = Prophet(
            yearly_seasonality=10,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.01,
            seasonality_mode='multiplicative',
            mcmc_samples=0,
            uncertainty_samples=100
        )
    
    # Fit model
    model.fit(train_df)
    
    # Optimize model size before saving
    if hasattr(model, 'stan_backend'):
        model.stan_backend = None
    
    # Save model to local folder first
    os.makedirs('models', exist_ok=True)  # Create models directory if it doesn't exist
    local_model_path = f'models/prophet_model_{name}.joblib'
    joblib.dump(model, local_model_path, compress=3)
    print(f"Model for {name} saved locally to: {local_model_path}")
    
    # Save model to Firebase
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        temp_path = tmp.name
        joblib.dump(model, temp_path, compress=3)
        
        firebase_model_path = f'models/prophet_model_{name}.joblib'
        blob = bucket.blob(firebase_model_path)
        blob.upload_from_filename(temp_path)
        
    os.unlink(temp_path)
    print(f"Model for {name} saved to Firebase: {firebase_model_path}")
    
    # Generate and save forecast
    future = model.make_future_dataframe(periods=6, freq='ME')
    forecast = model.predict(future)
    
    # Format forecast data for API consumption
    last_6_forecast = forecast.tail(6)
    forecast_data = []
    for _, row in last_6_forecast.iterrows():
        forecast_data.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "forecast": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper'])
        })
    
    # Create the forecast JSON object
    forecast_json = {
        "data_type": name,
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "forecast_period": "6 months",
        "data": forecast_data
    }
    
    # Save forecast locally
    os.makedirs('forecasts', exist_ok=True)  # Create forecasts directory if it doesn't exist
    local_forecast_path = f'forecasts/{name}_forecast.json'
    with open(local_forecast_path, 'w') as f:
        json.dump(forecast_json, f, indent=4)
    print(f"Forecast for {name} saved locally to: {local_forecast_path}")
    
    # Save forecast to Firebase
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        temp_path = tmp.name
        with open(temp_path, 'w') as f:
            json.dump(forecast_json, f)
        
        firebase_forecast_path = f'forecasts/{name}_forecast.json'
        blob = bucket.blob(firebase_forecast_path)
        blob.upload_from_filename(temp_path)
    
    os.unlink(temp_path)
    print(f"Forecast for {name} saved to Firebase: {firebase_forecast_path}")
    
    # Calculate metrics
    test_preds = forecast.iloc[-6:][['yhat']].reset_index(drop=True)
    test_actuals = test_df[['y']].reset_index(drop=True)
    
    mae = mean_absolute_error(test_actuals, test_preds)
    rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    print(f"Model for {name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Generate visualization
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
    
    # Save visualization only locally
    viz_filename = f'{name}_model_forecast.png'
    plt.savefig(viz_filename)
    print(f"Visualization saved locally: {viz_filename}")
    
    return model, forecast

# Main execution
if __name__ == "__main__":
    # Get current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Generate market data from 2020 to current month
    print("Generating market data...")
    market_data = generate_market_data(start_date='2020-01-01', end_date=current_date)
    
    # Train models for each dataset
    print("\nTraining specialized models...")
    
    model_types = ['supply', 'demand', 'price', 'bulk_price']
    models = {}
    forecasts = {}
    
    for data_type in model_types:
        print(f"\nTraining {data_type} model...")
        models[data_type], forecasts[data_type] = train_save_model(market_data[data_type], data_type)
    
    print("\nAll models trained and saved successfully!")
    
    # Display summary of all models
    print("\n======= Model Evaluation Summary =======")
    for data_type in model_types:
        df = market_data[data_type]
        test_df = df.iloc[-6:]
        forecast = forecasts[data_type]
        
        test_preds = forecast.iloc[-6:][['yhat']].reset_index(drop=True)
        test_actuals = test_df[['y']].reset_index(drop=True)
        
        mae = mean_absolute_error(test_actuals, test_preds)
        rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
        
        print(f"{data_type.capitalize()} Model:")
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")
        print(f"  - Model saved to Firebase: models/prophet_model_{data_type}.joblib")
