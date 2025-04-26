import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import joblib
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Paths to the specialized models
MODEL_PATHS = {
    'supply': 'API/prophet_model_supply.joblib',
    'demand': 'API/prophet_model_demand.joblib',
    'price': 'API/prophet_model_price.joblib'
}

def load_models():
    """Load all the trained Prophet models"""
    models = {}
    
    for data_type, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[data_type] = joblib.load(path)
            print(f"Loaded {data_type} model from {path}")
        else:
            print(f"Warning: Model not found at {path}")
            
    return models

def generate_test_data(data_type, start_date='2023-01-01', periods=6):
    """
    Generate test data for a specific data type that follows expected patterns
    """
    dates = pd.date_range(start=start_date, periods=periods, freq='ME')
    
    # Generate appropriate test data based on data type
    if data_type == 'supply':
        # Supply has a base around 1300-1500 with summer seasonality
        base = 1400
        values = base + 100 * np.sin(np.linspace(0, np.pi, periods)) + np.random.normal(0, 30, periods)
    
    elif data_type == 'demand':
        # Demand has a base around 1400-1800 with winter seasonality
        base = 1650
        values = base + 150 * np.sin(np.linspace(np.pi, 2*np.pi, periods)) + np.random.normal(0, 50, periods)
        
    elif data_type == 'price':
        # Price is in IDR, around 750k with fluctuations
        base = 750000
        values = base + 50000 * np.sin(np.linspace(0, 2*np.pi, periods)) + np.random.normal(0, 20000, periods)
    
    else:
        # Generic data
        base = 150
        values = base + 20 * np.sin(np.linspace(0, 2*np.pi, periods)) + np.random.normal(0, 5, periods)
    
    # Create DataFrame in Prophet format
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    return df

def test_market_data():
    """Load the real market data files"""
    # Load existing market data files
    try:
        with open('supply.json', 'r') as f:
            supply_data = json.load(f)
        with open('demand.json', 'r') as f:
            demand_data = json.load(f)
        with open('average-price.json', 'r') as f:
            price_data = json.load(f)
            
        print("Successfully loaded market data files")
        
        # Extract the data from JSON
        supply_df = pd.DataFrame(supply_data['data'])
        demand_df = pd.DataFrame(demand_data['data'])
        price_df = pd.DataFrame(price_data['data'])
        
        # Convert to proper format for Prophet
        supply_df['ds'] = pd.to_datetime(supply_df['date'])
        supply_df['y'] = supply_df['value']
        
        demand_df['ds'] = pd.to_datetime(demand_df['date'])
        demand_df['y'] = demand_df['value']
        
        price_df['ds'] = pd.to_datetime(price_df['date'])
        price_df['y'] = price_df['value']
        
        # Return all three datasets
        return {
            'supply': supply_df[['ds', 'y']],
            'demand': demand_df[['ds', 'y']],
            'price': price_df[['ds', 'y']]
        }
    except Exception as e:
        print(f"Error loading market data: {str(e)}")
        return None

def main():
    # Load all specialized models
    models = load_models()
    
    if not models:
        print("No models could be loaded. Please run main.py first to train the models.")
        return
        
    print("\n====== TESTING SPECIALIZED MODELS WITH NEW TEST DATA ======")
    
    # Test each model with appropriate test data
    for data_type, model in models.items():
        print(f"\nTesting {data_type} model with new test data:")
        
        # Generate appropriate test data for this model
        test_data = generate_test_data(data_type)
        
        # Make prediction using the specialized model
        future = model.make_future_dataframe(periods=6, freq='ME')
        forecast = model.predict(future)
        
        # Extract just the forecast period (last 6 months)
        forecast_result = forecast.tail(6)
        
        # Calculate MAE and MSE between forecast and test data
        # Ensure the dates match up
        merged_data = pd.merge(
            forecast_result[['ds', 'yhat']], 
            test_data,
            on='ds'
        )
        
        if not merged_data.empty:
            mae = mean_absolute_error(merged_data['y'], merged_data['yhat'])
            mse = mean_squared_error(merged_data['y'], merged_data['yhat'])
            rmse = np.sqrt(mse)
            
            print(f"\nError Metrics for {data_type}:")
            print(f"  MAE: {mae:.2f}")
            print(f"  MSE: {mse:.2f}")
            print(f"  RMSE: {rmse:.2f}")
        
        # Display forecast metrics
        for i, (_, row) in enumerate(forecast_result.iterrows()):
            print(f"  Month {i+1}: {row['ds'].strftime('%Y-%m-%d')} - " 
                  f"Forecast: {row['yhat']:.2f} [{row['yhat_lower']:.2f}, {row['yhat_upper']:.2f}]")
                  
        # Compare with our generated test data
        print(f"\nComparison with generated test data:")
        for i, (_, row) in enumerate(test_data.iterrows()):
            print(f"  Month {i+1}: {row['ds'].strftime('%Y-%m-%d')} - Actual: {row['y']:.2f}")
        
        # Plot the results
        plt.figure(figsize=(14, 8))
        
        # For more complete context, load the original data if available
        market_data = test_market_data()
        if market_data and data_type in market_data:
            # Plot historical data
            historical_data = market_data[data_type]
            plt.plot(historical_data['ds'], historical_data['y'], 'b-', 
                     label=f'Historical {data_type} data')
            
            # Mark the end of historical data
            last_date = historical_data['ds'].iloc[-1]
            plt.axvline(x=last_date, color='red', linestyle='--', 
                        label='End of Historical Data')
        
        # Plot the forecast
        plt.plot(forecast_result['ds'], forecast_result['yhat'], 'g-', 
                 marker='o', linewidth=2, label='Model Forecast')
        
        # Add confidence interval
        plt.fill_between(
            forecast_result['ds'],
            forecast_result['yhat_lower'],
            forecast_result['yhat_upper'],
            color='green', alpha=0.2,
            label='95% Confidence Interval'
        )
        
        # Plot the test data points
        plt.plot(test_data['ds'], test_data['y'], 'ro--', 
                 linewidth=2, markersize=8, label='Generated Test Data')
        
        # Add metrics annotation to the plot
        if 'mae' in locals() and 'mse' in locals() and 'rmse' in locals():
            plt.annotate(
                f'Error Metrics:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}',
                xy=(0.02, 0.96),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                fontsize=12,
                verticalalignment='top'
            )
        
        # Add labels and title
        plt.title(f'{data_type.capitalize()} Forecast Test', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel(f'{data_type.capitalize()} Value', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'{data_type}_model_test.png')
        plt.show()
    
    print("\nAll model tests completed!")

if __name__ == "__main__":
    main()
