import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import firebase_admin
from firebase_admin import credentials, storage
import tempfile

# Initialize Firebase with credentials
try:
    cred = credentials.Certificate('hackfest-2025-7f1a4-firebase-adminsdk-fbsvc-a86f273b3e.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'hackfest-2025-7f1a4.firebasestorage.app'
    })
    bucket = storage.bucket()
    print("Firebase storage initialized successfully")
    firebase_available = True
except Exception as e:
    print(f"Firebase initialization failed: {str(e)}")
    print("Will save JSON files locally only.")
    firebase_available = False

def generate_commodity_data(commodity, start_date='2020-01-01', end_date=None):
    """
    Generate market data (supply, demand, price, bulk_price) for a specific commodity
    
    Parameters:
    - commodity: String, one of 'ikan' (fish), 'apel' (apple), or 'beras' (rice)
    - start_date: String, start date in YYYY-MM-DD format
    - end_date: String, end date in YYYY-MM-DD format (defaults to current date)
    
    Returns:
    - Dictionary with dataframes for supply, demand, price, and bulk_price
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
    
    # Configure commodity-specific parameters
    if commodity == 'ikan':  # fish
        # Fish has more seasonal variations due to weather and fishing seasons
        supply_base_range = (800, 1200)  # in tons
        demand_base_range = (750, 1100)  # in tons
        price_base = 45000  # IDR per kg
        supply_seasonality_amplitude = 300  # Higher seasonal variation
        supply_noise_level = 70          # More unpredictable
        phase_shift = 0                  # Peak in January
        
    elif commodity == 'apel':  # apple
        # Apples are highly seasonal with harvest periods
        supply_base_range = (1200, 1600)  # in tons
        demand_base_range = (1000, 1300)  # in tons
        price_base = 25000  # IDR per kg
        supply_seasonality_amplitude = 400  # Very seasonal
        supply_noise_level = 50           # Medium unpredictability
        phase_shift = np.pi/2            # Peak shifted by 3 months (April)
        
    elif commodity == 'beras':  # rice
        # Rice is more stable as it's a staple food with controlled prices
        supply_base_range = (5000, 6000)  # in tons (higher volume)
        demand_base_range = (4800, 5800)  # in tons
        price_base = 12000  # IDR per kg (cheaper)
        supply_seasonality_amplitude = 200  # Less seasonal variation
        supply_noise_level = 30           # More predictable
        phase_shift = np.pi              # Peak in July (opposite)
        
    else:
        raise ValueError(f"Unsupported commodity: {commodity}")
    
    # Generate base trends with commodity-specific ranges
    supply_base = np.linspace(supply_base_range[0], supply_base_range[1], num_months)
    demand_base = np.linspace(demand_base_range[0], demand_base_range[1], num_months)
    
    # Generate supply with seasonality
    supply_seasonality = supply_seasonality_amplitude * np.sin(np.linspace(phase_shift, phase_shift + 2 * np.pi * (num_months/12), num_months))
    supply_noise = np.random.normal(0, supply_noise_level, num_months)
    supply_values = np.round(supply_base + supply_seasonality + supply_noise, 2)
    
    # Generate demand (less seasonal than supply but follows similar pattern with delay)
    demand_seasonality = (supply_seasonality_amplitude * 0.7) * np.sin(np.linspace(phase_shift + np.pi/6, phase_shift + np.pi/6 + 2 * np.pi * (num_months/12), num_months))
    demand_noise = np.random.normal(0, supply_noise_level * 0.6, num_months)
    demand_values = np.round(demand_base + demand_seasonality + demand_noise, 2)
    
    # Price calculations based on supply-demand dynamics
    price_sensitivity = price_base * 0.3  # 30% of base price can be affected by supply/demand
    price_seasonality = price_base * 0.1 * np.sin(np.linspace(phase_shift + np.pi, phase_shift + np.pi + 2 * np.pi * (num_months/12), num_months))
    price_noise = np.random.normal(0, price_base * 0.05, num_months)
    
    # Calculate supply-demand ratio effect on price
    supply_demand_ratio = demand_values / supply_values
    ratio_effect = price_sensitivity * (supply_demand_ratio - 1)
    
    # Generate final price
    price_values = np.round(price_base + price_seasonality + ratio_effect + price_noise, 2)
    
    # Ensure all prices are positive
    price_values = np.maximum(price_values, price_base * 0.5)
    
    # Generate bulk price (with discount compared to normal price)
    bulk_discount = np.random.uniform(0.8, 0.9, num_months)  # 10-20% discount
    bulk_price_values = np.round(price_values * bulk_discount, 2)
    
    # Create dataset identifier
    dataset_id = f"{commodity}"
    
    # Create and save JSON files
    for name, values in [
        ('supply', supply_values), 
        ('demand', demand_values), 
        ('price', price_values), 
        ('bulk_price', bulk_price_values)
    ]:
        # Create JSON structure
        unit = "IDR/kg" if "price" in name else "tons"
        metric = f"{name}"
        
        json_data = {
            "commodity": commodity,
            "metric": metric,
            "unit": unit,
            "data": [{"date": date, "value": float(value)} for date, value in zip(dates_str, values)]
        }
        
        # Create folder for the commodity if it doesn't exist
        os.makedirs(commodity, exist_ok=True)
        
        # Save locally
        filename = f"{commodity}/{name}.json"
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=4)
            
        # Save to Firebase if available
        if firebase_available:
            try:
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                    temp_path = tmp.name
                    with open(temp_path, 'w') as f:
                        json.dump(json_data, f)
                    
                    # Upload to Firebase Storage
                    firebase_path = f'data/{commodity}/{name}.json'
                    blob = bucket.blob(firebase_path)
                    blob.upload_from_filename(temp_path)
                
                # Remove temp file
                os.unlink(temp_path)
            except Exception as e:
                print(f"Firebase upload failed for {filename}: {str(e)}")
    
    print(f"Market data files for {commodity} created and saved to '{commodity}' folder")
    
    # Return the dataframes for model training if needed
    return {
        'supply': pd.DataFrame({'ds': dates, 'y': supply_values}),
        'demand': pd.DataFrame({'ds': dates, 'y': demand_values}),
        'price': pd.DataFrame({'ds': dates, 'y': price_values}),
        'bulk_price': pd.DataFrame({'ds': dates, 'y': bulk_price_values})
    }

if __name__ == "__main__":
    # Get current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Generate market data for each commodity
    commodities = ['ikan', 'apel', 'beras']
    
    for commodity in commodities:
        print(f"\nGenerating market data for {commodity}...")
        data = generate_commodity_data(commodity, start_date='2020-01-01', end_date=current_date)
    
    print("\nAll market data generation completed!")
    print(f"Successfully generated 12 JSON files (4 metrics for each of the 3 commodities)")
