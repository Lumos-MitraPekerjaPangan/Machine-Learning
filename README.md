# Hackathon Market Data & Forecasting

This project provides a pipeline for generating, training, and evaluating time series forecasting models for multiple commodities (ikan, apel, beras) using Prophet and Firebase for storage.

## Structure

- `generate_data_dummy.py`  
  Generate dummy market data (supply, demand, price, bulk_price) for each commodity. Saves JSON files locally and uploads to Firebase Storage.

- `generate_market_data.py`  
  Contains the `generate_commodity_data` function used for generating market data for a single commodity. Used by other scripts.

- `main.py`  
  Trains Prophet models for each metric (supply, demand, price, bulk_price) for a selected commodity using the generated data. Saves models and forecasts locally and to Firebase.

- `model_test.py`  
  Loads the generated JSON data and models, retrains and tests the models for each commodity and metric, and saves comparison plots to the `plots/` directory.

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dummy Data

This will create folders and JSON files for each commodity (ikan, apel, beras):

```bash
python generate_data_dummy.py
```

### 3. Train Models

By default, `main.py` will use the generated data for a selected commodity (default: ikan), train Prophet models for each metric, and save the models and forecasts both locally and to Firebase:

```bash
python main.py
```

To train for all commodities, modify the loop in `main.py` to iterate over all generated data.

### 4. Test Models

This will retrain and test the models for each commodity and metric, and save comparison plots to `plots/{commodity}/{metric}_prediction_vs_actual.png`:

```bash
python model_test.py
```

## Firebase

- Requires a Firebase project and a service account key JSON (`hackfest-2025-7f1a4-firebase-adminsdk-fbsvc-a86f273b3e.json`).
- Data and models are uploaded to the bucket: `hackfest-2025-7f1a4.firebasestorage.app`.

## Notes

- All generated data and models are saved locally for reference.
- Modify the scripts as needed to change commodities, date ranges, or model parameters.
- Ensure your Firebase credentials JSON is present in the project directory.

---
