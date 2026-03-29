# How to Run the Premier League Predictor

## Quick Start (Step-by-Step)

### Step 1: Install Dependencies

```bash
# Make sure you're in the project directory
cd "/Users/mandarjoshi/Desktop/Cool_Coding_stuff/Machine Learning Projects/PremierLeagueChampions:Games predictor2025:26"

# Run the installation script
./install.sh
```

Or manually:
```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

### Step 2: Collect Data from FBref

This will scrape historical match data:

```bash
python3 -m src.pipeline --collect --start-season 2018-2019 --end-season 2024-2025
```

**Note**: This may take 10-20 minutes. The scraper includes delays to be respectful to FBref.

### Step 3: Engineer Features

Create features from the collected data:

```bash
python3 -m src.pipeline --features
```

Optional (squad values):
```bash
python3 collect_team_values.py
```

### Step 4: Train the Model

Train models with hyperparameter optimization (takes 30-60 minutes):

```bash
python3 -m src.pipeline --train
```

Or skip optimization for faster training:

```bash
python3 -m src.pipeline --train --no-optimize
```

### Step 5: Make Predictions

Once the model is trained, you can make predictions in several ways:

#### Option A: Use the API

Start the API server:
```bash
python3 -m src.api.main
```

Then in another terminal, make a prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea"}'
```

Or use Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"home_team": "Arsenal", "away_team": "Chelsea"}
)
print(response.json())
```

#### Option B: Use the Dashboard

Start the dashboard:
```bash
streamlit run src/dashboard/app.py
```

Then open http://localhost:8501 in your browser and use the interactive interface.

#### Option C: Use Python Directly

```python
from src.models.predictor import MatchPredictor

# Load the trained model
predictor = MatchPredictor("models/best_model.pkl")

# Make a prediction
result = predictor.predict_match("Arsenal", "Chelsea", {})
print(result)
```

## Running Everything at Once

You can run the full pipeline (collect → features → train) in one command:

```bash
python3 -m src.pipeline --all
```

## Complete Example Workflow

```bash
# 1. Install (if not done)
./install.sh

# 2. Run full pipeline
python3 -m src.pipeline --all

# 3. Start API (Terminal 1)
python3 -m src.api.main

# 4. Start Dashboard (Terminal 2)
streamlit run src/dashboard/app.py
```

## Using Helper Scripts

```bash
# Run full pipeline
./run_pipeline.sh

# Fetch remaining fixtures and team list
python3 collect_pl_assets.py

# Fetch squad values
python3 collect_team_values.py

# Start API
./run_api.sh

# Start Dashboard  
./run_dashboard.sh
```

## What Each Step Does

1. **--collect**: Scrapes match data from FBref
2. **--features**: Creates features (form, stats, H2H, etc.)
3. **--train**: Trains XGBoost, LightGBM, CatBoost, and ensemble models
4. **API**: Serves predictions via REST API
5. **Dashboard**: Interactive web interface for predictions

## Expected Output

After training, you'll see:
- Model accuracy scores
- Log loss metrics
- Best model selected
- Model saved to `models/best_model.pkl`

## Troubleshooting

- **No data collected**: Check internet connection, FBref may have rate limiting
- **Model not found**: Make sure you've run `--train` first
- **Import errors**: Run `./install.sh` again
- **API not starting**: Check if port 8000 is available

## Next Steps

Once you have a trained model:
1. View MLflow experiments: `mlflow ui` then open http://localhost:5000
2. Make predictions for upcoming matches
3. Retrain with more data for better accuracy

