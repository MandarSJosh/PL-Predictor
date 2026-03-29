# Quick Start Guide

Get your Premier League prediction system up and running in minutes!

## Step 1: Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Collect Data

Collect historical Premier League data from FBref:

```bash
python -m src.pipeline --collect --start-season 2018-2019 --end-season 2024-2025
```

**Note**: This may take 10-20 minutes depending on your internet connection. The scraper includes rate limiting to be respectful to FBref.

## Step 3: Engineer Features

Create features from the collected match data:

```bash
python -m src.pipeline --features
```

## Step 4: Train Models

Train models with hyperparameter optimization (this may take 30-60 minutes):

```bash
python -m src.pipeline --train
```

Or skip optimization for faster training:

```bash
python -m src.pipeline --train --no-optimize
```

## Step 5: Start Services

### Start API (Terminal 1)

```bash
python -m src.api.main
```

Or use the helper script:

```bash
./run_api.sh
```

API will be available at:
- Main: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Start Dashboard (Terminal 2)

```bash
streamlit run src/dashboard/app.py
```

Or use the helper script:

```bash
./run_dashboard.sh
```

Dashboard will be available at: http://localhost:8501

## Step 6: Make Predictions

### Via Dashboard

1. Open http://localhost:8501
2. Select "Match Prediction"
3. Choose home and away teams
4. Click "Predict Match"

### Via API

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "home_team": "Arsenal",
        "away_team": "Chelsea"
    }
)

print(response.json())
```

### Via Python

```python
from src.models.predictor import MatchPredictor

predictor = MatchPredictor("models/best_model.pkl")
result = predictor.predict_match("Arsenal", "Chelsea", {})
print(result)
```

## Running Everything at Once

You can run the full pipeline in one command:

```bash
python -m src.pipeline --all
```

Or use the helper script:

```bash
./run_pipeline.sh
```

## Troubleshooting

### "Model not loaded" error

Make sure you've trained a model first:
```bash
python -m src.pipeline --train
```

### API connection errors

Ensure the API is running before starting the dashboard.

### Data collection fails

- Check your internet connection
- FBref may have rate limiting - wait a few minutes and try again
- Verify the season format is correct (e.g., "2024-2025")

### Import errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Improve Accuracy**: 
   - Collect more historical data
   - Add more features (xG, player stats, etc.)
   - Experiment with different models

2. **View MLflow Experiments**:
   ```bash
   mlflow ui
   ```
   Then open http://localhost:5000

3. **Customize Configuration**:
   Edit `config.yaml` to adjust parameters

## Tips for Best Accuracy

1. **More Data**: Collect data from 2010-2011 onwards for more training examples
2. **Feature Engineering**: Add xG data, player injuries, weather conditions
3. **Model Tuning**: Increase Optuna trials in config.yaml
4. **Ensemble**: The system automatically creates an ensemble of best models

## Support

For issues or questions, check the main README.md file.

