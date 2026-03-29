# 🚀 Run the Model NOW - Step by Step

## Step 1: Install Dependencies

Run this command in your terminal:

```bash
./step_by_step.sh
```

Or manually install:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install pandas numpy scikit-learn xgboost lightgbm requests beautifulsoup4 lxml cloudscraper mlflow optuna fastapi uvicorn pydantic streamlit plotly python-dotenv joblib tqdm openpyxl pyyaml
```

## Step 2: Collect Data

This will scrape match data from FBref (takes 10-20 minutes):

```bash
python3 -m src.pipeline --collect --start-season 2018-2019 --end-season 2024-2025
```

## Step 3: Create Features

```bash
python3 -m src.pipeline --features
```

## Step 4: Train Model

**With optimization (slower, better accuracy):**
```bash
python3 -m src.pipeline --train
```

**Without optimization (faster):**
```bash
python3 -m src.pipeline --train --no-optimize
```

## Step 5: Make Predictions

### Option A: Use Dashboard (Easiest)

Terminal 1:
```bash
python3 -m src.api.main
```

Terminal 2:
```bash
streamlit run src/dashboard/app.py
```

Then open http://localhost:8501

### Option B: Use Python

```python
from src.models.predictor import MatchPredictor

predictor = MatchPredictor("models/best_model.pkl")
result = predictor.predict_match("Arsenal", "Chelsea", {})
print(result)
```

## OR: Run Everything at Once

```bash
python3 -m src.pipeline --all
```

Then start the API and dashboard as shown above.

