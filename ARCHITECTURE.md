# System Architecture

## Overview

This is an end-to-end MLOps application for predicting Premier League match outcomes. The system follows best practices for machine learning production systems.

## Architecture Components

### 1. Data Collection Layer

**Location**: `src/data_collection/fbref_scraper.py`

- **Purpose**: Scrapes match data and team statistics from FBref
- **Features**:
  - Rate limiting to respect website policies
  - Error handling and retry logic
  - Support for multiple seasons
  - Advanced statistics collection

**Data Collected**:
- Match results (scores, dates, teams)
- Team statistics
- Advanced metrics (when available)

### 2. Feature Engineering Layer

**Location**: `src/feature_engineering/features.py`

**Feature Categories**:

1. **Rolling Statistics** (Last N matches)
   - Average goals scored/conceded
   - Goal difference
   - Window size: 10 matches

2. **Form Features** (Recent performance)
   - Points in last 5 matches
   - Win rate in last 5 matches
   - Window size: 5 matches

3. **Head-to-Head** (Historical matchups)
   - Win/draw/loss ratios
   - Average goals scored/conceded

4. **Team Strength** (Overall performance)
   - Cumulative points
   - Overall win rate
   - Expanding window (all-time)

5. **Momentum** (Recent trend)
   - Points in last 3 vs last 5 matches
   - Trend indicator

6. **Position Features** (League standing)
   - Current league position
   - Position difference between teams

7. **Time Features** (Temporal)
   - Month, day of week
   - Days since season start
   - Weekend indicator

### 3. Model Training Layer

**Location**: `src/models/trainer.py`

**Models Implemented**:

1. **XGBoost**
   - Gradient boosting with tree-based learning
   - Handles non-linear relationships
   - Robust to outliers

2. **LightGBM**
   - Fast gradient boosting
   - Efficient with large datasets
   - Good default parameters

3. **CatBoost**
   - Handles categorical features well
   - Built-in regularization
   - Less prone to overfitting

4. **Ensemble**
   - Voting classifier combining best models
   - Soft voting (probability-based)
   - Often improves accuracy

**Hyperparameter Optimization**:
- Uses Optuna for automated tuning
- 50 trials per model (configurable)
- Maximizes accuracy
- Tracks all experiments in MLflow

### 4. MLOps Infrastructure

**MLflow Integration**:
- Experiment tracking
- Model versioning
- Parameter and metric logging
- Model artifact storage

**Benefits**:
- Reproducibility
- Model comparison
- Easy rollback to previous models
- Production deployment tracking

### 5. Prediction Service

**Location**: `src/api/main.py`

**FastAPI REST API**:
- `/predict`: Single match prediction
- `/predict/batch`: Multiple match predictions
- `/health`: Health check
- `/teams`: List of teams

**Features**:
- Type validation with Pydantic
- Error handling
- Async support
- Auto-generated documentation

### 6. Dashboard

**Location**: `src/dashboard/app.py`

**Streamlit Interface**:
- Match prediction interface
- Season overview
- Team analysis
- Interactive visualizations

**Features**:
- Real-time predictions
- Probability visualizations
- Team selection
- Responsive design

## Data Flow

```
FBref Website
    ↓
Data Collection (Scraper)
    ↓
Raw Match Data (CSV)
    ↓
Feature Engineering Pipeline
    ↓
Engineered Features (CSV)
    ↓
Model Training (XGBoost, LightGBM, CatBoost)
    ↓
Model Selection & Ensemble
    ↓
Trained Model (Pickle)
    ↓
Prediction Service (FastAPI)
    ↓
Dashboard (Streamlit)
```

## Model Training Pipeline

1. **Data Preparation**
   - Load features
   - Split train/test (80/20, time-based)
   - Scale features (StandardScaler)
   - Handle missing values

2. **Model Training**
   - Train each model individually
   - Optimize hyperparameters (Optuna)
   - Evaluate on test set
   - Log to MLflow

3. **Ensemble Creation**
   - Select best models (accuracy > 0.5)
   - Create voting classifier
   - Evaluate ensemble

4. **Model Selection**
   - Compare all models
   - Select best (highest accuracy)
   - Save model artifacts

## Improving Accuracy

### 1. More Data
- Collect more historical seasons (2010+)
- Include cup competitions
- Add international matches for teams

### 2. Better Features

**Advanced Statistics**:
- Expected Goals (xG)
- Expected Assists (xA)
- Shot quality metrics
- Possession statistics

**Contextual Features**:
- Player injuries/suspensions
- Weather conditions
- Referee statistics
- Home/away form separately

**Market Data**:
- Betting odds (as features)
- Transfer market activity
- Manager changes

### 3. Model Improvements

**Hyperparameter Tuning**:
- Increase Optuna trials (100+)
- Use more sophisticated search spaces
- Multi-objective optimization (accuracy + log loss)

**Advanced Models**:
- Neural networks (LSTM for sequences)
- Transformer models
- Deep learning ensembles

**Feature Selection**:
- Remove redundant features
- Use feature importance
- Try different feature combinations

### 4. Data Quality

**Validation**:
- Check for data inconsistencies
- Validate team names
- Handle missing values better
- Remove outliers

**Enrichment**:
- Cross-reference multiple sources
- Validate against official records
- Handle data updates

## Deployment Considerations

### Production Deployment

1. **Model Serving**:
   - Use MLflow model serving
   - Or containerize with Docker
   - Deploy to cloud (AWS, GCP, Azure)

2. **API Deployment**:
   - Use production ASGI server (Gunicorn + Uvicorn)
   - Add authentication
   - Rate limiting
   - Monitoring and logging

3. **Dashboard**:
   - Deploy to Streamlit Cloud
   - Or containerize
   - Add caching for performance

4. **Data Pipeline**:
   - Schedule data collection (cron, Airflow)
   - Automated retraining pipeline
   - Model monitoring and drift detection

### Monitoring

- Model performance metrics
- Prediction accuracy over time
- Feature drift detection
- API performance metrics
- Error tracking

## Best Practices Implemented

1. ✅ **Modular Design**: Separate concerns (data, features, models, API)
2. ✅ **Reproducibility**: MLflow tracking, version control
3. ✅ **Error Handling**: Comprehensive try/except blocks
4. ✅ **Logging**: Structured logging throughout
5. ✅ **Type Hints**: Python type annotations
6. ✅ **Documentation**: Docstrings and README
7. ✅ **Configuration**: YAML config file
8. ✅ **Testing Ready**: Structure supports unit tests

## Future Enhancements

1. **Real-time Updates**: Live data feeds
2. **Player-level Models**: Individual player impact
3. **Injury Prediction**: Predict player availability
4. **Betting Integration**: Odds comparison
5. **Mobile App**: Native mobile interface
6. **Advanced Analytics**: Expected points, relegation probabilities
7. **Multi-league Support**: Expand to other leagues

## Performance Expectations

**Baseline Accuracy**: ~50-55% (random would be ~33%)

**With Good Features**: ~55-60%

**With Advanced Features**: ~60-65%

**State-of-the-art**: ~65-70% (with betting odds, xG, etc.)

**Note**: Football is inherently unpredictable. Even the best models struggle to exceed 70% accuracy due to the high variance in match outcomes.

