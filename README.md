# PL-Predictor — Premier League 2025/26 Match Outcome & Table Predictor

**GitHub:** [github.com/MandarSJosh/PL-Predictor](https://github.com/MandarSJosh/PL-Predictor)

XGBoost ensemble (calibrated, **~51.9%** accuracy, macro F1 **~0.49**) trained on H2H history, rolling form (5 & 10 game windows), squad value, manager PPG & honours, and related signals. Predicts remaining fixtures and projects the final table.

**Interactive Streamlit dashboard**

- Predicted final table with UCL / UEL / UECL / relegation styling  
- Points trajectory: actual results → linear projection to model final points  
- Head-to-head home-win probability heatmap (20×20)  
- Feature importance (XGBoost gain)  
- Remaining fixtures with win / draw / loss probabilities  

**Stack:** Python · XGBoost · LightGBM · scikit-learn · Optuna · MLflow · Streamlit · Plotly · Altair  

---

## Repository layout

```
PL-Predictor/
├── app.py                      # Streamlit entry → src/dashboard/app.py
├── requirements.txt
├── build_predicted_table.py    # Rebuild predicted table from current_table + model
├── config.yaml
├── src/
│   ├── dashboard/app.py        # Main Streamlit UI
│   ├── data_collection/
│   ├── feature_engineering/
│   ├── models/
│   └── utils/
├── data/                       # CSVs used by the dashboard (tables, features, metrics)
├── models/
│   └── best_model.pkl          # Git LFS (~130+ MB) — add per README below
├── docs/                       # Install guides, architecture, weather API, etc.
└── README.md
```

---

## Git LFS (required for `models/best_model.pkl`)

The trained artifact is **~130+ MB**, above GitHub’s **100 MB** per-file cap. **Install Git LFS before adding the model to any commit.**

**Recommended two-step first push**

1. Commit the codebase **without** the model (or unstage the file before commit), push, and confirm the repo is on GitHub.  
2. Then:

```bash
brew install git-lfs          # or your OS package manager
git lfs install
git lfs track "models/best_model.pkl"
git add .gitattributes models/best_model.pkl
git commit -m "chore: add trained model via Git LFS"
git push
```

`.gitattributes` in this repo already declares LFS for `models/best_model.pkl`. If you accidentally committed the raw binary, migrate with:

`git lfs migrate import --include="models/best_model.pkl" --everything`

---

## Clone this repo

```bash
git clone https://github.com/MandarSJosh/PL-Predictor.git
cd PL-Predictor
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Ensure `data/` contains at least `predicted_table.csv`, `current_table.csv`, `raw_matches.csv`, `remaining_predictions.csv`, `features.csv`, and `performance_metrics.txt` as expected by the app.

---

## Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (with LFS for the model).  
2. Sign in at [share.streamlit.io](https://share.streamlit.io).  
3. **New app** → pick the repo → **Main file path:** `app.py` (repo root).  
4. Python version: **3.11** (or 3.10) is a safe default.  
5. Deploy; cold start may take a minute while dependencies install.

---

## LinkedIn-style blurb

Built a full Premier League predictor from scratch — XGBoost ensemble trained on form, squad value, manager signals, and H2H-style features — deployed as an interactive Streamlit dashboard with a live predicted table, points trajectory, and H2H probability heatmap. Model accuracy ~**52%** vs ~**33%** random baseline for 3-class outcomes; much of the work was feature pipelines, calibration, and a Bloomberg-style UI in Python.
