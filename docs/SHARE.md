Sharing checklist (interview-ready)

Recommended contents:
- README.md
- requirements.txt (or requirements-minimal.txt)
- src/
- data/README.md (or small sample data if requested)
- models/README.md (and models/best_model.pkl if you want a demo-ready build)
- .env.example

Recommended exclusions:
- mlruns/
- mlflow.db
- __pycache__/
- .ipynb_checkpoints/
- venv/ or .venv/
- large raw datasets unless explicitly requested

Optional: quick zip command (from repo root)
zip -r premier-league-predictor.zip \
  README.md requirements.txt src data/README.md models/README.md \
  .env.example

If including a model:
zip -r premier-league-predictor.zip \
  README.md requirements.txt src data/README.md models/README.md \
  models/best_model.pkl .env.example
