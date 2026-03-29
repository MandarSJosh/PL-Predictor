Models folder

This folder stores trained model artifacts.

Typical files:
- `best_model.pkl`: Serialized model, scaler, and feature metadata

Notes:
- You can retrain and regenerate this file with:
  - `python -m src.pipeline --train`
- For sharing with interviewers, you can include `best_model.pkl` for a
  quick demo, or exclude it and point to the training command above.
