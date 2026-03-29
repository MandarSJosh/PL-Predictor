# 🎯 Complete Guide: Predict 2025/26 Premier League Table

This guide will walk you through:
1. ✅ Verifying the model is trained with class weights (for draw prediction)
2. ✅ Testing the model to confirm draws are predicted
3. ✅ Collecting 2024/25 season data (up to Jan 11, 2026)
4. ✅ Predicting the full 2025/26 Premier League season table

---

## 📋 Prerequisites

- Python 3.12+ installed
- All dependencies installed (run `python3 -m pip install -r requirements.txt` if needed)
- Historical match data (already collected: `data/raw_matches.csv`)

---

## Step 1: Verify Training Status

**Check if training is complete:**

```bash
# Check if the model was recently updated
ls -lh models/best_model.pkl
```

If the timestamp is from before you started training, the training may still be running or needs to be completed.

**If training is NOT complete yet:**
- Wait for it to finish (it shows progress in the terminal)
- OR run it in the background:
  ```bash
  nohup python3 -m src.pipeline --features --train > training.log 2>&1 &
  ```

**If you want to skip waiting and use the current model:**
- The current model exists but doesn't predict draws yet
- You can still proceed, but draws won't be predicted

---

## Step 2: Test the Model (Verify Draw Prediction)

**Run the test script:**

```bash
python3 test_predictions.py
```

**What to look for:**
- ✅ **Good sign**: Classification report shows Draw predictions (precision/recall > 0)
- ✅ **Good sign**: Confusion matrix shows some draws in the "Draw" row
- ❌ **Bad sign**: Draw precision/recall = 0.00 (model still doesn't predict draws)

**Example of GOOD output:**
```
              precision    recall  f1-score   support

    Away Win       0.XX      0.XX      0.XX      181
        Draw       0.XX      0.XX      0.XX      127    <-- Should NOT be all 0.00!
    Home Win       0.XX      0.XX      0.XX      224
```

**If draws are NOT predicted yet:**
- You can still proceed, but the model will only predict wins
- The table will still be generated, but may have fewer draws

---

## Step 3: Collect 2024/25 Season Data (Up to Jan 11, 2026)

**Run the data collection script:**

```bash
python3 collect_current_season.py
```

**What this does:**
- Scrapes 2024/25 season matches from FBref
- Engineers features for all matches
- Saves to `data/features_2024_25.csv`

**Expected output:**
```
============================================================
COLLECTING 2024/25 SEASON DATA
============================================================
Scraping 2024/25 season matches...
✅ Saved XXX matches to data/raw_matches_2024_25.csv
✅ Saved features for XXX matches to data/features_2024_25.csv
```

**Note:** This requires network access and may take 10-20 minutes depending on how many matches need to be scraped.

**If scraping fails:**
- Check your internet connection
- FBref may have rate limiting - wait a few minutes and try again
- The script uses a 2-second delay between requests

---

## Step 4: Predict 2025/26 Season Table

**Run the prediction script:**

```bash
python3 predict_2025_26_table.py
```

**What this does:**
1. Loads the trained model
2. Loads historical data (2018-2025)
3. Generates fixture list for 2025/26 season (380 matches total)
4. For each fixture:
   - Calculates features based on current team form
   - Makes prediction (Home Win / Draw / Away Win)
5. Simulates the full season
6. Calculates final league table
7. Displays the table and saves to `data/table_2025_26.csv`

**Expected output:**
```
============================================================
PREDICTING 2025/26 PREMIER LEAGUE SEASON
============================================================
✅ Model loaded
✅ Loaded 2660 historical matches
✅ Added 2024/25 season data
Generating 2025/26 fixture list...
Generated 380 fixtures for 2025/26 season
Predicting matches (this may take a while)...
  Progress: 50/380 matches...
  Progress: 100/380 matches...
  ...
✅ Saved 380 predictions to data/predictions_2025_26.csv

Calculating final table...

============================================================
PREDICTED 2025/26 PREMIER LEAGUE TABLE
============================================================

Pos  Team                  P    W    D    L    GF   GA   GD    Pts
----------------------------------------------------------------------
1    Manchester City       38   25   8    5    78   32   46    83
2    Arsenal               38   24   9    5    75   35   40    81
...
20   Wolverhampton         38   8    7    23   32   58   -26   31

Champions League (Top 4):
  1. Manchester City
  2. Arsenal
  3. Liverpool
  4. Chelsea

Relegation (Bottom 3):
  18. Southampton
  19. Nott'ham Forest
  20. Wolverhampton

✅ Saved table to data/table_2025_26.csv
```

**This may take 30-60 minutes** because:
- Each match requires feature engineering
- 380 matches need to be processed
- Feature calculation involves historical lookups

---

## Step 5: View Results

**View the predicted table:**

```bash
# View as CSV
cat data/table_2025_26.csv

# Or open in Excel/Numbers
open data/table_2025_26.csv

# View individual predictions
head -20 data/predictions_2025_26.csv
```

**Files created:**
- `data/table_2025_26.csv` - Final league table with positions, points, goals
- `data/predictions_2025_26.csv` - All 380 match predictions with probabilities

---

## 🔄 Quick Start (All Steps at Once)

If you want to run everything in sequence:

```bash
# 1. Test model (optional - to verify draws)
python3 test_predictions.py

# 2. Collect current season data
python3 collect_current_season.py

# 3. Predict 2025/26 table
python3 predict_2025_26_table.py
```

---

## ⚠️ Troubleshooting

### Problem: Training not complete
**Solution:**
```bash
# Check training log
tail -f training.log

# Or restart training
python3 -m src.pipeline --features --train
```

### Problem: Model doesn't predict draws
**Solution:**
- This is OK for now - the table will still be generated
- Draws will just be predicted as wins based on probabilities
- To fix: ensure training completed with class weights (check logs)

### Problem: Data collection fails
**Solution:**
- Check internet connection
- FBref may be blocking - wait 10 minutes and retry
- Try running with a longer delay (edit `collect_current_season.py` and increase delay)

### Problem: Predictions script is slow
**Solution:**
- This is normal - 380 matches takes time
- Each match needs feature engineering
- Estimated time: 30-60 minutes
- Progress is shown every 50 matches

### Problem: "Model not found" error
**Solution:**
```bash
# Train the model first
python3 -m src.pipeline --features --train
```

### Problem: "Features not found" error
**Solution:**
```bash
# Generate features first
python3 -m src.pipeline --features
```

---

## 📊 Understanding the Results

### Table Columns:
- **Pos**: Final league position (1-20)
- **Team**: Team name
- **P**: Matches played (should be 38 for all teams)
- **W**: Wins
- **D**: Draws (may be 0 if model doesn't predict draws)
- **L**: Losses
- **GF**: Goals For (simulated)
- **GA**: Goals Against (simulated)
- **GD**: Goal Difference
- **Pts**: Points (W×3 + D×1)

### Predictions File:
- Contains all 380 match predictions
- Columns: `home_team`, `away_team`, `predicted_outcome`, `home_win_prob`, `draw_prob`, `away_win_prob`, `confidence`, `matchday`, `date`

---

## 🎯 Expected Timeline

- **Step 1 (Verify)**: 1 minute
- **Step 2 (Test)**: 2-3 minutes
- **Step 3 (Collect Data)**: 10-20 minutes (requires network)
- **Step 4 (Predict Table)**: 30-60 minutes
- **Total**: ~45-90 minutes

---

## ✅ Success Criteria

You'll know everything worked if:

1. ✅ `test_predictions.py` shows draws being predicted (precision > 0 for Draws)
2. ✅ `data/features_2024_25.csv` exists with 2024/25 season data
3. ✅ `data/predictions_2025_26.csv` exists with 380 match predictions
4. ✅ `data/table_2025_26.csv` exists with final league table
5. ✅ Table shows all 20 teams with 38 matches each
6. ✅ Points add up correctly (W×3 + D×1)
7. ✅ Table is sorted by points, then goal difference

---

## 🚀 Next Steps (After Getting Results)

Once you have the predicted table:

1. **Compare with actual 2024/25 table** (to validate model)
2. **Analyze predictions** - Which teams are over/under-rated?
3. **Check fixture difficulty** - Look at predictions CSV to see tough/easy fixtures
4. **Update with new data** - As 2024/25 season continues, re-run predictions
5. **Improve model** - Collect more features, retrain with more data

---

## 📝 Notes

- **Current date assumption**: Predictions are based on data up to Jan 11, 2026
- **Fixtures**: Generated automatically (not the actual PL schedule)
- **Goals**: Simulated based on win/draw/loss (simplified)
- **Future matches**: Features are estimated from current form
- **Model accuracy**: ~55-60% (better than random 33%)

---

**Ready to start? Run Step 1!** 🎉

