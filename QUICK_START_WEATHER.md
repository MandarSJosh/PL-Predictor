# Quick Start: After Setting API Key

## Step 1: Verify Your API Key is Set

```bash
# Check if the environment variable is set:
echo $OPENWEATHER_API_KEY

# If it shows your key, you're good! If it's empty, you need to export it again.
```

## Step 2: Test the API Key

```bash
# Quick test to see if it works:
python3 -c "
from src.data_collection.weather_api import WeatherAPI
api = WeatherAPI()
if api.api_key:
    print('✅ API key loaded successfully!')
    print(f'Key starts with: {api.api_key[:8]}...')
    # Try to get weather for a test match
    weather = api.get_weather_for_match('Arsenal', '2024-01-15')
    if weather:
        print(f'✅ Weather API working! Temperature: {weather[\"temperature\"]}°C')
    else:
        print('⚠️  API key set but weather fetch failed (might need to wait 10 min for activation)')
else:
    print('❌ API key not found. Make sure you exported it:')
    print('   export OPENWEATHER_API_KEY=your_key_here')
"
```

## Step 3: Run the Pipeline with Weather Features

Once your API key is verified, run the full pipeline:

```bash
# This will automatically include weather features:
./run_pipeline_fixed.sh --features --train
```

The pipeline will:
1. Load existing match data
2. Add weather features (using your API key)
3. Add injury features (framework - uses defaults)
4. Engineer all features including xG, referee, weather, injuries
5. Train models with the enhanced features

## Step 4: Check the Results

Look for these log messages:

```
INFO: Adding weather features...
INFO: Processing weather for match 1/2660...
INFO: Weather features added!
INFO: Adding advanced features (xG, referee, weather, injuries)...
```

## What You'll See

- **Weather columns** in your features:
  - `weather_temperature`
  - `weather_humidity`
  - `weather_wind_speed`
  - `weather_is_rainy`
  - `weather_is_snowy`
  - `weather_is_clear`

- **Improved accuracy**: Weather features should help the model make better predictions!

## Troubleshooting

### "API key not found"
- Make sure you exported it in the same terminal session
- Or use `.env` file instead (see below)

### "401 Unauthorized"
- Wait 10 minutes after creating the key (activation delay)
- Double-check you copied the entire key correctly

### "Rate limit exceeded"
- Free tier: 60 calls/minute
- If processing many matches, it will automatically rate limit
- This is normal - just wait a bit

## Alternative: Use .env File (Permanent)

If you want the key to persist across terminal sessions:

```bash
# Create .env file:
echo "OPENWEATHER_API_KEY=your_actual_key_here" > .env

# Now it will work every time, no need to export!
```

The `.env` file is already in `.gitignore`, so it won't be committed to git.

## Next: Run Full Pipeline

```bash
./run_pipeline_fixed.sh --features --train
```

That's it! Weather features will be automatically included. 🎉

