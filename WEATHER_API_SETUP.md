# OpenWeatherMap API Setup Guide

## Step-by-Step Instructions

### Step 1: Sign Up for Free Account

1. **Visit OpenWeatherMap**:
   - Go to: https://openweathermap.org/api
   - Or directly: https://home.openweathermap.org/users/sign_up

2. **Create Account**:
   - Click "Sign Up" or "Sign In" if you already have an account
   - Fill in:
     - Username
     - Email address
     - Password
   - Accept terms and conditions
   - Click "Create Account"

3. **Verify Email**:
   - Check your email inbox
   - Click the verification link sent by OpenWeatherMap

### Step 2: Get Your API Key

1. **Log In**:
   - Go to: https://home.openweathermap.org/
   - Log in with your credentials

2. **Navigate to API Keys**:
   - Click on your username (top right)
   - Select "API keys" from the dropdown menu
   - Or go directly to: https://home.openweathermap.org/api_keys

3. **Create API Key**:
   - You'll see a default key called "Default" (or create a new one)
   - If no key exists, click "Create Key"
   - Give it a name (e.g., "Premier League Predictor")
   - Click "Generate"

4. **Copy Your API Key**:
   - Your API key will look like: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`
   - **Important**: Keep this key secret! Don't share it publicly.

### Step 3: Set Up in Your Project

You have three options:

#### Option A: Environment Variable (Recommended)

```bash
# In your terminal:
export OPENWEATHER_API_KEY=your_actual_api_key_here

# To make it permanent, add to your ~/.zshrc or ~/.bashrc:
echo 'export OPENWEATHER_API_KEY=your_actual_api_key_here' >> ~/.zshrc
source ~/.zshrc
```

#### Option B: .env File (Recommended for Development)

1. **Create .env file** in project root:
   ```bash
   cd "/Users/mandarjoshi/Desktop/Cool_Coding_stuff/Machine Learning Projects/PremierLeagueChampions:Games predictor2025:26"
   echo "OPENWEATHER_API_KEY=your_actual_api_key_here" > .env
   ```

2. **Make sure .env is in .gitignore** (so you don't commit your key):
   ```bash
   echo ".env" >> .gitignore
   ```

#### Option C: Direct in Code (Not Recommended)

Only for testing - don't commit this to git!

```python
# In weather_api.py (temporary)
api_key = "your_actual_api_key_here"
```

### Step 4: Verify It Works

Test the API key:

```bash
# Test from command line:
curl "https://api.openweathermap.org/data/2.5/weather?q=London&appid=YOUR_API_KEY&units=metric"
```

Or test in Python:

```python
python3 -c "
from src.data_collection.weather_api import WeatherAPI
import os
api = WeatherAPI()
if api.api_key:
    print('✅ API key loaded successfully!')
    weather = api.get_weather_for_match('Arsenal', '2024-01-15')
    if weather:
        print(f'✅ Weather data retrieved: {weather[\"temperature\"]}°C')
    else:
        print('⚠️  Could not fetch weather (check API key)')
else:
    print('❌ No API key found. Set OPENWEATHER_API_KEY environment variable.')
"
```

## Free Tier Limits

- **1,000 API calls/day** (60 calls/minute)
- **Current weather only** (not historical)
- **No credit card required**
- **Perfect for development and testing**

## Troubleshooting

### "Invalid API key"
- Make sure you copied the entire key (no spaces)
- Wait 10 minutes after creating the key (activation delay)
- Check you're using the right key name

### "API key not found"
- Make sure you set the environment variable correctly
- Restart your terminal after setting the variable
- Check `.env` file is in the project root

### "401 Unauthorized"
- Your API key might not be activated yet (wait 10 minutes)
- Check you're using the correct key

### Rate Limiting
- Free tier: 60 calls/minute
- If you hit the limit, wait 1 minute and try again
- For production, consider upgrading to a paid plan

## Next Steps

Once your API key is set up:

1. **Run the pipeline**:
   ```bash
   ./run_pipeline_fixed.sh --features --train
   ```

2. **Weather features will be automatically included** in your model!

3. **Check the logs** - you should see:
   ```
   INFO: Adding weather features...
   INFO: Weather features added!
   ```

## Security Notes

⚠️ **Important**: 
- Never commit your API key to git
- Add `.env` to `.gitignore`
- Don't share your API key publicly
- If you accidentally commit it, regenerate a new key immediately

## Need Help?

- OpenWeatherMap Docs: https://openweathermap.org/api
- Support: https://openweathermap.org/faq
- API Status: https://status.openweathermap.org/

