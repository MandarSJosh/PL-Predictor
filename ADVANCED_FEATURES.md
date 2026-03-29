# Advanced Features Guide

This document explains the advanced features that have been added to improve model accuracy.

## Features Added

### 1. Expected Goals (xG)
- **Source**: FBref match pages
- **Features**:
  - `home_xg`: Home team expected goals
  - `away_xg`: Away team expected goals
  - `xg_difference`: Difference in xG
  - `home_avg_xg`: Rolling average xG for home team
  - `away_avg_xg`: Rolling average xG for away team

**Status**: Framework implemented. xG data extraction from FBref requires visiting individual match pages, which is slow. The framework is ready but will return None values until match URLs are properly extracted.

### 2. Referee Statistics
- **Source**: FBref match pages
- **Features**:
  - `referee`: Referee name
  - `referee_experience`: Number of matches officiated (proxy)
  - `referee_strictness`: Referee strictness score (placeholder - needs historical data)

**Status**: Framework implemented. Referee names are extracted from match pages. Historical referee statistics (cards per game, etc.) would need to be collected separately.

### 3. Weather Conditions
- **Source**: OpenWeatherMap API
- **Features**:
  - `weather_temperature`: Temperature in Celsius
  - `weather_humidity`: Humidity percentage
  - `weather_wind_speed`: Wind speed in m/s
  - `weather_is_rainy`: Binary indicator for rain
  - `weather_is_snowy`: Binary indicator for snow
  - `weather_is_clear`: Binary indicator for clear weather

**Setup Required**:
1. Get a free API key from: https://openweathermap.org/api
2. Set environment variable: `export OPENWEATHER_API_KEY=your_key_here`
3. Or create a `.env` file: `OPENWEATHER_API_KEY=your_key_here`

**Note**: Free tier provides current weather only. Historical weather data requires a paid subscription.

### 4. Player Injuries & Suspensions
- **Source**: Framework ready (needs data source)
- **Features**:
  - `home_key_players_injured`: Number of key players injured
  - `away_key_players_injured`: Number of key players injured
  - `home_key_players_suspended`: Number of key players suspended
  - `away_key_players_suspended`: Number of key players suspended
  - `injury_difference`: Difference in injuries between teams

**Status**: Framework implemented with default values (0). To get real data, you would need to:
- Scrape news sites (BBC Sport, Sky Sports)
- Use Transfermarkt API
- Use Premier League official API
- Use paid sports data APIs (Opta, StatsBomb)
- Manually collect and load via CSV/JSON

### 5. Manager Quality Signals
- **Source**: `data/managers_2025_26.csv`
- **Features**:
  - `home_manager_win_pct` / `away_manager_win_pct`: Career PPG (SofaScore)
  - `home_manager_trophies` / `away_manager_trophies`: Major trophies (definition in `data/manager_stats_sources.md`)

**Status**: Implemented via CSV mapping. Update the CSV as managers change.

### 6. Squad Value (Transfermarkt)
- **Source**: Transfermarkt squad values
- **Features**:
  - `home_squad_value_eur`, `away_squad_value_eur`, `squad_value_diff`

**Setup**:
```bash
python3 collect_team_values.py
```

## Usage

All advanced features are automatically included when running the pipeline:

```bash
./run_pipeline_fixed.sh --features --train
```

The features will be added with default/placeholder values if data sources are unavailable.

## Configuration

### Weather API Setup

1. **Get API Key**:
   ```bash
   # Visit https://openweathermap.org/api and sign up for free tier
   ```

2. **Set Environment Variable**:
   ```bash
   export OPENWEATHER_API_KEY=your_api_key_here
   ```

3. **Or use .env file**:
   ```bash
   echo "OPENWEATHER_API_KEY=your_api_key_here" > .env
   ```

### Injury Data Setup

To use real injury data, you can:

1. **Load from CSV**:
   ```python
   from src.data_collection.injury_tracker import InjuryTracker
   tracker = InjuryTracker()
   tracker.load_injury_data('path/to/injuries.csv')
   ```

2. **CSV Format**:
   ```csv
   date,team,key_players_injured,total_injured,key_players_suspended
   2024-01-15,Arsenal,1,2,0
   2024-01-15,Chelsea,0,1,1
   ```

## Feature Impact

These advanced features should improve model accuracy by:
- **xG**: Captures match quality beyond just goals
- **Referee**: Accounts for officiating style differences
- **Weather**: Affects playing conditions and style
- **Injuries**: Critical for team strength assessment
- **Squad Value**: Proxy for overall team quality and depth
- **Manager Stats**: Captures tactical quality and experience

Expected improvement: +2-5% accuracy when all features have real data.

## Limitations

1. **xG Data**: Requires visiting each match page (slow)
2. **Weather**: Free tier only provides current weather, not historical
3. **Injuries**: Requires external data source (not automatically scraped)
4. **Referee Stats**: Needs historical data collection
5. **Squad Values**: Requires Transfermarkt scrape to populate `data/team_values.csv`

## Next Steps

To fully utilize these features:
1. Set up OpenWeatherMap API key for weather data
2. Implement xG extraction from match URLs
3. Collect or integrate injury data source
4. Build referee statistics database
5. Run `python3 collect_team_values.py` to populate squad values

The framework is ready - just needs data sources!

