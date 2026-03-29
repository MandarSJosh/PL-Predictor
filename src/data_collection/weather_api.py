"""
Weather API Integration (lightweight)
Uses OpenWeatherMap current conditions as a proxy.
"""

import logging
import os
from typing import Optional, Dict

import pandas as pd
import requests

from src.utils.teams import load_teams_list, normalize_team_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAPI:
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    STADIUM_COORDS = {
        "Arsenal": {"lat": 51.5550, "lon": -0.1086},
        "Chelsea": {"lat": 51.4817, "lon": -0.1910},
        "Liverpool": {"lat": 53.4308, "lon": -2.9608},
        "Manchester City": {"lat": 53.4831, "lon": -2.2004},
        "Manchester United": {"lat": 53.4631, "lon": -2.2913},
        "Tottenham": {"lat": 51.6043, "lon": -0.0664},
        "Newcastle": {"lat": 54.9756, "lon": -1.6219},
        "Brighton": {"lat": 50.8619, "lon": -0.0833},
        "West Ham": {"lat": 51.5386, "lon": -0.0164},
        "Aston Villa": {"lat": 52.5092, "lon": -1.8847},
        "Crystal Palace": {"lat": 51.3983, "lon": -0.0855},
        "Everton": {"lat": 53.4389, "lon": -2.9664},
        "Fulham": {"lat": 51.4749, "lon": -0.2216},
        "Leeds": {"lat": 53.7777, "lon": -1.5722},
        "Leicester": {"lat": 52.6203, "lon": -1.1422},
        "Southampton": {"lat": 50.9058, "lon": -1.3906},
        "Wolves": {"lat": 52.5903, "lon": -2.1304},
        "Burnley": {"lat": 53.7890, "lon": -2.2302},
        "Watford": {"lat": 51.6498, "lon": -0.4015},
        "Norwich": {"lat": 52.6223, "lon": 1.3090},
        "Brentford": {"lat": 51.4908, "lon": -0.2886},
        "Bournemouth": {"lat": 50.7353, "lon": -1.8383},
        "Sheffield United": {"lat": 53.3703, "lon": -1.4708},
        "Luton": {"lat": 51.8847, "lon": -0.4319},
        "Nottingham Forest": {"lat": 52.9400, "lon": -1.1322},
        "Ipswich": {"lat": 52.0547, "lon": 1.1444},
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self._team_cache: Dict[str, Dict] = {}
        self._valid_teams = set(load_teams_list())
        if not self.api_key:
            logger.warning("No OpenWeather API key found; weather features set to 0.")

    def _normalize_team(self, team_name: str) -> str:
        return normalize_team_name(team_name)

    def _get_stadium_coords(self, team_name: str) -> Dict[str, float]:
        normalized = self._normalize_team(team_name)
        if self._valid_teams and normalized not in self._valid_teams:
            return {"lat": 51.5074, "lon": -0.1278}
        if normalized in self.STADIUM_COORDS:
            return self.STADIUM_COORDS[normalized]
        for stadium_team, coords in self.STADIUM_COORDS.items():
            if normalized.lower() in stadium_team.lower() or stadium_team.lower() in normalized.lower():
                return coords
        return {"lat": 51.5074, "lon": -0.1278}

    def get_weather_for_team(self, team_name: str) -> Optional[Dict]:
        if not self.api_key:
            return None
        normalized = self._normalize_team(team_name)
        if normalized in self._team_cache:
            return self._team_cache[normalized]
        coords = self._get_stadium_coords(normalized)
        try:
            response = requests.get(
                f"{self.BASE_URL}/weather",
                params={"lat": coords["lat"], "lon": coords["lon"], "appid": self.api_key, "units": "metric"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            weather_data = {
                "temperature": data.get("main", {}).get("temp", 0.0),
                "humidity": data.get("main", {}).get("humidity", 0.0),
                "wind_speed": data.get("wind", {}).get("speed", 0.0),
                "weather_main": data.get("weather", [{}])[0].get("main", "").lower(),
            }
            weather_data["is_rainy"] = float("rain" in weather_data["weather_main"])
            weather_data["is_snowy"] = float("snow" in weather_data["weather_main"])
            weather_data["is_clear"] = float(weather_data["weather_main"] == "clear")
            self._team_cache[normalized] = weather_data
            return weather_data
        except Exception as exc:
            logger.warning(f"Weather lookup failed for {team_name}: {exc}")
            return None

    def add_weather_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        if not self.api_key:
            for col in [
                "weather_temperature",
                "weather_humidity",
                "weather_wind_speed",
                "weather_is_rainy",
                "weather_is_snowy",
                "weather_is_clear",
            ]:
                matches_df[col] = 0.0
            return matches_df

        logger.info("Adding weather features (current conditions as proxy)...")
        temps = []
        hums = []
        winds = []
        rainy = []
        snowy = []
        clear = []
        for _, row in matches_df.iterrows():
            team = row.get("home_team", "")
            weather = self.get_weather_for_team(str(team)) or {}
            temps.append(weather.get("temperature", 0.0))
            hums.append(weather.get("humidity", 0.0))
            winds.append(weather.get("wind_speed", 0.0))
            rainy.append(weather.get("is_rainy", 0.0))
            snowy.append(weather.get("is_snowy", 0.0))
            clear.append(weather.get("is_clear", 0.0))
        matches_df["weather_temperature"] = temps
        matches_df["weather_humidity"] = hums
        matches_df["weather_wind_speed"] = winds
        matches_df["weather_is_rainy"] = rainy
        matches_df["weather_is_snowy"] = snowy
        matches_df["weather_is_clear"] = clear
        return matches_df
