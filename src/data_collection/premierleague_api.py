"""
PremierLeague.com (pulselive) API client.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.utils.teams import normalize_team_name, load_teams_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PremierLeagueAPIScraper:
    BASE_URL = "https://footballapi.pulselive.com/football"

    DEFAULT_HEADERS = {
        "Origin": "https://www.premierleague.com",
        "Referer": "https://www.premierleague.com/",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

    def _request(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.BASE_URL}{path}"
        response = self.session.get(url, params=params or {}, timeout=20)
        response.raise_for_status()
        return response.json()

    def _season_label(self, season: str) -> str:
        if "/" in season:
            return season
        parts = season.split("-")
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1][2:]}"
        return season

    def _get_comp_seasons(self) -> List[Dict]:
        data = self._request("/competitions/1/compseasons", params={"pageSize": 100})
        return data.get("compSeasons", []) or data.get("content", [])

    def _find_season_id(self, season: str) -> Optional[int]:
        label = self._season_label(season)
        for item in self._get_comp_seasons():
            if str(item.get("label", "")).strip() == label:
                return int(item.get("id"))
        # fallback: try numeric year match
        for item in self._get_comp_seasons():
            if label.startswith(str(item.get("season", ""))):
                return int(item.get("id"))
        return None

    def get_season_teams(self, season: str) -> List[str]:
        season_id = self._find_season_id(season)
        if not season_id:
            logger.warning(f"Could not resolve season id for {season}")
            return []
        data = self._request("/teams", params={"compSeasons": season_id, "pageSize": 100})
        teams = data.get("teams", []) or data.get("content", [])
        names = [normalize_team_name(t.get("name", "")) for t in teams if t.get("name")]
        return sorted({n for n in names if n})

    def get_remaining_fixtures(self, season: str) -> pd.DataFrame:
        season_id = self._find_season_id(season)
        if not season_id:
            logger.warning(f"Could not resolve season id for {season}")
            return pd.DataFrame()

        fixtures: List[Dict] = []
        page = 0
        while True:
            data = self._request(
                "/fixtures",
                params={
                    "comps": 1,
                    "compSeasons": season_id,
                    "page": page,
                    "pageSize": 200,
                    "statuses": "U",
                },
            )
            content = data.get("content", []) or data.get("fixtures", [])
            fixtures.extend(content)
            page_info = data.get("pageInfo", {})
            if not page_info or page + 1 >= page_info.get("numPages", 1):
                break
            page += 1

        valid_teams = set(load_teams_list())
        rows = []
        for fixture in fixtures:
            teams = fixture.get("teams", [])
            home_team = ""
            away_team = ""
            for idx, team in enumerate(teams):
                side = team.get("side")
                name = normalize_team_name(team.get("team", {}).get("name", ""))
                if side == "home" or (side is None and idx == 0):
                    home_team = name
                elif side == "away" or (side is None and idx == 1):
                    away_team = name

            kickoff = fixture.get("kickoff", {}) or fixture.get("kickoffTime", {})
            date_val = kickoff.get("label") or kickoff.get("iso") or kickoff.get("dateTime")
            matchweek = fixture.get("gameweek") or fixture.get("matchweek", {}).get("week")

            if valid_teams and (home_team not in valid_teams or away_team not in valid_teams):
                continue
            rows.append(
                {
                    "fixture_id": fixture.get("id"),
                    "date": date_val,
                    "home_team": home_team,
                    "away_team": away_team,
                    "matchweek": matchweek,
                    "season": season,
                    "status": fixture.get("status"),
                }
            )

        return pd.DataFrame(rows)

    def save_remaining_fixtures(self, season: str, data_dir: str = "data") -> Path:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        fixtures_df = self.get_remaining_fixtures(season)
        output = data_path / "remaining_fixtures.csv"
        fixtures_df.to_csv(output, index=False)
        logger.info(f"Saved {len(fixtures_df)} remaining fixtures to {output}")
        return output

    def save_season_teams(self, season: str, data_dir: str = "data") -> Path:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        teams = self.get_season_teams(season)
        output = data_path / "teams_2025_26.txt"
        output.write_text("\n".join(teams) + "\n")
        logger.info(f"Saved {len(teams)} teams to {output}")
        return output

    def get_season_matches(self, season: str) -> pd.DataFrame:
        season_id = self._find_season_id(season)
        if not season_id:
            logger.warning(f"Could not resolve season id for {season}")
            return pd.DataFrame()

        fixtures: List[Dict] = []
        page = 0
        while True:
            data = self._request(
                "/fixtures",
                params={
                    "compSeasons": season_id,
                    "page": page,
                    "pageSize": 200,
                    "statuses": "C",
                },
            )
            content = data.get("content", []) or data.get("fixtures", [])
            fixtures.extend(content)
            page_info = data.get("pageInfo", {})
            if not page_info or page + 1 >= page_info.get("numPages", 1):
                break
            page += 1

        rows = []
        for fixture in fixtures:
            teams = fixture.get("teams", [])
            home_team = ""
            away_team = ""
            home_score = None
            away_score = None
            for idx, team in enumerate(teams):
                side = team.get("side")
                name = normalize_team_name(team.get("team", {}).get("name", ""))
                score = team.get("score")
                if side == "home" or (side is None and idx == 0):
                    home_team = name
                    home_score = score
                elif side == "away" or (side is None and idx == 1):
                    away_team = name
                    away_score = score

            kickoff = fixture.get("kickoff", {}) or fixture.get("kickoffTime", {})
            date_val = kickoff.get("label") or kickoff.get("iso") or kickoff.get("dateTime")

            rows.append(
                {
                    "date": date_val,
                    "home_team": home_team,
                    "away_team": away_team,
                    "season": season,
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )

        return pd.DataFrame(rows)

    def get_historical_seasons(
        self, start_season: str, end_season: str, force: bool = False
    ) -> pd.DataFrame:
        data_path = Path("data") / "raw_matches.csv"
        if data_path.exists() and not force:
            logger.info(f"Using existing data from {data_path}")
            return pd.read_csv(data_path)

        def season_range(start: str, end: str) -> List[str]:
            start_year = int(start.split("-")[0])
            end_year = int(end.split("-")[0])
            return [f"{y}-{y+1}" for y in range(start_year, end_year + 1)]

        all_rows = []
        for season in season_range(start_season, end_season):
            season_df = self.get_season_matches(season)
            if not season_df.empty:
                all_rows.append(season_df)

        if not all_rows:
            logger.warning("No PL API data collected; returning empty dataframe.")
            return pd.DataFrame()

        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(data_path, index=False)
        logger.info(f"Saved {len(combined)} matches to {data_path}")
        return combined
