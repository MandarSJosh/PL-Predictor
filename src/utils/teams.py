"""
Team name utilities and shared list loader.
"""

from pathlib import Path
from typing import List


TEAM_ALIASES = {
    "Manchester Utd": "Manchester United",
    "Man United": "Manchester United",
    "Manchester City": "Manchester City",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Sheffield Utd": "Sheffield United",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
}


def normalize_team_name(team_name: str) -> str:
    cleaned = team_name.strip()
    cleaned = cleaned.replace(" FC", "").replace("FC ", "").replace(".", "")
    return TEAM_ALIASES.get(cleaned, cleaned)


def load_teams_list(data_dir: str = "data", filename: str = "teams_2025_26.txt") -> List[str]:
    path = Path(data_dir) / filename
    if not path.exists():
        return []
    teams = [normalize_team_name(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted({t for t in teams if t})
