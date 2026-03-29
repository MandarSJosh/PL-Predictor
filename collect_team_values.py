"""
Fetch squad values from Transfermarkt (Premier League) and save to data/team_values.csv.
"""

import logging
import re
from pathlib import Path

import pandas as pd
import requests

from src.utils.teams import normalize_team_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_value(raw: str) -> float:
    value = raw.replace("€", "").strip()
    if value.endswith("bn"):
        return float(value[:-2]) * 1_000_000_000
    if value.endswith("m"):
        return float(value[:-1]) * 1_000_000
    if value.endswith("k"):
        return float(value[:-1]) * 1_000
    return 0.0


def main() -> None:
    url = "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.transfermarkt.com/",
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    html = response.text

    rows = []
    # Extract team and squad value from table rows
    row_pattern = re.compile(
        r'<td class="hauptlink[^"]*"><a[^>]+>(?P<team>[^<]+)</a>.*?'
        r'<td class="rechts.*?">(?P<value>€[^<]+)</td>',
        re.DOTALL,
    )
    for match in row_pattern.finditer(html):
        team = normalize_team_name(match.group("team"))
        value = _parse_value(match.group("value"))
        if team and value > 0:
            rows.append({"team": team, "squad_value_eur": value})

    if not rows:
        logger.warning("Could not parse team values from Transfermarkt.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])
    output = Path("data") / "team_values.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info(f"Saved {len(df)} team values to {output}")


if __name__ == "__main__":
    main()
