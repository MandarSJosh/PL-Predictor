"""
Fetch remaining fixtures and confirmed teams for 2025/26.
"""

import logging
from src.data_collection.premierleague_api import PremierLeagueAPIScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(season: str = "2025-2026") -> None:
    scraper = PremierLeagueAPIScraper()
    scraper.save_remaining_fixtures(season, "data")
    scraper.save_season_teams(season, "data")


if __name__ == "__main__":
    main()
