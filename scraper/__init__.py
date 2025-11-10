"""
Scraper Module
Centralized web scraping functionality for MMA fight history
"""
from .fight_scraper import FightHistoryScraper, scrape_fight_history
from .config import load_config, ScraperConfig
from .validator import FightDataValidator, validate_fight_data

__all__ = [
    'FightHistoryScraper',
    'scrape_fight_history',
    'load_config',
    'ScraperConfig',
    'FightDataValidator',
    'validate_fight_data'
]

__version__ = '1.0.0'
