"""
Fight History Scraper Module
Core scraping functionality for MMA fighter histories
"""
import os
import time
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, quote

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from bs4 import BeautifulSoup
except ImportError:
    # Graceful fallback if dependencies not installed
    requests = None
    HTTPAdapter = None
    Retry = None
    BeautifulSoup = None

from .config import load_config, ScraperConfig
from .validator import FightDataValidator, validate_fight_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FightHistoryScraper:
    """Scraper for fighter history from UFC Stats and other sources"""
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize scraper with configuration
        
        Args:
            config: Optional ScraperConfig object. If None, loads from config.yaml
        """
        self.config = config or load_config()
        self.validator = FightDataValidator(self.config)
        self.session = self._create_session()
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _create_session(self):
        """Create requests session with retry logic"""
        if requests is None:
            # Return None if requests not available
            return None
        
        session = requests.Session()
        
        # Configure retries
        if Retry is not None and HTTPAdapter is not None:
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.retry_backoff,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        return session
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # File handler
        log_path = os.path.join(self.config.data_dir, self.config.log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    def _get_cache_key(self, fighter_name: str) -> str:
        """Generate cache key for fighter"""
        return hashlib.md5(fighter_name.lower().encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get path to cache file"""
        return os.path.join(self.config.cache_dir, f'fight_history_{cache_key}.json')
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache file is valid and not expired"""
        if not os.path.exists(cache_path):
            return False
        
        # Check age
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        max_age = timedelta(days=self.config.cache_expiration_days)
        
        return cache_age < max_age
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Load fight history from cache"""
        cache_path = self._get_cache_path(cache_key)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded fight history from cache: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, fights: List[Dict[str, Any]]):
        """Save fight history to cache"""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(fights, f, indent=2, default=str)
            logger.debug(f"Saved fight history to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def scrape_fight_history(self, fighter_url: str, fighter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scrape fight history for a fighter from their profile URL
        
        Args:
            fighter_url: URL to fighter's profile page
            fighter_name: Optional name of the fighter (for logging)
        
        Returns:
            List of fight dictionaries
        """
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 library not available. Install with: pip install beautifulsoup4")
        
        logger.info(f"Scraping fight history for {fighter_name or 'fighter'}: {fighter_url}")
        
        try:
            # Fetch page with timeout
            response = self.session.get(fighter_url, timeout=self.config.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find fight history table
            fights = self._parse_fight_history(soup, fighter_name)
            
            logger.info(f"Scraped {len(fights)} fights for {fighter_name}")
            
            # Rate limiting
            time.sleep(self.config.delay_between_requests)
            
            return fights
            
        except requests.RequestException as e:
            logger.error(f"Error fetching {fighter_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing fight history: {e}")
            return []
    
    def _parse_fight_history(self, soup: BeautifulSoup, fighter_name: Optional[str]) -> List[Dict[str, Any]]:
        """
        Parse fight history from BeautifulSoup object
        
        Args:
            soup: BeautifulSoup parsed HTML
            fighter_name: Name of the fighter
        
        Returns:
            List of fight dictionaries
        """
        fights = []
        
        # Find fight history table body
        table_selector = self.config.selectors.get('fight_history_table', 'tbody.b-fight-details__table-body')
        table = soup.select_one(table_selector)
        
        if not table:
            logger.warning("Fight history table not found")
            return []
        
        # Find all fight rows
        row_selector = self.config.selectors.get('fight_rows', 'tr.b-fight-details__table-row')
        rows = table.select(row_selector)
        
        for row in rows:
            try:
                fight = self._parse_fight_row(row, fighter_name)
                if fight:
                    fights.append(fight)
            except Exception as e:
                logger.warning(f"Error parsing fight row: {e}")
                continue
        
        return fights
    
    def _parse_fight_row(self, row, fighter_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse a single fight row"""
        fight = {}
        
        # Get all cells
        cells = row.find_all(['td', 'th'])
        
        if len(cells) < 7:
            return None
        
        try:
            # Result (Win/Loss/Draw)
            result_cell = cells[0]
            result_icon = result_cell.find('i')
            if result_icon:
                result_class = result_icon.get('class', [])
                if 'b-flag_style_green' in result_class or 'win' in str(result_class).lower():
                    fight['result'] = 'Win'
                elif 'b-flag_style_red' in result_class or 'loss' in str(result_class).lower():
                    fight['result'] = 'Loss'
                else:
                    fight['result'] = 'Draw'
            
            # Opponent name
            opponent_link = cells[1].find('a')
            if opponent_link:
                fight['opponent_name'] = opponent_link.text.strip()
            
            # Event name and date
            event_link = cells[2].find('a')
            if event_link:
                fight['event_name'] = event_link.text.strip()
            
            event_date_span = cells[2].find('span')
            if event_date_span:
                date_text = event_date_span.text.strip()
                fight['event_date'] = self._parse_date(date_text)
            
            # Strikes (if available)
            if len(cells) > 3:
                strikes_text = cells[3].text.strip()
                fight['sig_strikes_landed'], fight['sig_strikes_received'] = self._parse_strikes(strikes_text)
            
            # Method
            if len(cells) > 6:
                method_text = cells[6].text.strip()
                fight['win_method'] = method_text
                fight['fight_ending_technique'] = method_text
            
            # Round
            if len(cells) > 7:
                round_text = cells[7].text.strip()
                try:
                    fight['round_number'] = int(round_text)
                except (ValueError, TypeError):
                    fight['round_number'] = None
            
            # Time
            if len(cells) > 8:
                time_text = cells[8].text.strip()
                fight['fight_duration_seconds'] = self._parse_time_to_seconds(time_text, fight.get('round_number'))
            
            # Add fighter name
            fight['fighter_name'] = fighter_name
            
            # Add scraped timestamp
            fight['scraped_at'] = datetime.now().isoformat()
            
            # Calculate data quality score
            fight['data_quality_score'] = self._calculate_quality_score(fight)
            
            return fight
            
        except Exception as e:
            logger.warning(f"Error extracting fight data: {e}")
            return None
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date string to standard format"""
        try:
            # Try common formats
            for fmt in ['%b. %d, %Y', '%B %d, %Y', '%Y-%m-%d', '%m/%d/%Y']:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime(self.config.date_format)
                except ValueError:
                    continue
            
            # If no format works, return as-is
            return date_str
        except Exception:
            return date_str
    
    def _parse_strikes(self, strikes_text: str) -> tuple:
        """Parse strikes text like '45 of 89' or '45/89'"""
        try:
            # Handle "X of Y" or "X/Y" format
            if ' of ' in strikes_text:
                parts = strikes_text.split(' of ')
            elif '/' in strikes_text:
                parts = strikes_text.split('/')
            else:
                return None, None
            
            landed = int(parts[0].strip())
            attempted = int(parts[1].strip())
            received = attempted - landed  # Approximation
            
            return landed, received
        except Exception:
            return None, None
    
    def _parse_time_to_seconds(self, time_str: str, round_num: Optional[int]) -> Optional[int]:
        """Convert fight time to seconds"""
        try:
            # Format: "M:SS"
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                
                # Add previous rounds (each round is 5 minutes)
                if round_num:
                    total_seconds = (round_num - 1) * 300 + minutes * 60 + seconds
                else:
                    total_seconds = minutes * 60 + seconds
                
                return total_seconds
        except Exception:
            pass
        
        return None
    
    def _calculate_quality_score(self, fight: Dict[str, Any]) -> float:
        """Calculate data quality score for a fight record"""
        score = 1.0
        
        # Deduct for missing fields
        important_fields = ['opponent_name', 'event_date', 'event_name', 'win_method']
        for field in important_fields:
            if not fight.get(field):
                score -= 0.1
        
        # Bonus for having detailed stats
        if fight.get('sig_strikes_landed') is not None:
            score += 0.05
        if fight.get('fight_duration_seconds') is not None:
            score += 0.05
        
        return max(0.0, min(1.0, score))


def scrape_fight_history(fighter_url: str, fighter_name: Optional[str] = None, 
                        config: Optional[ScraperConfig] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to scrape fight history for a fighter
    
    Args:
        fighter_url: URL to fighter's profile page
        fighter_name: Optional name of the fighter
        config: Optional ScraperConfig object
    
    Returns:
        List of validated and cleaned fight dictionaries
    """
    scraper = FightHistoryScraper(config)
    
    # Check cache first
    if scraper.config.use_cache and fighter_name:
        cache_key = scraper._get_cache_key(fighter_name)
        cached_fights = scraper._load_from_cache(cache_key)
        if cached_fights:
            logger.info(f"Using cached fight history for {fighter_name}")
            return cached_fights
    
    # Scrape fresh data
    fights = scraper.scrape_fight_history(fighter_url, fighter_name)
    
    # Validate and clean
    if scraper.config.validate_on_scrape:
        fights = validate_fight_data(fights, scraper.config)
    
    # Save to cache
    if scraper.config.use_cache and fighter_name:
        cache_key = scraper._get_cache_key(fighter_name)
        scraper._save_to_cache(cache_key, fights)
    
    return fights


if __name__ == '__main__':
    # Test scraper
    config = load_config()
    scraper = FightHistoryScraper(config)
    
    # Example: scrape a fighter (would need actual URL)
    # fights = scraper.scrape_fight_history("http://ufcstats.com/fighter-details/...", "Test Fighter")
    # print(f"Scraped {len(fights)} fights")
    
    print("Fight history scraper initialized successfully")
    print(f"Config: Base URL = {config.base_url}")
    print(f"Config: Cache enabled = {config.use_cache}")
