"""
Configuration Module
Loads and manages scraper configuration from YAML file
"""
import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ScraperConfig:
    """Configuration for fight history scraper"""
    
    # Scraping settings
    base_url: str = "http://ufcstats.com"
    search_url: str = "http://ufcstats.com/statistics/fighters/search"
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 2.0
    user_agent: str = "Mozilla/5.0"
    delay_between_requests: float = 1.0
    delay_between_fighters: float = 2.0
    
    # Selectors
    selectors: Dict[str, str] = field(default_factory=dict)
    
    # Validation rules
    valid_results: List[str] = field(default_factory=lambda: ["Win", "Loss", "Draw", "NC", "DQ"])
    valid_methods: List[str] = field(default_factory=lambda: ["KO/TKO", "Submission", "Decision"])
    date_format: str = "%Y-%m-%d"
    min_year: int = 1993
    max_year: int = 2030
    min_round: int = 1
    max_round: int = 5
    
    # Output settings
    data_dir: str = "data"
    cache_dir: str = "cache"
    fight_history_file: str = "fight_history.csv"
    cache_expiration_days: int = 30
    
    # Feature flags
    use_cache: bool = True
    save_html_cache: bool = True
    validate_on_scrape: bool = True
    normalize_text: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "scraper.log"
    log_to_console: bool = True
    
    def __post_init__(self):
        """Initialize defaults after dataclass creation"""
        if not self.valid_results:
            self.valid_results = ["Win", "Loss", "Draw", "NC", "DQ"]
        if not self.valid_methods:
            self.valid_methods = ["KO/TKO", "Submission", "Decision"]


def load_config(config_path: Optional[str] = None) -> ScraperConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        ScraperConfig object with loaded settings
    """
    if config_path is None:
        # Default to config.yaml in project root
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config.yaml'
        )
    
    # Return default config if file doesn't exist
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return ScraperConfig()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Extract relevant sections
        scraping = config_data.get('scraping', {})
        selectors = config_data.get('selectors', {})
        validation = config_data.get('validation', {})
        output = config_data.get('output', {})
        logging_config = config_data.get('logging', {})
        features = config_data.get('features', {})
        
        # Create default config for fallback values
        default = ScraperConfig()
        
        # Build config object
        config = ScraperConfig(
            # Scraping settings
            base_url=scraping.get('base_url', default.base_url),
            search_url=scraping.get('search_url', default.search_url),
            timeout=scraping.get('timeout', default.timeout),
            max_retries=scraping.get('max_retries', default.max_retries),
            retry_backoff=scraping.get('retry_backoff', default.retry_backoff),
            user_agent=scraping.get('user_agent', default.user_agent),
            delay_between_requests=scraping.get('delay_between_requests', default.delay_between_requests),
            delay_between_fighters=scraping.get('delay_between_fighters', default.delay_between_fighters),
            
            # Selectors
            selectors=selectors,
            
            # Validation
            valid_results=validation.get('valid_results', default.valid_results),
            valid_methods=validation.get('valid_methods', default.valid_methods),
            date_format=validation.get('date_format', default.date_format),
            min_year=validation.get('min_year', default.min_year),
            max_year=validation.get('max_year', default.max_year),
            min_round=validation.get('min_round', default.min_round),
            max_round=validation.get('max_round', default.max_round),
            
            # Output
            data_dir=output.get('data_dir', default.data_dir),
            cache_dir=output.get('cache_dir', default.cache_dir),
            fight_history_file=output.get('fight_history_file', default.fight_history_file),
            cache_expiration_days=output.get('cache_expiration_days', default.cache_expiration_days),
            
            # Features
            use_cache=features.get('use_cache', default.use_cache),
            save_html_cache=features.get('save_html_cache', default.save_html_cache),
            validate_on_scrape=features.get('validate_on_scrape', default.validate_on_scrape),
            normalize_text=features.get('normalize_text', default.normalize_text),
            
            # Logging
            log_level=logging_config.get('level', default.log_level),
            log_file=logging_config.get('file', default.log_file),
            log_to_console=logging_config.get('console', default.log_to_console),
        )
        
        return config
        
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Using default configuration")
        return ScraperConfig()


def get_selector(config: ScraperConfig, selector_name: str) -> Optional[str]:
    """
    Get a CSS selector from config
    
    Args:
        config: ScraperConfig object
        selector_name: Name of the selector
    
    Returns:
        Selector string or None if not found
    """
    return config.selectors.get(selector_name)


if __name__ == '__main__':
    # Test configuration loading
    config = load_config()
    print(f"Loaded configuration:")
    print(f"  Base URL: {config.base_url}")
    print(f"  Cache enabled: {config.use_cache}")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Data directory: {config.data_dir}")
