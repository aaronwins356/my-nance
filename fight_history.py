"""
Fight History Module
Functions to load and display detailed fight history for fighters
"""
import os
import pandas as pd
from typing import Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_fight_history() -> pd.DataFrame:
    """
    Load detailed fight history for all fighters
    
    Returns:
        DataFrame with fight history data
    """
    history_file = os.path.join(DATA_DIR, 'fight_history.csv')
    
    if not os.path.exists(history_file):
        print("Warning: Fight history file not found")
        return pd.DataFrame()
    
    return pd.read_csv(history_file)


def get_fighter_history(fighter_name: str, fight_history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Get complete fight history for a specific fighter
    
    Args:
        fighter_name: Name of the fighter
        fight_history_df: Optional pre-loaded fight history DataFrame
    
    Returns:
        DataFrame with all fights for the specified fighter
    """
    if fight_history_df is None:
        fight_history_df = load_fight_history()
    
    if fight_history_df.empty:
        return pd.DataFrame()
    
    # Get all fights where the fighter participated
    fighter_fights = fight_history_df[
        fight_history_df['fighter_name'] == fighter_name
    ].copy()
    
    # Sort by date (most recent first)
    if 'event_date' in fighter_fights.columns:
        fighter_fights['event_date'] = pd.to_datetime(fighter_fights['event_date'])
        fighter_fights = fighter_fights.sort_values('event_date', ascending=False)
    
    return fighter_fights


def format_fight_duration(duration_seconds: int) -> str:
    """
    Format fight duration from seconds to MM:SS format
    
    Args:
        duration_seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "2:03")
    """
    if pd.isna(duration_seconds):
        return "N/A"
    
    try:
        duration = int(duration_seconds)
        if duration < 0:
            return "N/A"
        minutes = duration // 60
        seconds = duration % 60
        return f"{minutes}:{seconds:02d}"
    except (ValueError, TypeError):
        return "N/A"


def get_fighter_record_summary(fighter_name: str, fight_history_df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """
    Get summary statistics for a fighter's record
    
    Args:
        fighter_name: Name of the fighter
        fight_history_df: Optional pre-loaded fight history DataFrame
    
    Returns:
        Dictionary with wins, losses, and method breakdown
    """
    fighter_fights = get_fighter_history(fighter_name, fight_history_df)
    
    if fighter_fights.empty:
        return {
            'total_fights': 0,
            'wins': 0,
            'losses': 0,
            'ko_tko_wins': 0,
            'submission_wins': 0,
            'decision_wins': 0
        }
    
    total_fights = len(fighter_fights)
    wins = len(fighter_fights[fighter_fights['result'] == 'Win'])
    losses = total_fights - wins
    
    win_fights = fighter_fights[fighter_fights['result'] == 'Win']
    ko_tko_wins = len(win_fights[win_fights['win_method'] == 'KO/TKO'])
    submission_wins = len(win_fights[win_fights['win_method'] == 'Submission'])
    decision_wins = len(win_fights[win_fights['win_method'] == 'Decision'])
    
    return {
        'total_fights': total_fights,
        'wins': wins,
        'losses': losses,
        'ko_tko_wins': ko_tko_wins,
        'submission_wins': submission_wins,
        'decision_wins': decision_wins
    }


def scrape_and_update_fight_history(fighter_list_df: pd.DataFrame = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Scrape real fight history data for fighters
    
    This function replaces the old generate_sample_fight_history with real scraping.
    It uses the scraper module to fetch actual fight histories from UFC Stats.
    
    Args:
        fighter_list_df: DataFrame with fighter information (names and optionally URLs)
        use_cache: Whether to use cached data if available
    
    Returns:
        DataFrame with scraped fight history
    """
    try:
        from scraper import scrape_fight_history, load_config
        from scraper.fight_scraper import FightHistoryScraper
        
        config = load_config()
        config.use_cache = use_cache
        scraper = FightHistoryScraper(config)
        
        all_fights = []
        
        if fighter_list_df is not None:
            print(f"Scraping fight histories for {len(fighter_list_df)} fighters...")
            
            for idx, fighter in fighter_list_df.iterrows():
                fighter_name = fighter.get('name', fighter.get('fighter_name', ''))
                fighter_url = fighter.get('fighter_url', fighter.get('url', ''))
                
                if not fighter_name:
                    continue
                
                try:
                    if fighter_url:
                        fights = scrape_fight_history(fighter_url, fighter_name, config)
                        all_fights.extend(fights)
                        print(f"  Scraped {len(fights)} fights for {fighter_name}")
                    else:
                        print(f"  No URL for {fighter_name}, skipping")
                except Exception as e:
                    print(f"  Error scraping {fighter_name}: {e}")
                    continue
        
        if not all_fights:
            print("Warning: No fight data scraped. Check fighter list and URLs.")
            return pd.DataFrame()
        
        return pd.DataFrame(all_fights)
        
    except ImportError as e:
        print(f"Error: Could not import scraper module: {e}")
        print("Make sure to install dependencies: pip install requests beautifulsoup4 pyyaml")
        return pd.DataFrame()


if __name__ == '__main__':
    # Test the module
    history = load_fight_history()
    print(f"Loaded {len(history)} fight records")
    
    if not history.empty:
        # Show example
        fighter = history['fighter_name'].iloc[0]
        fighter_hist = get_fighter_history(fighter, history)
        print(f"\nFight history for {fighter}:")
        print(fighter_hist.head())
