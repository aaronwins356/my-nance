#!/usr/bin/env python3
"""
Collect Fight Data
Main entry point for scraping and collecting fight history data
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from scraper import FightHistoryScraper, load_config, scrape_fight_history
    from scraper.validator import validate_fight_data
except ImportError:
    print("Error: Could not import scraper modules")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_fighter_list(filepath: str) -> pd.DataFrame:
    """
    Load fighter list from CSV file
    
    Args:
        filepath: Path to CSV with fighter information
    
    Returns:
        DataFrame with fighter data
    """
    if not os.path.exists(filepath):
        logger.error(f"Fighter list not found: {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} fighters from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading fighter list: {e}")
        return pd.DataFrame()


def scrape_all_fighters(fighter_df: pd.DataFrame, config, resume: bool = False) -> List[Dict[str, Any]]:
    """
    Scrape fight histories for all fighters in the list
    
    Args:
        fighter_df: DataFrame with fighter information
        config: ScraperConfig object
        resume: If True, skip fighters already in cache
    
    Returns:
        List of all fight records
    """
    scraper = FightHistoryScraper(config)
    all_fights = []
    
    stats = {
        'total_fighters': len(fighter_df),
        'fighters_scraped': 0,
        'total_fights': 0,
        'errors': 0,
        'skipped_cache': 0,
        'start_time': datetime.now()
    }
    
    for idx, row in fighter_df.iterrows():
        fighter_name = row.get('name', row.get('fighter_name', ''))
        
        if not fighter_name:
            logger.warning(f"No fighter name in row {idx}")
            continue
        
        logger.info(f"Processing {idx + 1}/{len(fighter_df)}: {fighter_name}")
        
        try:
            # Check cache if resuming
            if resume and config.use_cache:
                cache_key = scraper._get_cache_key(fighter_name)
                cached_fights = scraper._load_from_cache(cache_key)
                if cached_fights:
                    logger.info(f"  Using cached data for {fighter_name}")
                    all_fights.extend(cached_fights)
                    stats['skipped_cache'] += 1
                    stats['total_fights'] += len(cached_fights)
                    continue
            
            # Try to get fighter URL if available
            fighter_url = row.get('fighter_url', row.get('url', ''))
            
            if fighter_url:
                # Scrape from URL
                fights = scraper.scrape_fight_history(fighter_url, fighter_name)
            else:
                # Try to search for fighter
                logger.warning(f"  No URL for {fighter_name}, skipping")
                stats['errors'] += 1
                continue
            
            # Validate and clean
            if config.validate_on_scrape:
                fights = validate_fight_data(fights, config)
            
            # Save to cache
            if config.use_cache:
                cache_key = scraper._get_cache_key(fighter_name)
                scraper._save_to_cache(cache_key, fights)
            
            all_fights.extend(fights)
            stats['fighters_scraped'] += 1
            stats['total_fights'] += len(fights)
            
            logger.info(f"  Scraped {len(fights)} fights for {fighter_name}")
            
            # Rate limiting between fighters
            import time
            time.sleep(config.delay_between_fighters)
            
        except Exception as e:
            logger.error(f"  Error processing {fighter_name}: {e}")
            stats['errors'] += 1
            continue
    
    stats['end_time'] = datetime.now()
    stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
    
    # Log summary statistics
    logger.info("\n" + "="*60)
    logger.info("SCRAPING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total fighters in list: {stats['total_fighters']}")
    logger.info(f"Fighters successfully scraped: {stats['fighters_scraped']}")
    logger.info(f"Fighters loaded from cache: {stats['skipped_cache']}")
    logger.info(f"Total fights collected: {stats['total_fights']}")
    logger.info(f"Errors encountered: {stats['errors']}")
    logger.info(f"Time elapsed: {stats['duration']:.1f} seconds")
    logger.info("="*60)
    
    return all_fights


def save_fight_history(fights: List[Dict[str, Any]], output_path: str):
    """
    Save fight history to CSV file
    
    Args:
        fights: List of fight dictionaries
        output_path: Path to output CSV file
    """
    if not fights:
        logger.warning("No fights to save")
        return
    
    try:
        df = pd.DataFrame(fights)
        
        # Reorder columns for readability
        column_order = [
            'fighter_name', 'opponent_name', 'event_date', 'event_name',
            'result', 'win_method', 'round_number', 'fight_duration_seconds',
            'sig_strikes_landed', 'sig_strikes_received', 'fight_ending_technique',
            'data_quality_score', 'scraped_at'
        ]
        
        # Keep only columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        
        df = df[final_columns]
        
        # Sort by fighter name and date
        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
            df = df.sort_values(['fighter_name', 'event_date'], ascending=[True, False])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} fight records to {output_path}")
        
        # Log statistics
        if 'fighter_name' in df.columns:
            unique_fighters = df['fighter_name'].nunique()
            logger.info(f"Data includes {unique_fighters} unique fighters")
        
        if 'data_quality_score' in df.columns:
            avg_quality = df['data_quality_score'].mean()
            logger.info(f"Average data quality score: {avg_quality:.3f}")
        
    except Exception as e:
        logger.error(f"Error saving fight history: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Collect fight history data for MMA fighters'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--fighters',
        type=str,
        default=None,
        help='Path to CSV file with fighter list'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output CSV file'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume scraping using cached data'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine fighter list path
    if args.fighters:
        fighter_file = args.fighters
    else:
        # Try common locations
        possible_paths = [
            os.path.join(config.data_dir, 'fighter_stats.csv'),
            os.path.join(config.data_dir, 'ufc_rankings.csv'),
            os.path.join(config.data_dir, 'fighters.csv'),
        ]
        
        fighter_file = None
        for path in possible_paths:
            if os.path.exists(path):
                fighter_file = path
                break
        
        if not fighter_file:
            logger.error("No fighter list found. Please specify with --fighters")
            logger.info(f"Tried: {', '.join(possible_paths)}")
            return 1
    
    # Load fighter list
    fighter_df = load_fighter_list(fighter_file)
    
    if fighter_df.empty:
        logger.error("Fighter list is empty or could not be loaded")
        return 1
    
    # Scrape all fighters
    logger.info("Starting fight history collection...")
    all_fights = scrape_all_fighters(fighter_df, config, resume=args.resume)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(config.data_dir, config.fight_history_file)
    
    # Save results
    save_fight_history(all_fights, output_path)
    
    logger.info("Fight history collection complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
