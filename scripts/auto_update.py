"""
Auto-Update System
Automatically checks for new events and updates the system
"""
import os
import sys
import time
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline components
from scripts import scrape_rankings, scrape_fighter_stats, scrape_events, build_dataset
import elo_pipeline
import train

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def check_for_new_events():
    """
    Check if there are new events since last update
    
    Returns:
        bool: True if new events detected
    """
    events_file = os.path.join(DATA_DIR, 'fight_results.csv')
    
    if not os.path.exists(events_file):
        print("No existing events file found - full update needed")
        return True
    
    # Check file modification time
    file_mod_time = os.path.getmtime(events_file)
    current_time = time.time()
    
    # Check if file is older than 7 days
    days_since_update = (current_time - file_mod_time) / (24 * 3600)
    
    print(f"Days since last update: {days_since_update:.1f}")
    
    if days_since_update >= 7:
        print("Update needed (>7 days since last update)")
        return True
    
    print("No update needed yet")
    return False

def run_full_pipeline():
    """
    Run the complete data pipeline
    """
    print("\n" + "="*60)
    print("Running Full FightIQ Update Pipeline")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Scrape Rankings
        print("\n[1/6] Scraping UFC Rankings...")
        scrape_rankings.ensure_directories()
        scrape_rankings.save_rankings_to_csv()
        
        # Step 2: Scrape Fighter Stats
        print("\n[2/6] Scraping Fighter Statistics...")
        scrape_fighter_stats.ensure_directories()
        scrape_fighter_stats.scrape_all_ranked_fighters()
        
        # Step 3: Scrape Events
        print("\n[3/6] Scraping Event Results...")
        scrape_events.ensure_directories()
        scrape_events.save_events_to_csv()
        
        # Step 4: Build Dataset
        print("\n[4/6] Building Training Dataset...")
        build_dataset.build_pipeline()
        
        # Step 5: Update ELO Ratings
        print("\n[5/6] Updating ELO Ratings...")
        elo_pipeline.main()
        
        # Step 6: Retrain Model
        print("\n[6/6] Retraining Model...")
        train.train_and_compare_models()
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
    
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_update():
    """
    Run a quick update (just events and ELO, no model retraining)
    """
    print("\n" + "="*60)
    print("Running Quick FightIQ Update")
    print("="*60)
    
    try:
        # Step 1: Scrape Events
        print("\n[1/3] Scraping Event Results...")
        scrape_events.save_events_to_csv()
        
        # Step 2: Rebuild Dataset
        print("\n[2/3] Rebuilding Dataset...")
        build_dataset.build_pipeline()
        
        # Step 3: Update ELO
        print("\n[3/3] Updating ELO Ratings...")
        elo_pipeline.main()
        
        print("\n" + "="*60)
        print("Quick Update Complete!")
        print("="*60)
        
        return True
    
    except Exception as e:
        print(f"\nError during quick update: {e}")
        return False

def schedule_weekly_update():
    """
    Run weekly update check (meant to be called as a cron job or scheduled task)
    """
    print("FightIQ Auto-Update System")
    print(f"Checking for updates at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if check_for_new_events():
        print("\nStarting update process...")
        success = run_full_pipeline()
        
        if success:
            print("\nUpdate completed successfully!")
        else:
            print("\nUpdate failed - please check logs")
    else:
        print("\nNo update needed at this time")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FightIQ Auto-Update System')
    parser.add_argument('--full', action='store_true', 
                       help='Run full pipeline (including model retraining)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick update (events and ELO only)')
    parser.add_argument('--check', action='store_true',
                       help='Check if update is needed')
    parser.add_argument('--schedule', action='store_true',
                       help='Run scheduled update check')
    
    args = parser.parse_args()
    
    if args.full:
        run_full_pipeline()
    elif args.quick:
        run_quick_update()
    elif args.check:
        if check_for_new_events():
            print("Update recommended")
            sys.exit(1)
        else:
            print("No update needed")
            sys.exit(0)
    elif args.schedule:
        schedule_weekly_update()
    else:
        # Default: run full pipeline
        print("No action specified, running full pipeline...")
        run_full_pipeline()

if __name__ == '__main__':
    main()
