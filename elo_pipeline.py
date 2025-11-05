"""
ELO Pipeline
Orchestrates ELO rating updates from fight results
"""
import os
import pandas as pd
from datetime import datetime
from elo_system import ELOSystem

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def ensure_directories():
    """Ensure data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)

def load_fight_results():
    """Load fight results from data directory"""
    fight_results_file = os.path.join(DATA_DIR, 'fight_results.csv')
    
    if not os.path.exists(fight_results_file):
        raise FileNotFoundError(
            f"Fight results not found at {fight_results_file}. "
            "Run scrape_events.py first."
        )
    
    df = pd.read_csv(fight_results_file)
    print(f"Loaded {len(df)} fight results")
    
    return df

def update_elo_ratings():
    """
    Update ELO ratings based on fight results
    """
    print("Starting ELO rating update...")
    
    # Load fight results
    fight_results = load_fight_results()
    
    # Sort by date (oldest first)
    fight_results['event_date'] = pd.to_datetime(fight_results['event_date'])
    fight_results = fight_results.sort_values('event_date')
    
    # Initialize ELO system
    elo = ELOSystem(k_factor=32, initial_rating=1500)
    
    # Process each fight
    print(f"Processing {len(fight_results)} fights...")
    
    for idx, fight in fight_results.iterrows():
        winner = fight['winner']
        loser = fight['loser']
        method = fight['method']
        is_title = fight.get('is_title_fight', False)
        fight_date = fight['event_date'].isoformat()
        round_finished = fight.get('round', 3)
        
        # Update ratings
        result = elo.update_rating(
            winner=winner,
            loser=loser,
            method=method,
            is_title_fight=is_title,
            fight_date=fight_date,
            round_finished=round_finished
        )
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(fight_results)} fights...")
    
    # Apply inactivity decay
    print("\nApplying inactivity decay...")
    decayed = elo.apply_inactivity_decay()
    
    if decayed:
        print(f"Applied decay to {len(decayed)} inactive fighters")
        for item in decayed[:5]:  # Show first 5
            print(f"  {item['fighter']}: {item['old_rating']:.0f} -> {item['new_rating']:.0f} "
                  f"({item['days_inactive']} days inactive)")
    
    # Get all ratings
    all_ratings = elo.get_all_ratings(sort_by_rating=True)
    
    # Convert to DataFrame
    ratings_df = pd.DataFrame([
        {
            'fighter': fighter,
            'elo_rating': rating,
            'fights': fights,
            'last_fight_date': last_date
        }
        for fighter, rating, fights, last_date in all_ratings
    ])
    
    # Save ratings
    output_file = os.path.join(DATA_DIR, 'elo_ratings.csv')
    ratings_df.to_csv(output_file, index=False)
    
    print(f"\nSaved ELO ratings to {output_file}")
    print(f"Total fighters rated: {len(ratings_df)}")
    
    # Display top 10
    print("\nTop 10 Fighters by ELO:")
    print("-" * 60)
    for idx, row in ratings_df.head(10).iterrows():
        print(f"{idx+1:2d}. {row['fighter']:30s} {row['elo_rating']:7.0f} ({row['fights']} fights)")
    
    return ratings_df, elo

def add_elo_to_dataset():
    """
    Add ELO ratings to training dataset
    """
    print("\nAdding ELO ratings to training dataset...")
    
    # Load dataset
    dataset_file = os.path.join(DATA_DIR, 'fighters_top10_men.csv')
    elo_file = os.path.join(DATA_DIR, 'elo_ratings.csv')
    
    if not os.path.exists(dataset_file):
        print("Training dataset not found. Skipping ELO integration.")
        return
    
    if not os.path.exists(elo_file):
        print("ELO ratings not found. Run update_elo_ratings first.")
        return
    
    dataset = pd.read_csv(dataset_file)
    elo_ratings = pd.read_csv(elo_file)
    
    # Create ELO lookup
    elo_lookup = dict(zip(elo_ratings['fighter'], elo_ratings['elo_rating']))
    
    # Add ELO ratings to dataset
    dataset['f1_elo'] = dataset['fighter1_name'].map(elo_lookup).fillna(1500)
    dataset['f2_elo'] = dataset['fighter2_name'].map(elo_lookup).fillna(1500)
    dataset['elo_diff'] = dataset['f1_elo'] - dataset['f2_elo']
    
    # Save updated dataset
    dataset.to_csv(dataset_file, index=False)
    
    print(f"Added ELO ratings to {len(dataset)} training samples")
    print(f"ELO range: {dataset['f1_elo'].min():.0f} - {dataset['f1_elo'].max():.0f}")
    
    return dataset

def main():
    """Main pipeline"""
    ensure_directories()
    
    # Update ELO ratings
    ratings_df, elo = update_elo_ratings()
    
    # Add ELO to training dataset
    add_elo_to_dataset()
    
    print("\nELO pipeline complete!")
    
    return ratings_df, elo

if __name__ == '__main__':
    main()
