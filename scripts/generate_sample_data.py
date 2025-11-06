"""
Generate sample data for testing the pipeline
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def generate_fighter_stats():
    """Generate sample fighter statistics"""
    print("Generating fighter stats...")
    
    # Load fighter names from styles file
    styles_file = os.path.join(DATA_DIR, 'fighter_styles.csv')
    styles_df = pd.read_csv(styles_file)
    fighter_names = styles_df['fighter_name'].tolist()
    
    fighters = []
    for name in fighter_names:
        # Generate realistic stats with some randomness
        wins = np.random.randint(8, 25)
        losses = np.random.randint(0, 8)
        
        fighter = {
            'name': name,
            'wins': wins,
            'losses': losses,
            'draws': np.random.choice([0, 0, 0, 1]),
            'wins_by_ko': int(wins * np.random.uniform(0.2, 0.6)),
            'wins_by_submission': int(wins * np.random.uniform(0.1, 0.4)),
            'wins_by_decision': 0,  # Will be calculated
            'height': np.random.randint(68, 78),
            'reach': np.random.randint(70, 82),
            'age': np.random.randint(24, 38),
            'weight': np.random.randint(135, 265),
            'sig_strikes_per_min': round(np.random.uniform(2.5, 6.5), 2),
            'sig_strikes_absorbed_per_min': round(np.random.uniform(2.0, 5.0), 2),
            'takedown_avg_per_15min': round(np.random.uniform(0.5, 4.5), 2),
            'takedown_defense_pct': round(np.random.uniform(50, 90), 1),
            'submission_avg_per_15min': round(np.random.uniform(0.1, 2.0), 2),
            'striking_accuracy_pct': round(np.random.uniform(35, 55), 1),
            'striking_defense_pct': round(np.random.uniform(45, 65), 1)
        }
        
        # Calculate wins by decision
        fighter['wins_by_decision'] = wins - fighter['wins_by_ko'] - fighter['wins_by_submission']
        
        fighters.append(fighter)
    
    df = pd.DataFrame(fighters)
    output_file = os.path.join(DATA_DIR, 'fighter_stats.csv')
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} fighter profiles")
    print(f"Saved to {output_file}")
    
    return df

def generate_fight_results(fighter_stats):
    """Generate sample fight results"""
    print("\nGenerating fight results...")
    
    fighters = fighter_stats['name'].tolist()
    fights = []
    
    # Generate 200 random fights
    for i in range(200):
        # Pick two random fighters
        fighter1, fighter2 = np.random.choice(fighters, size=2, replace=False)
        
        # Get fighter stats
        f1_stats = fighter_stats[fighter_stats['name'] == fighter1].iloc[0]
        f2_stats = fighter_stats[fighter_stats['name'] == fighter2].iloc[0]
        
        # Simple logic: fighter with better win rate more likely to win
        f1_win_rate = f1_stats['wins'] / max(f1_stats['wins'] + f1_stats['losses'], 1)
        f2_win_rate = f2_stats['wins'] / max(f2_stats['wins'] + f2_stats['losses'], 1)
        
        # Add some randomness
        f1_score = f1_win_rate + np.random.uniform(-0.2, 0.2)
        f2_score = f2_win_rate + np.random.uniform(-0.2, 0.2)
        
        winner = fighter1 if f1_score > f2_score else fighter2
        loser = fighter2 if winner == fighter1 else fighter1
        
        # Generate random date within last 3 years
        days_ago = np.random.randint(0, 1095)
        event_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # Determine method
        method = np.random.choice(
            ['KO/TKO', 'Submission', 'Decision'],
            p=[0.25, 0.15, 0.6]
        )
        
        fight = {
            'fighter1': fighter1,
            'fighter2': fighter2,
            'winner': winner,
            'loser': loser,
            'method': method,
            'event_date': event_date,
            'event_name': f'UFC Event {i+1}',
            'is_title_fight': np.random.choice([False, False, False, False, True])
        }
        
        fights.append(fight)
    
    df = pd.DataFrame(fights)
    output_file = os.path.join(DATA_DIR, 'fight_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} fight results")
    print(f"Saved to {output_file}")
    
    return df

def main():
    """Main function to generate all sample data"""
    print("="*60)
    print("Generating Sample Data for Testing")
    print("="*60)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Generate fighter stats
    fighter_stats = generate_fighter_stats()
    
    # Generate fight results
    fight_results = generate_fight_results(fighter_stats)
    
    print("\n" + "="*60)
    print("Sample data generation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python scripts/build_dataset.py")
    print("2. Run: python elo_pipeline.py")
    print("3. Run: python train.py")

if __name__ == '__main__':
    main()
