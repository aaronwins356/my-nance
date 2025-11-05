"""
Dataset Builder
Merges and cleans scraped data into training dataset
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Use relative path imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def ensure_directories():
    """Ensure data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)

def load_data():
    """Load all scraped data"""
    print("Loading scraped data...")
    
    # Load rankings
    rankings_file = os.path.join(DATA_DIR, 'ufc_rankings.csv')
    fighter_stats_file = os.path.join(DATA_DIR, 'fighter_stats.csv')
    fight_results_file = os.path.join(DATA_DIR, 'fight_results.csv')
    
    if not os.path.exists(fighter_stats_file):
        raise FileNotFoundError(f"Fighter stats not found at {fighter_stats_file}. Run scrape_fighter_stats.py first.")
    
    if not os.path.exists(fight_results_file):
        raise FileNotFoundError(f"Fight results not found at {fight_results_file}. Run scrape_events.py first.")
    
    fighter_stats = pd.read_csv(fighter_stats_file)
    fight_results = pd.read_csv(fight_results_file)
    
    print(f"Loaded {len(fighter_stats)} fighter profiles")
    print(f"Loaded {len(fight_results)} fight results")
    
    return fighter_stats, fight_results

def build_training_dataset(fighter_stats, fight_results):
    """
    Build training dataset with features from both fighters
    """
    print("Building training dataset...")
    
    training_data = []
    
    for idx, fight in fight_results.iterrows():
        fighter1_name = fight['fighter1']
        fighter2_name = fight['fighter2']
        winner = fight['winner']
        
        # Get stats for both fighters
        f1_stats = fighter_stats[fighter_stats['name'] == fighter1_name]
        f2_stats = fighter_stats[fighter_stats['name'] == fighter2_name]
        
        if f1_stats.empty or f2_stats.empty:
            continue
        
        f1 = f1_stats.iloc[0]
        f2 = f2_stats.iloc[0]
        
        # Create feature set
        features = {
            # Fight metadata
            'event_date': fight['event_date'],
            'fighter1_name': fighter1_name,
            'fighter2_name': fighter2_name,
            'is_title_fight': fight['is_title_fight'],
            
            # Fighter 1 stats
            'f1_wins': f1['wins'],
            'f1_losses': f1['losses'],
            'f1_win_rate': f1['wins'] / (f1['wins'] + f1['losses']) if (f1['wins'] + f1['losses']) > 0 else 0,
            'f1_ko_pct': f1['wins_by_ko'] / f1['wins'] if f1['wins'] > 0 else 0,
            'f1_sub_pct': f1['wins_by_submission'] / f1['wins'] if f1['wins'] > 0 else 0,
            'f1_height': f1['height'],
            'f1_reach': f1['reach'],
            'f1_age': f1['age'],
            'f1_sig_strikes_per_min': f1['sig_strikes_per_min'],
            'f1_sig_strikes_absorbed': f1['sig_strikes_absorbed_per_min'],
            'f1_takedown_avg': f1['takedown_avg_per_15min'],
            'f1_takedown_defense': f1['takedown_defense_pct'],
            'f1_submission_avg': f1['submission_avg_per_15min'],
            'f1_striking_accuracy': f1['striking_accuracy_pct'],
            'f1_striking_defense': f1['striking_defense_pct'],
            
            # Fighter 2 stats
            'f2_wins': f2['wins'],
            'f2_losses': f2['losses'],
            'f2_win_rate': f2['wins'] / (f2['wins'] + f2['losses']) if (f2['wins'] + f2['losses']) > 0 else 0,
            'f2_ko_pct': f2['wins_by_ko'] / f2['wins'] if f2['wins'] > 0 else 0,
            'f2_sub_pct': f2['wins_by_submission'] / f2['wins'] if f2['wins'] > 0 else 0,
            'f2_height': f2['height'],
            'f2_reach': f2['reach'],
            'f2_age': f2['age'],
            'f2_sig_strikes_per_min': f2['sig_strikes_per_min'],
            'f2_sig_strikes_absorbed': f2['sig_strikes_absorbed_per_min'],
            'f2_takedown_avg': f2['takedown_avg_per_15min'],
            'f2_takedown_defense': f2['takedown_defense_pct'],
            'f2_submission_avg': f2['submission_avg_per_15min'],
            'f2_striking_accuracy': f2['striking_accuracy_pct'],
            'f2_striking_defense': f2['striking_defense_pct'],
            
            # Derived features (differences/advantages)
            'height_diff': f1['height'] - f2['height'],
            'reach_diff': f1['reach'] - f2['reach'],
            'age_diff': f1['age'] - f2['age'],
            'win_rate_diff': (f1['wins'] / (f1['wins'] + f1['losses']) if (f1['wins'] + f1['losses']) > 0 else 0) - 
                             (f2['wins'] / (f2['wins'] + f2['losses']) if (f2['wins'] + f2['losses']) > 0 else 0),
            'striking_diff': f1['sig_strikes_per_min'] - f2['sig_strikes_per_min'],
            'defense_diff': f1['striking_defense_pct'] - f2['striking_defense_pct'],
            
            # Target variable
            'fighter1_won': 1 if winner == fighter1_name else 0
        }
        
        training_data.append(features)
    
    df = pd.DataFrame(training_data)
    
    print(f"Built training dataset with {len(df)} samples")
    print(f"Fighter 1 win rate: {df['fighter1_won'].mean():.2%}")
    
    return df

def calculate_data_quality_score(df):
    """
    Calculate data quality score based on completeness and validity
    """
    # Check for missing values
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    
    # Check for outliers (basic check)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_pct = 0
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
        outlier_pct += outliers / len(df)
    
    outlier_pct = outlier_pct / len(numeric_cols) if len(numeric_cols) > 0 else 0
    
    # Calculate score (0-1 scale)
    quality_score = (1 - missing_pct) * (1 - outlier_pct * 0.1)
    
    return quality_score

def save_dataset(df):
    """Save processed dataset"""
    output_file = os.path.join(DATA_DIR, 'fighters_top10_men.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\nSaved training dataset to {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Calculate and display data quality score
    quality_score = calculate_data_quality_score(df)
    print(f"Data quality score: {quality_score:.2%}")
    
    return output_file

def build_pipeline():
    """
    Main pipeline to build dataset
    """
    ensure_directories()
    
    # Load data
    fighter_stats, fight_results = load_data()
    
    # Build training dataset
    training_df = build_training_dataset(fighter_stats, fight_results)
    
    # Save dataset
    output_file = save_dataset(training_df)
    
    print("\nDataset build complete!")
    return training_df

if __name__ == '__main__':
    build_pipeline()
