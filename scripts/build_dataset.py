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

from utils import american_odds_to_probability

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def ensure_directories():
    """Ensure data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)

def load_fighter_styles():
    """Load fighter style classifications with detailed styles"""
    styles_file = os.path.join(DATA_DIR, 'fighter_styles.csv')
    
    if os.path.exists(styles_file):
        styles_df = pd.read_csv(styles_file)
        print(f"Loaded {len(styles_df)} fighter styles")
        
        # Use detailed_style if available, otherwise fall back to primary_style
        if 'detailed_style' in styles_df.columns:
            return dict(zip(styles_df['fighter_name'], styles_df['detailed_style']))
        else:
            return dict(zip(styles_df['fighter_name'], styles_df['primary_style']))
    else:
        print("Warning: Fighter styles not found, will use default 'Unknown' style")
        return {}

def load_fight_odds():
    """Load historical betting odds"""
    odds_file = os.path.join(DATA_DIR, 'fight_odds.csv')
    
    if os.path.exists(odds_file):
        odds_df = pd.read_csv(odds_file)
        print(f"Loaded odds for {len(odds_df)} fights")
        return odds_df
    else:
        print("Warning: Fight odds not found, odds features will be set to default")
        return pd.DataFrame()

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
    
    # Load additional data
    fighter_styles = load_fighter_styles()
    fight_odds = load_fight_odds()
    
    return fighter_stats, fight_results, fighter_styles, fight_odds

def build_training_dataset(fighter_stats, fight_results, fighter_styles, fight_odds):
    """
    Build training dataset with features from both fighters including styles and odds
    """
    print("Building training dataset...")
    
    training_data = []
    
    # Available styles for one-hot encoding
    all_styles = ['Striker', 'Wrestler', 'BJJ', 'Well-Rounded', 'Boxer', 'Unknown']
    
    for idx, fight in fight_results.iterrows():
        fighter1_name = fight['fighter1']
        fighter2_name = fight['fighter2']
        winner = fight['winner']
        event_date = fight['event_date']
        
        # Get stats for both fighters
        f1_stats = fighter_stats[fighter_stats['name'] == fighter1_name]
        f2_stats = fighter_stats[fighter_stats['name'] == fighter2_name]
        
        if f1_stats.empty or f2_stats.empty:
            continue
        
        f1 = f1_stats.iloc[0]
        f2 = f2_stats.iloc[0]
        
        # Get fighter styles
        f1_style = fighter_styles.get(fighter1_name, 'Unknown')
        f2_style = fighter_styles.get(fighter2_name, 'Unknown')
        
        # Get odds for this fight
        odds_row = fight_odds[
            (fight_odds['fighter1'] == fighter1_name) & 
            (fight_odds['fighter2'] == fighter2_name) &
            (fight_odds['event_date'] == event_date)
        ]
        
        if not odds_row.empty and odds_row.iloc[0]['odds_type'] == 'american':
            f1_odds = odds_row.iloc[0]['fighter1_odds']
            f2_odds = odds_row.iloc[0]['fighter2_odds']
            f1_implied_prob = american_odds_to_probability(f1_odds)
            f2_implied_prob = american_odds_to_probability(f2_odds)
            # Normalize to sum to 1
            total_prob = f1_implied_prob + f2_implied_prob
            f1_implied_prob = f1_implied_prob / total_prob
            f2_implied_prob = f2_implied_prob / total_prob
        else:
            # Default to 50-50 if no odds available
            f1_implied_prob = 0.5
            f2_implied_prob = 0.5
        
        # Create feature set
        features = {
            # Fight metadata
            'event_date': event_date,
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
            
            # Betting odds features
            'f1_odds_implied_prob': f1_implied_prob,
            'f2_odds_implied_prob': f2_implied_prob,
            'odds_diff': f1_implied_prob - f2_implied_prob,
            
            # Target variable
            'fighter1_won': 1 if winner == fighter1_name else 0
        }
        
        # Add style one-hot encoding for fighter 1
        for style in all_styles:
            features[f'f1_style_{style.lower()}'] = 1 if f1_style == style else 0
        
        # Add style one-hot encoding for fighter 2
        for style in all_styles:
            features[f'f2_style_{style.lower()}'] = 1 if f2_style == style else 0
        
        # Add style matchup features
        features['style_matchup'] = f'{f1_style}_vs_{f2_style}'
        # One-hot encode common matchups
        common_matchups = [
            'Striker_vs_Wrestler', 'Wrestler_vs_Striker',
            'Striker_vs_BJJ', 'BJJ_vs_Striker',
            'Wrestler_vs_BJJ', 'BJJ_vs_Wrestler',
            'Striker_vs_Striker', 'Wrestler_vs_Wrestler', 'BJJ_vs_BJJ'
        ]
        for matchup in common_matchups:
            features[f'matchup_{matchup.lower()}'] = 1 if features['style_matchup'] == matchup else 0
        
        training_data.append(features)
    
    df = pd.DataFrame(training_data)
    
    print(f"Built training dataset with {len(df)} samples")
    print(f"Fighter 1 win rate: {df['fighter1_won'].mean():.2%}")
    print(f"Columns with style features: {len([c for c in df.columns if 'style' in c])}")
    print(f"Columns with odds features: {len([c for c in df.columns if 'odds' in c])}")
    
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
    fighter_stats, fight_results, fighter_styles, fight_odds = load_data()
    
    # Build training dataset
    training_df = build_training_dataset(fighter_stats, fight_results, fighter_styles, fight_odds)
    
    # Save dataset
    output_file = save_dataset(training_df)
    
    print("\nDataset build complete!")
    return training_df

if __name__ == '__main__':
    build_pipeline()
