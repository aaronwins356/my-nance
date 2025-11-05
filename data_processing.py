"""
Data Processing Module
Feature engineering and data preparation for ML models
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_dataset(filepath=None):
    """
    Load the training dataset
    
    Args:
        filepath: Optional path to dataset file
    
    Returns:
        pandas DataFrame
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'fighters_top10_men.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns")
    
    return df

def engineer_features(df):
    """
    Engineer additional features from raw data
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Experience difference
    df['experience_diff'] = (df['f1_wins'] + df['f1_losses']) - (df['f2_wins'] + df['f2_losses'])
    
    # Finish rate (KO + Sub percentage)
    df['f1_finish_rate'] = df['f1_ko_pct'] + df['f1_sub_pct']
    df['f2_finish_rate'] = df['f2_ko_pct'] + df['f2_sub_pct']
    df['finish_rate_diff'] = df['f1_finish_rate'] - df['f2_finish_rate']
    
    # Striking advantage
    df['striking_advantage'] = (df['f1_striking_accuracy'] - df['f1_sig_strikes_absorbed']) - \
                                (df['f2_striking_accuracy'] - df['f2_sig_strikes_absorbed'])
    
    # Takedown advantage
    df['takedown_advantage'] = (df['f1_takedown_avg'] + df['f1_takedown_defense']) - \
                               (df['f2_takedown_avg'] + df['f2_takedown_defense'])
    
    # Submission threat
    df['submission_threat_diff'] = df['f1_submission_avg'] - df['f2_submission_avg']
    
    # Overall striking differential
    df['striking_output_diff'] = df['f1_sig_strikes_per_min'] - df['f2_sig_strikes_per_min']
    df['striking_defense_quality'] = (df['f1_striking_defense'] - df['f2_sig_strikes_per_min']) - \
                                      (df['f2_striking_defense'] - df['f1_sig_strikes_per_min'])
    
    # Physical advantages
    df['physical_advantage'] = df['height_diff'] + df['reach_diff']
    
    # ELO advantage (if available)
    if 'f1_elo' in df.columns and 'f2_elo' in df.columns:
        if 'elo_diff' not in df.columns:
            df['elo_diff'] = df['f1_elo'] - df['f2_elo']
    
    print(f"Engineered features, now have {len(df.columns)} columns")
    
    return df

def select_features(df, feature_list=None):
    """
    Select features for model training
    
    Args:
        df: Input DataFrame
        feature_list: Optional list of feature names to use
    
    Returns:
        X (features), y (target), feature_names
    """
    # Exclude metadata and target columns
    exclude_cols = ['event_date', 'fighter1_name', 'fighter2_name', 'fighter1_won']
    
    if feature_list is None:
        # Auto-select numerical features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64', 'bool']]
    else:
        feature_cols = feature_list
    
    X = df[feature_cols]
    y = df['fighter1_won']
    
    print(f"Selected {len(feature_cols)} features for training")
    print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y, feature_cols

def prepare_data(test_size=0.2, random_state=42):
    """
    Complete data preparation pipeline
    
    Args:
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    ensure_directories()
    
    # Load dataset
    df = load_dataset()
    
    # Engineer features
    df = engineer_features(df)
    
    # Select features
    X, y, feature_names = select_features(df)
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("Filling missing values...")
        X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Training target distribution: {y_train.value_counts(normalize=True).to_dict()}")
    
    # Save feature names for later use
    feature_names_file = os.path.join(ARTIFACTS_DIR, 'feature_names.json')
    with open(feature_names_file, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"Saved feature names to {feature_names_file}")
    
    return X_train, X_test, y_train, y_test, feature_names

def load_feature_names():
    """Load saved feature names"""
    feature_names_file = os.path.join(ARTIFACTS_DIR, 'feature_names.json')
    
    if not os.path.exists(feature_names_file):
        raise FileNotFoundError(f"Feature names not found at {feature_names_file}")
    
    with open(feature_names_file, 'r') as f:
        return json.load(f)

def prepare_single_fight(fighter1_stats, fighter2_stats, elo_ratings=None):
    """
    Prepare features for a single fight prediction
    
    Args:
        fighter1_stats: Dict with fighter 1 statistics
        fighter2_stats: Dict with fighter 2 statistics
        elo_ratings: Optional dict with ELO ratings
    
    Returns:
        pandas DataFrame with single row of features
    """
    # Create base features
    features = {
        'f1_wins': fighter1_stats.get('wins', 0),
        'f1_losses': fighter1_stats.get('losses', 0),
        'f1_win_rate': fighter1_stats.get('wins', 0) / max(fighter1_stats.get('wins', 0) + fighter1_stats.get('losses', 0), 1),
        'f1_ko_pct': fighter1_stats.get('wins_by_ko', 0) / max(fighter1_stats.get('wins', 1), 1),
        'f1_sub_pct': fighter1_stats.get('wins_by_submission', 0) / max(fighter1_stats.get('wins', 1), 1),
        'f1_height': fighter1_stats.get('height', 72),
        'f1_reach': fighter1_stats.get('reach', 72),
        'f1_age': fighter1_stats.get('age', 30),
        'f1_sig_strikes_per_min': fighter1_stats.get('sig_strikes_per_min', 4.0),
        'f1_sig_strikes_absorbed': fighter1_stats.get('sig_strikes_absorbed_per_min', 3.5),
        'f1_takedown_avg': fighter1_stats.get('takedown_avg_per_15min', 1.5),
        'f1_takedown_defense': fighter1_stats.get('takedown_defense_pct', 70),
        'f1_submission_avg': fighter1_stats.get('submission_avg_per_15min', 0.5),
        'f1_striking_accuracy': fighter1_stats.get('striking_accuracy_pct', 45),
        'f1_striking_defense': fighter1_stats.get('striking_defense_pct', 55),
        
        'f2_wins': fighter2_stats.get('wins', 0),
        'f2_losses': fighter2_stats.get('losses', 0),
        'f2_win_rate': fighter2_stats.get('wins', 0) / max(fighter2_stats.get('wins', 0) + fighter2_stats.get('losses', 0), 1),
        'f2_ko_pct': fighter2_stats.get('wins_by_ko', 0) / max(fighter2_stats.get('wins', 1), 1),
        'f2_sub_pct': fighter2_stats.get('wins_by_submission', 0) / max(fighter2_stats.get('wins', 1), 1),
        'f2_height': fighter2_stats.get('height', 72),
        'f2_reach': fighter2_stats.get('reach', 72),
        'f2_age': fighter2_stats.get('age', 30),
        'f2_sig_strikes_per_min': fighter2_stats.get('sig_strikes_per_min', 4.0),
        'f2_sig_strikes_absorbed': fighter2_stats.get('sig_strikes_absorbed_per_min', 3.5),
        'f2_takedown_avg': fighter2_stats.get('takedown_avg_per_15min', 1.5),
        'f2_takedown_defense': fighter2_stats.get('takedown_defense_pct', 70),
        'f2_submission_avg': fighter2_stats.get('submission_avg_per_15min', 0.5),
        'f2_striking_accuracy': fighter2_stats.get('striking_accuracy_pct', 45),
        'f2_striking_defense': fighter2_stats.get('striking_defense_pct', 55),
        
        'is_title_fight': False
    }
    
    # Add derived features
    features['height_diff'] = features['f1_height'] - features['f2_height']
    features['reach_diff'] = features['f1_reach'] - features['f2_reach']
    features['age_diff'] = features['f1_age'] - features['f2_age']
    features['win_rate_diff'] = features['f1_win_rate'] - features['f2_win_rate']
    features['striking_diff'] = features['f1_sig_strikes_per_min'] - features['f2_sig_strikes_per_min']
    features['defense_diff'] = features['f1_striking_defense'] - features['f2_striking_defense']
    
    # Add ELO if available
    if elo_ratings:
        fighter1_name = fighter1_stats.get('name')
        fighter2_name = fighter2_stats.get('name')
        features['f1_elo'] = elo_ratings.get(fighter1_name, 1500)
        features['f2_elo'] = elo_ratings.get(fighter2_name, 1500)
        features['elo_diff'] = features['f1_elo'] - features['f2_elo']
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Engineer same features as training
    df = engineer_features(df)
    
    return df

if __name__ == '__main__':
    # Test data preparation
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    print("\nData preparation complete!")
    print(f"Feature names: {feature_names[:10]}... (showing first 10)")
