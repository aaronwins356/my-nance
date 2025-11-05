"""
Inference Module
Makes predictions for fights using trained model
"""
import os
import pickle
import json
import pandas as pd
import numpy as np
from data_processing import prepare_single_fight, load_feature_names

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def load_model():
    """
    Load trained model from artifacts
    
    Returns:
        Loaded model
    """
    model_file = os.path.join(ARTIFACTS_DIR, 'model.pkl')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model not found at {model_file}. "
            "Please run train.py first to train a model."
        )
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    return model

def load_model_metadata():
    """
    Load model metadata
    
    Returns:
        Dictionary with model metadata
    """
    metadata_file = os.path.join(ARTIFACTS_DIR, 'model_metadata.json')
    
    if not os.path.exists(metadata_file):
        return None
    
    with open(metadata_file, 'r') as f:
        return json.load(f)

def load_fighter_stats():
    """
    Load fighter stats from data directory
    
    Returns:
        pandas DataFrame with fighter stats
    """
    fighter_stats_file = os.path.join(DATA_DIR, 'fighter_stats.csv')
    
    if not os.path.exists(fighter_stats_file):
        raise FileNotFoundError(
            f"Fighter stats not found at {fighter_stats_file}. "
            "Please run scrape_fighter_stats.py first."
        )
    
    return pd.read_csv(fighter_stats_file)

def load_elo_ratings():
    """
    Load ELO ratings
    
    Returns:
        Dictionary mapping fighter name to ELO rating
    """
    elo_file = os.path.join(DATA_DIR, 'elo_ratings.csv')
    
    if not os.path.exists(elo_file):
        print("Warning: ELO ratings not found, using default values")
        return {}
    
    elo_df = pd.read_csv(elo_file)
    return dict(zip(elo_df['fighter'], elo_df['elo_rating']))

def get_fighter_stats_dict(fighter_name, fighter_stats_df):
    """
    Get stats for a single fighter
    
    Args:
        fighter_name: Name of the fighter
        fighter_stats_df: DataFrame with all fighter stats
    
    Returns:
        Dictionary with fighter stats
    """
    fighter_row = fighter_stats_df[fighter_stats_df['name'] == fighter_name]
    
    if fighter_row.empty:
        raise ValueError(f"Fighter '{fighter_name}' not found in database")
    
    return fighter_row.iloc[0].to_dict()

def predict_fight(fighter1_name, fighter2_name, model=None, is_title_fight=False):
    """
    Predict outcome of a fight
    
    Args:
        fighter1_name: Name of first fighter
        fighter2_name: Name of second fighter
        model: Optional pre-loaded model
        is_title_fight: Whether this is a title fight
    
    Returns:
        Dictionary with prediction results
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Load data
    fighter_stats_df = load_fighter_stats()
    elo_ratings = load_elo_ratings()
    feature_names = load_feature_names()
    
    # Get fighter stats
    fighter1_stats = get_fighter_stats_dict(fighter1_name, fighter_stats_df)
    fighter2_stats = get_fighter_stats_dict(fighter2_name, fighter_stats_df)
    
    # Prepare features
    fight_features = prepare_single_fight(fighter1_stats, fighter2_stats, elo_ratings)
    fight_features['is_title_fight'] = is_title_fight
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in fight_features.columns:
            fight_features[feature] = 0
    
    # Select only the features used by the model
    X = fight_features[feature_names]
    
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Interpret results
    fighter1_win_prob = probabilities[1]
    fighter2_win_prob = probabilities[0]
    
    predicted_winner = fighter1_name if prediction == 1 else fighter2_name
    confidence = max(fighter1_win_prob, fighter2_win_prob)
    
    result = {
        'fighter1': fighter1_name,
        'fighter2': fighter2_name,
        'fighter1_win_probability': float(fighter1_win_prob),
        'fighter2_win_probability': float(fighter2_win_prob),
        'predicted_winner': predicted_winner,
        'confidence': float(confidence),
        'is_title_fight': is_title_fight,
        'fighter1_elo': elo_ratings.get(fighter1_name, 1500),
        'fighter2_elo': elo_ratings.get(fighter2_name, 1500)
    }
    
    return result

def predict_multiple_fights(fights_list, model=None):
    """
    Predict outcomes for multiple fights
    
    Args:
        fights_list: List of tuples (fighter1_name, fighter2_name, is_title_fight)
        model: Optional pre-loaded model
    
    Returns:
        List of prediction dictionaries
    """
    if model is None:
        model = load_model()
    
    results = []
    
    for fight in fights_list:
        if len(fight) == 2:
            fighter1, fighter2 = fight
            is_title = False
        else:
            fighter1, fighter2, is_title = fight
        
        try:
            result = predict_fight(fighter1, fighter2, model, is_title)
            results.append(result)
        except Exception as e:
            print(f"Error predicting {fighter1} vs {fighter2}: {e}")
            results.append({
                'fighter1': fighter1,
                'fighter2': fighter2,
                'error': str(e)
            })
    
    return results

def explain_prediction_simple(prediction_result, top_n=5):
    """
    Provide simple explanation of prediction
    
    Args:
        prediction_result: Result from predict_fight
        top_n: Number of top factors to show
    
    Returns:
        Dictionary with explanation
    """
    # Load metadata to get feature importance
    metadata = load_model_metadata()
    
    if metadata is None or 'feature_importance' not in metadata:
        return {
            'explanation': 'Feature importance not available'
        }
    
    # Get top features
    feature_importance = metadata['feature_importance']
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    explanation = {
        'predicted_winner': prediction_result['predicted_winner'],
        'confidence': prediction_result['confidence'],
        'top_factors': [
            {'feature': feature, 'importance': importance}
            for feature, importance in top_features
        ],
        'key_insights': []
    }
    
    # Add key insights based on probabilities
    prob_diff = abs(prediction_result['fighter1_win_probability'] - 
                   prediction_result['fighter2_win_probability'])
    
    if prob_diff > 0.3:
        explanation['key_insights'].append("Strong favorite - significant skill/ELO advantage")
    elif prob_diff > 0.15:
        explanation['key_insights'].append("Moderate favorite - clear advantages in key areas")
    else:
        explanation['key_insights'].append("Close matchup - fight could go either way")
    
    # ELO comparison
    elo_diff = abs(prediction_result['fighter1_elo'] - prediction_result['fighter2_elo'])
    if elo_diff > 100:
        explanation['key_insights'].append(f"Large ELO difference ({elo_diff:.0f} points)")
    
    return explanation

def print_prediction(prediction_result):
    """
    Pretty print prediction result
    
    Args:
        prediction_result: Result from predict_fight
    """
    print("\n" + "="*60)
    print("FIGHT PREDICTION")
    print("="*60)
    print(f"\n{prediction_result['fighter1']} vs {prediction_result['fighter2']}")
    
    if prediction_result.get('is_title_fight'):
        print("(Title Fight)")
    
    print(f"\nELO Ratings:")
    print(f"  {prediction_result['fighter1']}: {prediction_result['fighter1_elo']:.0f}")
    print(f"  {prediction_result['fighter2']}: {prediction_result['fighter2_elo']:.0f}")
    
    print(f"\nWin Probabilities:")
    print(f"  {prediction_result['fighter1']}: {prediction_result['fighter1_win_probability']:.1%}")
    print(f"  {prediction_result['fighter2']}: {prediction_result['fighter2_win_probability']:.1%}")
    
    print(f"\nPredicted Winner: {prediction_result['predicted_winner']}")
    print(f"Confidence: {prediction_result['confidence']:.1%}")
    
    # Add explanation
    explanation = explain_prediction_simple(prediction_result)
    
    if 'key_insights' in explanation:
        print(f"\nKey Insights:")
        for insight in explanation['key_insights']:
            print(f"  • {insight}")
    
    print("="*60)

if __name__ == '__main__':
    # Example prediction
    import sys
    
    if len(sys.argv) >= 3:
        fighter1 = sys.argv[1]
        fighter2 = sys.argv[2]
        is_title = len(sys.argv) > 3 and sys.argv[3].lower() in ['true', 'yes', '1']
        
        result = predict_fight(fighter1, fighter2, is_title_fight=is_title)
        print_prediction(result)
    else:
        # Default example
        print("Testing prediction system with example fighters...")
        
        result = predict_fight("Islam Makhachev", "Charles Oliveira", is_title_fight=True)
        print_prediction(result)
        
        print("\n" + "-"*60 + "\n")
        
        result2 = predict_fight("Max Holloway", "Alexander Volkanovski")
        print_prediction(result2)
