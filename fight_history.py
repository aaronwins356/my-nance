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


def generate_sample_fight_history(fighter_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate sample fight history data for fighters without existing history
    
    Args:
        fighter_stats_df: DataFrame with fighter statistics
    
    Returns:
        DataFrame with generated fight history
    """
    import random
    import numpy as np
    from datetime import datetime, timedelta
    
    history_records = []
    
    for _, fighter in fighter_stats_df.iterrows():
        fighter_name = fighter['name']
        total_fights = fighter['wins'] + fighter['losses']
        
        # Generate fight history for each fight
        for fight_num in range(int(total_fights)):
            # Determine if this was a win or loss
            is_win = fight_num < fighter['wins']
            
            # Generate date (going back in time)
            days_ago = (total_fights - fight_num) * 120 + random.randint(0, 60)
            event_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            # Generate opponent name (simplified)
            opponent = f"Opponent {fight_num + 1}"
            
            # Determine method
            if is_win:
                method_choice = random.random()
                if method_choice < 0.3:
                    method = 'KO/TKO'
                    technique = random.choice(['Head kick KO', 'Punches', 'Left hook KO', 'Body shots'])
                elif method_choice < 0.5:
                    method = 'Submission'
                    technique = random.choice(['Rear-naked choke', 'Arm-triangle choke', 'Guillotine choke', 'Armbar'])
                else:
                    method = 'Decision'
                    technique = random.choice(['Unanimous Decision', 'Split Decision', 'Majority Decision'])
            else:
                method = random.choice(['KO/TKO', 'Submission', 'Decision'])
                technique = method if method == 'Decision' else 'N/A'
            
            # Generate round and duration
            if method == 'Decision':
                round_num = random.choice([3, 5])
                duration = round_num * 300
            else:
                round_num = random.randint(1, 3)
                duration = (round_num - 1) * 300 + random.randint(30, 300)
            
            # Generate strikes
            sig_strikes = random.randint(20, 150)
            sig_strikes_received = random.randint(15, 120)
            
            history_records.append({
                'fighter_name': fighter_name,
                'opponent_name': opponent,
                'event_date': event_date,
                'event_name': f'UFC Event {fight_num + 1}',
                'result': 'Win' if is_win else 'Loss',
                'win_method': method,
                'round_number': round_num,
                'fight_duration_seconds': duration,
                'sig_strikes_landed': sig_strikes,
                'sig_strikes_received': sig_strikes_received,
                'fight_ending_technique': technique
            })
    
    return pd.DataFrame(history_records)


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
