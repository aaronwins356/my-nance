"""
ELO Rating System for MMA
Implements dynamic ELO ratings with adjustments for finish method, title fights, and inactivity
"""
import math
from datetime import datetime, timedelta

class ELOSystem:
    """
    ELO rating system adapted for MMA
    """
    
    def __init__(self, k_factor=32, initial_rating=1500):
        """
        Initialize ELO system
        
        Args:
            k_factor: How much ratings change per match (default: 32)
            initial_rating: Starting rating for new fighters (default: 1500)
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}
        self.fight_history = {}
    
    def get_rating(self, fighter_name):
        """Get current rating for a fighter"""
        if fighter_name not in self.ratings:
            self.ratings[fighter_name] = {
                'rating': self.initial_rating,
                'last_fight_date': None,
                'fights': 0
            }
        return self.ratings[fighter_name]['rating']
    
    def expected_score(self, rating_a, rating_b):
        """
        Calculate expected score for fighter A against fighter B
        Returns: probability between 0 and 1
        """
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    
    def update_rating(self, winner, loser, method='Decision', is_title_fight=False, 
                     fight_date=None, round_finished=3):
        """
        Update ELO ratings after a fight
        
        Args:
            winner: Name of winning fighter
            loser: Name of losing fighter
            method: Finish method ('KO/TKO', 'Submission', 'Decision')
            is_title_fight: Whether this was a title fight
            fight_date: Date of the fight
            round_finished: Round the fight ended in
        """
        # Get current ratings
        winner_rating = self.get_rating(winner)
        loser_rating = self.get_rating(loser)
        
        # Calculate expected scores
        winner_expected = self.expected_score(winner_rating, loser_rating)
        loser_expected = self.expected_score(loser_rating, winner_rating)
        
        # Base K-factor
        k = self.k_factor
        
        # Adjust K-factor for finish method
        if method == 'KO/TKO':
            k *= 1.2  # More impressive win
        elif method == 'Submission':
            k *= 1.15
        elif method == 'Decision':
            k *= 0.9  # Less decisive
        
        # Adjust for early finish
        if method != 'Decision' and round_finished <= 1:
            k *= 1.1  # Very dominant finish
        
        # Adjust for title fights
        if is_title_fight:
            k *= 1.25  # Title fights matter more
        
        # Update ratings
        winner_new = winner_rating + k * (1 - winner_expected)
        loser_new = loser_rating + k * (0 - loser_expected)
        
        # Store updated ratings
        self.ratings[winner] = {
            'rating': winner_new,
            'last_fight_date': fight_date,
            'fights': self.ratings.get(winner, {'fights': 0})['fights'] + 1
        }
        
        self.ratings[loser] = {
            'rating': loser_new,
            'last_fight_date': fight_date,
            'fights': self.ratings.get(loser, {'fights': 0})['fights'] + 1
        }
        
        # Store fight in history
        if winner not in self.fight_history:
            self.fight_history[winner] = []
        if loser not in self.fight_history:
            self.fight_history[loser] = []
        
        fight_record = {
            'opponent': loser,
            'result': 'W',
            'method': method,
            'date': fight_date,
            'rating_change': winner_new - winner_rating
        }
        self.fight_history[winner].append(fight_record)
        
        fight_record = {
            'opponent': winner,
            'result': 'L',
            'method': method,
            'date': fight_date,
            'rating_change': loser_new - loser_rating
        }
        self.fight_history[loser].append(fight_record)
        
        return {
            'winner_old': winner_rating,
            'winner_new': winner_new,
            'winner_change': winner_new - winner_rating,
            'loser_old': loser_rating,
            'loser_new': loser_new,
            'loser_change': loser_new - loser_rating
        }
    
    def apply_inactivity_decay(self, current_date=None, decay_threshold_days=365, 
                               decay_rate=0.02):
        """
        Apply rating decay for inactive fighters
        
        Args:
            current_date: Current date (defaults to now)
            decay_threshold_days: Days of inactivity before decay starts
            decay_rate: Percentage to decay per month of inactivity
        """
        if current_date is None:
            current_date = datetime.now()
        
        if isinstance(current_date, str):
            current_date = datetime.fromisoformat(current_date)
        
        decayed_fighters = []
        
        for fighter, data in self.ratings.items():
            if data['last_fight_date'] is None:
                continue
            
            last_fight = data['last_fight_date']
            if isinstance(last_fight, str):
                last_fight = datetime.fromisoformat(last_fight)
            
            days_inactive = (current_date - last_fight).days
            
            if days_inactive > decay_threshold_days:
                # Calculate decay
                months_inactive = (days_inactive - decay_threshold_days) / 30
                decay_multiplier = math.pow(1 - decay_rate, months_inactive)
                
                old_rating = data['rating']
                new_rating = self.initial_rating + (old_rating - self.initial_rating) * decay_multiplier
                
                self.ratings[fighter]['rating'] = new_rating
                
                decayed_fighters.append({
                    'fighter': fighter,
                    'days_inactive': days_inactive,
                    'old_rating': old_rating,
                    'new_rating': new_rating,
                    'decay': old_rating - new_rating
                })
        
        return decayed_fighters
    
    def get_all_ratings(self, sort_by_rating=True):
        """
        Get all fighter ratings
        
        Args:
            sort_by_rating: Whether to sort by rating (descending)
        
        Returns:
            List of (fighter, rating, fights) tuples
        """
        ratings_list = [
            (fighter, data['rating'], data['fights'], data['last_fight_date'])
            for fighter, data in self.ratings.items()
        ]
        
        if sort_by_rating:
            ratings_list.sort(key=lambda x: x[1], reverse=True)
        
        return ratings_list
    
    def get_top_n(self, n=10):
        """Get top N rated fighters"""
        all_ratings = self.get_all_ratings(sort_by_rating=True)
        return all_ratings[:n]
    
    def predict_fight_outcome(self, fighter1, fighter2):
        """
        Predict outcome of a fight between two fighters
        
        Returns:
            Dictionary with probabilities and expected winner
        """
        rating1 = self.get_rating(fighter1)
        rating2 = self.get_rating(fighter2)
        
        prob_fighter1_wins = self.expected_score(rating1, rating2)
        prob_fighter2_wins = 1 - prob_fighter1_wins
        
        expected_winner = fighter1 if prob_fighter1_wins > 0.5 else fighter2
        confidence = max(prob_fighter1_wins, prob_fighter2_wins)
        
        return {
            'fighter1': fighter1,
            'fighter2': fighter2,
            'fighter1_rating': rating1,
            'fighter2_rating': rating2,
            'fighter1_win_probability': prob_fighter1_wins,
            'fighter2_win_probability': prob_fighter2_wins,
            'expected_winner': expected_winner,
            'confidence': confidence
        }
