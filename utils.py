"""
Utility functions shared across modules
"""

def american_odds_to_probability(odds):
    """
    Convert American odds format to implied probability
    
    Args:
        odds: American odds (negative for favorite, positive for underdog)
              e.g., -200 means bet $200 to win $100
              e.g., +150 means bet $100 to win $150
    
    Returns:
        float: Implied probability between 0 and 1
    
    Examples:
        >>> american_odds_to_probability(-200)
        0.6666666666666666
        >>> american_odds_to_probability(+150)
        0.4
    """
    if odds < 0:
        # Favorite: odds = -X means you bet X to win 100
        prob = (-odds) / (-odds + 100)
    else:
        # Underdog: odds = +X means you bet 100 to win X
        prob = 100 / (odds + 100)
    return prob
