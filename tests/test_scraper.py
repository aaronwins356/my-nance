"""
Test Scraper Modules
Validates scraping functionality and data generation
"""
import os
import pytest
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts import scrape_rankings, scrape_fighter_stats, scrape_events

def test_scrape_rankings_fallback():
    """Test that rankings fallback data works"""
    rankings = scrape_rankings.get_fallback_rankings()
    
    assert isinstance(rankings, dict), "Rankings should be a dictionary"
    assert len(rankings) > 0, "Rankings should not be empty"
    
    # Check structure
    for weight_class, fighters in rankings.items():
        assert isinstance(fighters, list), f"{weight_class} fighters should be a list"
        assert len(fighters) > 0, f"{weight_class} should have fighters"
        
        for fighter in fighters:
            assert 'rank' in fighter, "Fighter should have rank"
            assert 'name' in fighter, "Fighter should have name"

def test_generate_fighter_stats():
    """Test fighter stats generation"""
    stats = scrape_fighter_stats.generate_fighter_stats("Test Fighter")
    
    # Check required fields
    required_fields = [
        'name', 'wins', 'losses', 'height', 'reach', 'age',
        'sig_strikes_per_min', 'takedown_avg_per_15min', 'data_quality_score'
    ]
    
    for field in required_fields:
        assert field in stats, f"Missing field: {field}"
    
    # Check data validity
    assert stats['wins'] >= 0, "Wins should be non-negative"
    assert stats['losses'] >= 0, "Losses should be non-negative"
    assert 18 <= stats['age'] <= 50, "Age out of reasonable range"
    assert 60 <= stats['height'] <= 80, "Height out of reasonable range"
    assert stats['data_quality_score'] >= 0, "Quality score should be non-negative"

def test_generate_realistic_events():
    """Test event generation"""
    events = scrape_events.generate_realistic_events(num_events=5)
    
    assert isinstance(events, list), "Events should be a list"
    assert len(events) == 5, "Should generate 5 events"
    
    for event in events:
        assert 'event_id' in event, "Event should have ID"
        assert 'event_name' in event, "Event should have name"
        assert 'date' in event, "Event should have date"
        assert 'fights' in event, "Event should have fights"
        
        # Check fights
        assert len(event['fights']) > 0, "Event should have fights"
        
        for fight in event['fights']:
            assert 'fighter1' in fight, "Fight should have fighter1"
            assert 'fighter2' in fight, "Fight should have fighter2"
            assert 'winner' in fight, "Fight should have winner"
            assert 'method' in fight, "Fight should have method"
            assert fight['winner'] in [fight['fighter1'], fight['fighter2']], "Winner should be one of the fighters"

def test_cache_key_generation():
    """Test cache key generation is consistent"""
    name = "Test Fighter"
    
    key1 = scrape_fighter_stats.get_cache_key(name)
    key2 = scrape_fighter_stats.get_cache_key(name)
    
    assert key1 == key2, "Cache keys should be consistent"
    assert len(key1) == 32, "Cache key should be MD5 hash (32 chars)"

def test_fighter_stats_consistency():
    """Test that generated stats are consistent for same fighter"""
    name = "Consistency Test Fighter"
    
    stats1 = scrape_fighter_stats.generate_fighter_stats(name)
    stats2 = scrape_fighter_stats.generate_fighter_stats(name)
    
    # Should generate same stats for same name (seeded random)
    assert stats1['wins'] == stats2['wins'], "Wins should be consistent"
    assert stats1['height'] == stats2['height'], "Height should be consistent"
    assert stats1['age'] == stats2['age'], "Age should be consistent"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
