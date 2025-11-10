"""
Test Fight History Scraper Module
Tests for the new scraping functionality
"""
import os
import sys
import pytest
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scraper.config import load_config, ScraperConfig
from scraper.validator import FightDataValidator, validate_fight_data
from scraper.fight_scraper import FightHistoryScraper


class TestScraperConfig:
    """Test configuration loading"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = ScraperConfig()
        
        assert config.base_url == "http://ufcstats.com"
        assert config.max_retries == 3
        assert config.use_cache is True
        assert config.data_dir == "data"
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file"""
        config = load_config()
        
        assert config is not None
        assert isinstance(config, ScraperConfig)
        assert config.base_url is not None
        assert config.timeout > 0
    
    def test_config_validation_rules(self):
        """Test that config has validation rules"""
        config = load_config()
        
        assert "Win" in config.valid_results
        assert "Loss" in config.valid_results
        assert "KO/TKO" in config.valid_methods or any("KO" in m for m in config.valid_methods)
        assert config.min_year >= 1990
        assert config.max_year >= 2024


class TestFightDataValidator:
    """Test data validation functionality"""
    
    def test_validator_initialization(self):
        """Test validator can be initialized"""
        config = load_config()
        validator = FightDataValidator(config)
        
        assert validator is not None
        assert validator.config == config
    
    def test_validate_valid_fight(self):
        """Test validation of a valid fight record"""
        config = load_config()
        validator = FightDataValidator(config)
        
        fight = {
            'fighter_name': 'John Doe',
            'opponent_name': 'Jane Smith',
            'result': 'Win',
            'win_method': 'KO/TKO',
            'event_date': '2023-06-15',
            'round_number': 2,
        }
        
        is_valid, errors = validator.validate_fight_record(fight)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_missing_required_field(self):
        """Test validation fails for missing required fields"""
        config = load_config()
        validator = FightDataValidator(config)
        
        fight = {
            'fighter_name': 'John Doe',
            # Missing opponent_name
            'result': 'Win',
            'event_date': '2023-06-15',
        }
        
        is_valid, errors = validator.validate_fight_record(fight)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any('opponent_name' in err for err in errors)
    
    def test_validate_invalid_result(self):
        """Test validation fails for invalid result"""
        config = load_config()
        validator = FightDataValidator(config)
        
        fight = {
            'fighter_name': 'John Doe',
            'opponent_name': 'Jane Smith',
            'result': 'Maybe',  # Invalid
            'event_date': '2023-06-15',
        }
        
        is_valid, errors = validator.validate_fight_record(fight)
        
        assert is_valid is False
        assert any('result' in err.lower() for err in errors)
    
    def test_validate_invalid_date(self):
        """Test validation fails for invalid date"""
        config = load_config()
        validator = FightDataValidator(config)
        
        fight = {
            'fighter_name': 'John Doe',
            'opponent_name': 'Jane Smith',
            'result': 'Win',
            'event_date': '1800-01-01',  # Too old
        }
        
        is_valid, errors = validator.validate_fight_record(fight)
        
        assert is_valid is False
        assert any('date' in err.lower() for err in errors)
    
    def test_clean_fight_record(self):
        """Test cleaning of fight record"""
        config = load_config()
        validator = FightDataValidator(config)
        
        fight = {
            'fighter_name': '  John Doe  ',  # Extra whitespace
            'opponent_name': 'Jane Smith',
            'result': 'W',  # Should normalize to 'Win'
            'win_method': 'knockout',  # Should normalize to 'KO/TKO'
            'event_date': '2023-06-15',
            'round_number': '2',  # String should convert to int
        }
        
        cleaned = validator.clean_fight_record(fight)
        
        assert cleaned['fighter_name'] == 'John Doe'
        assert cleaned['result'] == 'Win'
        assert cleaned['win_method'] == 'KO/TKO'
        assert cleaned['round_number'] == 2
        assert isinstance(cleaned['round_number'], int)
    
    def test_normalize_result(self):
        """Test result normalization"""
        config = load_config()
        validator = FightDataValidator(config)
        
        assert validator._normalize_result('W') == 'Win'
        assert validator._normalize_result('L') == 'Loss'
        assert validator._normalize_result('D') == 'Draw'
        assert validator._normalize_result('win') == 'Win'
        assert validator._normalize_result('LOSS') == 'Loss'
    
    def test_normalize_method(self):
        """Test method normalization"""
        config = load_config()
        validator = FightDataValidator(config)
        
        assert validator._normalize_method('knockout') == 'KO/TKO'
        assert validator._normalize_method('TKO') == 'KO/TKO'
        assert validator._normalize_method('submission') == 'Submission'
        assert validator._normalize_method('unanimous decision') == 'Unanimous Decision'
    
    def test_validate_fight_data_batch(self):
        """Test batch validation of multiple fights"""
        config = load_config()
        
        fights = [
            {
                'fighter_name': 'Fighter 1',
                'opponent_name': 'Opponent 1',
                'result': 'Win',
                'event_date': '2023-01-15',
            },
            {
                'fighter_name': 'Fighter 2',
                'opponent_name': 'Opponent 2',
                'result': 'Loss',
                'event_date': '2023-02-20',
            },
            {
                # Invalid - missing required fields
                'fighter_name': 'Fighter 3',
            },
        ]
        
        valid_fights = validate_fight_data(fights, config)
        
        # Should return only valid fights
        assert len(valid_fights) == 2


class TestFightHistoryScraper:
    """Test fight history scraper"""
    
    def test_scraper_initialization(self):
        """Test scraper can be initialized"""
        try:
            config = load_config()
            scraper = FightHistoryScraper(config)
            
            assert scraper is not None
            assert scraper.config == config
            assert scraper.validator is not None
        except ImportError:
            pytest.skip("Required dependencies not installed (requests, beautifulsoup4)")
    
    def test_cache_key_generation(self):
        """Test cache key generation is consistent"""
        try:
            config = load_config()
            scraper = FightHistoryScraper(config)
            
            name = "Test Fighter"
            key1 = scraper._get_cache_key(name)
            key2 = scraper._get_cache_key(name)
            
            assert key1 == key2
            assert len(key1) == 32  # MD5 hash
        except ImportError:
            pytest.skip("Required dependencies not installed")
    
    def test_parse_date(self):
        """Test date parsing functionality"""
        try:
            config = load_config()
            scraper = FightHistoryScraper(config)
            
            # Test various date formats
            date1 = scraper._parse_date('Jan. 15, 2023')
            date2 = scraper._parse_date('2023-01-15')
            
            assert '2023' in date1
            assert '2023-01-15' in date2 or date2 == '2023-01-15'
        except ImportError:
            pytest.skip("Required dependencies not installed")
    
    def test_parse_strikes(self):
        """Test strike parsing functionality"""
        try:
            config = load_config()
            scraper = FightHistoryScraper(config)
            
            landed, received = scraper._parse_strikes('45 of 89')
            assert landed == 45
            
            landed, received = scraper._parse_strikes('30/60')
            assert landed == 30
        except ImportError:
            pytest.skip("Required dependencies not installed")
    
    def test_parse_time_to_seconds(self):
        """Test time parsing to seconds"""
        try:
            config = load_config()
            scraper = FightHistoryScraper(config)
            
            # Test round 1, 2:30
            seconds = scraper._parse_time_to_seconds('2:30', 1)
            assert seconds == 150  # 2*60 + 30
            
            # Test round 2, 1:45
            seconds = scraper._parse_time_to_seconds('1:45', 2)
            assert seconds == 405  # 300 (round 1) + 1*60 + 45
        except ImportError:
            pytest.skip("Required dependencies not installed")
    
    def test_quality_score_calculation(self):
        """Test data quality score calculation"""
        try:
            config = load_config()
            scraper = FightHistoryScraper(config)
            
            # Complete fight record
            complete_fight = {
                'opponent_name': 'Opponent',
                'event_date': '2023-01-15',
                'event_name': 'UFC 280',
                'win_method': 'KO/TKO',
                'sig_strikes_landed': 45,
                'fight_duration_seconds': 150,
            }
            
            score = scraper._calculate_quality_score(complete_fight)
            assert score >= 0.9  # High quality
            
            # Incomplete fight record
            incomplete_fight = {
                'opponent_name': 'Opponent',
            }
            
            score = scraper._calculate_quality_score(incomplete_fight)
            assert score < 0.8  # Lower quality
        except ImportError:
            pytest.skip("Required dependencies not installed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
