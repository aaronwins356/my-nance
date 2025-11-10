"""
Data Validator Module
Validates and cleans scraped fight history data
"""
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FightDataValidator:
    """Validates and cleans fight history data"""
    
    def __init__(self, config):
        """
        Initialize validator with configuration
        
        Args:
            config: ScraperConfig object with validation rules
        """
        self.config = config
        self.valid_results = set(config.valid_results)
        self.valid_methods = set(config.valid_methods)
    
    def validate_fight_record(self, fight: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single fight record
        
        Args:
            fight: Dictionary containing fight data
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        required_fields = ['fighter_name', 'opponent_name', 'result', 'event_date']
        for field in required_fields:
            if field not in fight or not fight[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate fighter name
        if 'fighter_name' in fight:
            if not self._is_valid_name(fight['fighter_name']):
                errors.append(f"Invalid fighter name: {fight['fighter_name']}")
        
        # Validate opponent name
        if 'opponent_name' in fight:
            if not self._is_valid_name(fight['opponent_name']):
                errors.append(f"Invalid opponent name: {fight['opponent_name']}")
        
        # Validate result
        if 'result' in fight:
            if fight['result'] not in self.valid_results:
                errors.append(f"Invalid result: {fight['result']}")
        
        # Validate method
        if 'win_method' in fight and fight['win_method']:
            # Normalize method for validation
            normalized_method = self._normalize_method(fight['win_method'])
            if not any(valid in normalized_method for valid in self.valid_methods):
                logger.debug(f"Method '{fight['win_method']}' not in predefined list, but allowing it")
        
        # Validate date
        if 'event_date' in fight:
            if not self._is_valid_date(fight['event_date']):
                errors.append(f"Invalid date: {fight['event_date']}")
        
        # Validate round
        if 'round_number' in fight and fight['round_number']:
            try:
                round_num = int(fight['round_number'])
                if round_num < self.config.min_round or round_num > self.config.max_round:
                    errors.append(f"Invalid round number: {round_num}")
            except (ValueError, TypeError):
                errors.append(f"Invalid round number format: {fight['round_number']}")
        
        return len(errors) == 0, errors
    
    def clean_fight_record(self, fight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalize a fight record
        
        Args:
            fight: Dictionary containing fight data
        
        Returns:
            Cleaned fight dictionary
        """
        cleaned = fight.copy()
        
        # Normalize text fields
        text_fields = ['fighter_name', 'opponent_name', 'event_name', 'win_method', 'fight_ending_technique']
        for field in text_fields:
            if field in cleaned and cleaned[field]:
                cleaned[field] = self._normalize_text(cleaned[field])
        
        # Normalize result
        if 'result' in cleaned:
            cleaned['result'] = self._normalize_result(cleaned['result'])
        
        # Normalize method
        if 'win_method' in cleaned and cleaned['win_method']:
            cleaned['win_method'] = self._normalize_method(cleaned['win_method'])
        
        # Ensure numeric fields are proper types
        numeric_fields = {
            'round_number': int,
            'fight_duration_seconds': int,
            'sig_strikes_landed': int,
            'sig_strikes_received': int,
            'total_strikes_landed': int,
            'total_strikes_received': int,
            'takedowns_landed': int,
            'takedowns_attempted': int,
        }
        
        for field, field_type in numeric_fields.items():
            if field in cleaned and cleaned[field] is not None:
                try:
                    if cleaned[field] == '' or cleaned[field] == 'N/A':
                        cleaned[field] = None
                    else:
                        cleaned[field] = field_type(cleaned[field])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {field}={cleaned[field]} to {field_type.__name__}")
                    cleaned[field] = None
        
        return cleaned
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if fighter name is valid"""
        if not name or not isinstance(name, str):
            return False
        
        # Remove whitespace
        name = name.strip()
        
        # Check length
        if len(name) < 2 or len(name) > 100:
            return False
        
        # Check against invalid patterns
        invalid_patterns = getattr(self.config, 'invalid_name_patterns', ['N/A', 'Unknown', 'TBD'])
        for pattern in invalid_patterns:
            if pattern.lower() in name.lower():
                return False
        
        return True
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Check if date string is valid"""
        if not date_str:
            return False
        
        try:
            # Try parsing the date
            date_obj = datetime.strptime(date_str, self.config.date_format)
            
            # Check year range
            if date_obj.year < self.config.min_year or date_obj.year > self.config.max_year:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text: strip whitespace, fix capitalization"""
        if not text:
            return text
        
        # Strip whitespace
        text = text.strip()
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _normalize_result(self, result: str) -> str:
        """Normalize fight result"""
        if not result:
            return result
        
        result = result.strip().upper()
        
        # Map common variations
        result_map = {
            'W': 'Win',
            'WIN': 'Win',
            'L': 'Loss',
            'LOSS': 'Loss',
            'D': 'Draw',
            'DRAW': 'Draw',
            'NC': 'NC',
            'NO CONTEST': 'NC',
            'DQ': 'DQ',
            'DISQUALIFICATION': 'DQ',
        }
        
        return result_map.get(result, result.title())
    
    def _normalize_method(self, method: str) -> str:
        """Normalize fight method"""
        if not method:
            return method
        
        method = method.strip()
        
        # Common mappings
        if 'KO' in method.upper() or 'TKO' in method.upper() or 'KNOCKOUT' in method.upper():
            return 'KO/TKO'
        elif 'SUB' in method.upper() or 'SUBMISSION' in method.upper():
            return 'Submission'
        elif 'DEC' in method.upper() or 'DECISION' in method.upper():
            if 'UNANIMOUS' in method.upper():
                return 'Unanimous Decision'
            elif 'SPLIT' in method.upper():
                return 'Split Decision'
            elif 'MAJORITY' in method.upper():
                return 'Majority Decision'
            else:
                return 'Decision'
        elif 'DOCTOR' in method.upper():
            return 'TKO - Doctor\'s Stoppage'
        
        return method


def validate_fight_data(fights: List[Dict[str, Any]], config) -> List[Dict[str, Any]]:
    """
    Validate and clean a list of fight records
    
    Args:
        fights: List of fight dictionaries
        config: ScraperConfig object
    
    Returns:
        List of valid, cleaned fight records
    """
    validator = FightDataValidator(config)
    valid_fights = []
    
    for i, fight in enumerate(fights):
        # Clean the record
        cleaned_fight = validator.clean_fight_record(fight)
        
        # Validate
        is_valid, errors = validator.validate_fight_record(cleaned_fight)
        
        if is_valid:
            valid_fights.append(cleaned_fight)
        else:
            logger.warning(f"Fight record {i} failed validation: {errors}")
    
    logger.info(f"Validated {len(valid_fights)}/{len(fights)} fight records")
    
    return valid_fights


if __name__ == '__main__':
    # Test validation
    from .config import load_config
    
    config = load_config()
    validator = FightDataValidator(config)
    
    # Test fight record
    test_fight = {
        'fighter_name': '  John Doe  ',
        'opponent_name': 'Jane Smith',
        'result': 'W',
        'win_method': 'KO',
        'event_date': '2023-06-15',
        'round_number': '2',
    }
    
    cleaned = validator.clean_fight_record(test_fight)
    is_valid, errors = validator.validate_fight_record(cleaned)
    
    print(f"Original: {test_fight}")
    print(f"Cleaned: {cleaned}")
    print(f"Valid: {is_valid}, Errors: {errors}")
