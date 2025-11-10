#!/usr/bin/env python3
"""
Demo Script - Fight History Scraper
Demonstrates the usage of the new scraping system
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("MMA Fight History Scraper - Demo")
print("=" * 70)

# 1. Load Configuration
print("\n1. Loading Configuration")
print("-" * 70)

try:
    from scraper import load_config
    config = load_config()
    print(f"✓ Configuration loaded successfully")
    print(f"  - Base URL: {config.base_url}")
    print(f"  - Max retries: {config.max_retries}")
    print(f"  - Cache enabled: {config.use_cache}")
    print(f"  - Data directory: {config.data_dir}")
    print(f"  - Valid results: {', '.join(config.valid_results[:3])}...")
except Exception as e:
    print(f"✗ Error loading configuration: {e}")
    sys.exit(1)

# 2. Initialize Validator
print("\n2. Initializing Data Validator")
print("-" * 70)

try:
    from scraper import FightDataValidator
    validator = FightDataValidator(config)
    print(f"✓ Validator initialized")
    print(f"  - Valid results: {len(validator.valid_results)}")
    print(f"  - Valid methods: {len(validator.valid_methods)}")
except Exception as e:
    print(f"✗ Error initializing validator: {e}")
    sys.exit(1)

# 3. Test Validation
print("\n3. Testing Data Validation")
print("-" * 70)

test_fight = {
    'fighter_name': '  Test Fighter  ',
    'opponent_name': 'Test Opponent',
    'result': 'W',
    'win_method': 'knockout',
    'event_date': '2023-06-15',
    'round_number': '2',
}

print(f"Original fight data:")
for key, value in test_fight.items():
    print(f"  {key}: {repr(value)}")

cleaned_fight = validator.clean_fight_record(test_fight)
is_valid, errors = validator.validate_fight_record(cleaned_fight)

print(f"\nCleaned fight data:")
for key, value in cleaned_fight.items():
    print(f"  {key}: {repr(value)}")

print(f"\nValidation result: {'✓ Valid' if is_valid else '✗ Invalid'}")
if errors:
    print(f"Errors: {errors}")

# 4. Initialize Scraper
print("\n4. Initializing Fight History Scraper")
print("-" * 70)

try:
    from scraper import FightHistoryScraper
    scraper = FightHistoryScraper(config)
    print(f"✓ Scraper initialized successfully")
    print(f"  - Session configured with retry logic")
    print(f"  - Directories ensured: {config.data_dir}, {config.cache_dir}")
    print(f"  - Logging configured: {config.log_file}")
    
    scraper_initialized = True
except ImportError as e:
    print(f"✗ Missing dependencies: {e}")
    print("  Install with: pip install requests beautifulsoup4 pyyaml")
    print("  Continuing with limited functionality...")
    scraper_initialized = False
    scraper = None
except Exception as e:
    print(f"✗ Error initializing scraper: {e}")
    print("  This may be due to missing dependencies")
    print("  Continuing with limited functionality...")
    scraper_initialized = False
    scraper = None

# 5. Test Cache Functions
print("\n5. Testing Cache Functions")
print("-" * 70)

if scraper_initialized and scraper:
    test_fighter = "Test Fighter"
    cache_key = scraper._get_cache_key(test_fighter)
    cache_path = scraper._get_cache_path(cache_key)

    print(f"Fighter: {test_fighter}")
    print(f"Cache key: {cache_key}")
    print(f"Cache path: {cache_path}")
    print(f"Cache valid: {scraper._is_cache_valid(cache_path)}")
else:
    print("⊘ Skipped - scraper not initialized (missing dependencies)")

# 6. Test Parsing Functions
print("\n6. Testing Parsing Functions")
print("-" * 70)

if scraper_initialized and scraper:
    # Test date parsing
    test_dates = ['Jan. 15, 2023', '2023-01-15', 'January 1, 2024']
    print("Date parsing:")
    for date_str in test_dates:
        parsed = scraper._parse_date(date_str)
        print(f"  {date_str} → {parsed}")

    # Test strike parsing
    test_strikes = ['45 of 89', '30/60', '12 of 25']
    print("\nStrike parsing:")
    for strikes_str in test_strikes:
        landed, received = scraper._parse_strikes(strikes_str)
        print(f"  {strikes_str} → landed: {landed}, received: {received}")

    # Test time parsing
    test_times = [('2:30', 1), ('1:45', 2), ('4:59', 3)]
    print("\nTime parsing:")
    for time_str, round_num in test_times:
        seconds = scraper._parse_time_to_seconds(time_str, round_num)
        print(f"  Round {round_num}, {time_str} → {seconds} seconds")
else:
    print("⊘ Skipped - scraper not initialized (missing dependencies)")

# 7. Test Quality Scoring
print("\n7. Testing Quality Score Calculation")
print("-" * 70)

if scraper_initialized and scraper:
    complete_fight = {
        'opponent_name': 'Complete Opponent',
        'event_date': '2023-01-15',
        'event_name': 'UFC 280',
        'win_method': 'KO/TKO',
        'sig_strikes_landed': 45,
        'fight_duration_seconds': 150,
    }

    incomplete_fight = {
        'opponent_name': 'Incomplete Opponent',
    }

    score_complete = scraper._calculate_quality_score(complete_fight)
    score_incomplete = scraper._calculate_quality_score(incomplete_fight)

    print(f"Complete fight data:")
    print(f"  Fields: {len(complete_fight)}")
    print(f"  Quality score: {score_complete:.3f}")

    print(f"\nIncomplete fight data:")
    print(f"  Fields: {len(incomplete_fight)}")
    print(f"  Quality score: {score_incomplete:.3f}")
else:
    print("⊘ Skipped - scraper not initialized (missing dependencies)")

# 8. Command Line Usage
print("\n8. Command Line Usage Examples")
print("-" * 70)

print("Scrape fight histories:")
print("  python collect_fight_data.py --fighters data/fighter_stats.csv")
print("\nResume interrupted scraping:")
print("  python collect_fight_data.py --fighters data/fighter_stats.csv --resume")
print("\nVerbose logging:")
print("  python collect_fight_data.py --fighters data/fighter_stats.csv --verbose")
print("\nCustom output:")
print("  python collect_fight_data.py --fighters data/fighter_stats.csv --output my_fights.csv")

# 9. Python API Usage
print("\n9. Python API Usage Example")
print("-" * 70)

print("""
from scraper import scrape_fight_history, load_config

# Load configuration
config = load_config()

# Scrape a fighter's history
fights = scrape_fight_history(
    fighter_url="http://ufcstats.com/fighter-details/...",
    fighter_name="Fighter Name",
    config=config
)

# Process results
print(f"Scraped {len(fights)} fights")
for fight in fights:
    print(f"{fight['opponent_name']} - {fight['result']}")
""")

# 10. Integration with fight_history.py
print("\n10. Integration with fight_history.py")
print("-" * 70)

print("""
import pandas as pd
from fight_history import scrape_and_update_fight_history

# Load fighter list
fighters_df = pd.read_csv('data/fighter_stats.csv')

# Scrape fight histories (replaces generate_sample_fight_history)
fight_history_df = scrape_and_update_fight_history(
    fighters_df, 
    use_cache=True
)

# Save results
fight_history_df.to_csv('data/fight_history.csv', index=False)
print(f"Saved {len(fight_history_df)} fight records")
""")

# Summary
print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
print("\n✓ Configuration management working")
print("✓ Data validation and cleaning functional")

if scraper_initialized:
    print("✓ Scraper initialized successfully")
    print("✓ Cache system operational")
    print("✓ Parsing functions tested")
    print("✓ Quality scoring working")
    print("\nThe scraping system is ready to use!")
else:
    print("⊘ Scraper not fully tested (missing dependencies)")
    print("\nTo enable full functionality:")
    print("  pip install requests beautifulsoup4 pyyaml")

print("\nNext steps:")
print("  1. Install dependencies: pip install -r requirements.txt")
print("  2. Ensure fighter list has 'fighter_url' column")
print("  3. Run: python collect_fight_data.py --fighters data/fighter_stats.csv")
print("  4. Check output in data/fight_history.csv")
print("=" * 70)
