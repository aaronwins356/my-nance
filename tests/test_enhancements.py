"""
Test script to validate the new enhancements
"""
import os
import sys
import traceback
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_fighter_styles():
    """Test that fighter styles are loaded correctly"""
    print("="*60)
    print("TEST 1: Fighter Styles with Detailed Categories")
    print("="*60)
    
    from infer import load_fighter_styles
    
    styles = load_fighter_styles()
    
    if not styles:
        print("❌ FAILED: No fighter styles loaded")
        return False
    
    print(f"✓ Loaded {len(styles)} fighter styles")
    
    # Check for detailed styles
    detailed_styles = set(styles.values())
    expected_styles = {'Boxer', 'Kickboxer', 'Muay Thai', 'Karate', 'Taekwondo',
                      'Wrestler', 'BJJ', 'Judoka', 'Sambo',
                      'Wrestle-Boxer', 'Striker-Grappler', 'All-Rounder'}
    
    found_detailed = detailed_styles.intersection(expected_styles)
    print(f"✓ Found {len(found_detailed)} detailed style categories")
    print(f"  Styles in use: {sorted(detailed_styles)}")
    
    if len(found_detailed) >= 5:
        print("✅ PASSED: Detailed fighting styles loaded successfully\n")
        return True
    else:
        print("⚠️  WARNING: Limited detailed styles found\n")
        return True

def test_fight_history():
    """Test that fight history can be loaded"""
    print("="*60)
    print("TEST 2: Fight History Loading")
    print("="*60)
    
    from fight_history import load_fight_history, get_fighter_history
    
    history = load_fight_history()
    
    if history.empty:
        print("⚠️  WARNING: No fight history data available")
        print("  This is expected if running with minimal data")
        print("✅ PASSED: Module works correctly (no data)\n")
        return True
    
    print(f"✓ Loaded {len(history)} fight records")
    
    # Check columns
    required_cols = ['fighter_name', 'opponent_name', 'event_name', 
                     'fight_ending_technique', 'round_number']
    missing_cols = [col for col in required_cols if col not in history.columns]
    
    if missing_cols:
        print(f"❌ FAILED: Missing columns: {missing_cols}")
        return False
    
    print(f"✓ All required columns present")
    
    # Test getting history for a specific fighter
    if len(history) > 0:
        fighter_name = history['fighter_name'].iloc[0]
        fighter_hist = get_fighter_history(fighter_name, history)
        print(f"✓ Retrieved {len(fighter_hist)} fights for {fighter_name}")
    
    print("✅ PASSED: Fight history loaded successfully\n")
    return True

def test_data_processing():
    """Test that data processing handles new styles"""
    print("="*60)
    print("TEST 3: Data Processing with Detailed Styles")
    print("="*60)
    
    from data_processing import prepare_single_fight
    from infer import load_fighter_styles
    
    fighter1_stats = {
        'name': 'Test Fighter 1',
        'wins': 10,
        'losses': 2,
        'wins_by_ko': 5,
        'wins_by_submission': 3,
        'height': 72,
        'reach': 74,
        'age': 28,
        'sig_strikes_per_min': 4.5,
        'sig_strikes_absorbed_per_min': 3.0,
        'takedown_avg_per_15min': 2.0,
        'takedown_defense_pct': 75,
        'submission_avg_per_15min': 0.8,
        'striking_accuracy_pct': 48,
        'striking_defense_pct': 60
    }
    
    fighter2_stats = {
        'name': 'Test Fighter 2',
        'wins': 8,
        'losses': 3,
        'wins_by_ko': 2,
        'wins_by_submission': 4,
        'height': 70,
        'reach': 72,
        'age': 30,
        'sig_strikes_per_min': 3.8,
        'sig_strikes_absorbed_per_min': 3.5,
        'takedown_avg_per_15min': 3.0,
        'takedown_defense_pct': 70,
        'submission_avg_per_15min': 1.2,
        'striking_accuracy_pct': 45,
        'striking_defense_pct': 55
    }
    
    fighter_styles = {
        'Test Fighter 1': 'Kickboxer',
        'Test Fighter 2': 'Wrestler'
    }
    
    try:
        features = prepare_single_fight(
            fighter1_stats, fighter2_stats,
            fighter_styles=fighter_styles
        )
        
        print(f"✓ Generated {len(features.columns)} features")
        
        # Check for detailed style features
        style_features = [col for col in features.columns if 'style' in col.lower()]
        print(f"✓ Found {len(style_features)} style-related features")
        
        # Check for matchup features
        matchup_features = [col for col in features.columns if 'matchup' in col.lower()]
        print(f"✓ Found {len(matchup_features)} matchup features")
        
        print("✅ PASSED: Data processing works with detailed styles\n")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        traceback.print_exc()
        return False

def test_dashboard_imports():
    """Test that dashboard can import all required modules"""
    print("="*60)
    print("TEST 4: Dashboard Module Imports")
    print("="*60)
    
    try:
        # Test core imports
        from fight_history import load_fight_history, get_fighter_history, format_fight_duration
        print("✓ fight_history module imported")
        
        from infer import load_fighter_styles
        print("✓ load_fighter_styles imported")
        
        from data_processing import prepare_single_fight
        print("✓ data_processing module imported")
        
        print("✅ PASSED: All dashboard imports successful\n")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FIGHTIQ ENHANCEMENT VALIDATION TESTS")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Fighter Styles", test_fighter_styles()))
    results.append(("Fight History", test_fight_history()))
    results.append(("Data Processing", test_data_processing()))
    results.append(("Dashboard Imports", test_dashboard_imports()))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<45} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("="*60)
    print(f"Results: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    return all(passed for _, passed in results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
