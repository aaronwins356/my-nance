"""
System Verification Script
Checks that all components of FightIQ are properly installed and configured
"""
import os
import sys

def check_directories():
    """Check that required directories exist"""
    print("Checking directories...")
    
    required_dirs = ['data', 'artifacts', 'cache', 'scripts', 'tests']
    missing = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing.append(dir_name)
    
    if missing:
        print(f"  ❌ Missing directories: {', '.join(missing)}")
        return False
    else:
        print("  ✅ All required directories exist")
        return True

def check_data_files():
    """Check that data files exist"""
    print("\nChecking data files...")
    
    required_files = [
        'data/ufc_rankings.csv',
        'data/fighter_stats.csv',
        'data/fight_results.csv',
        'data/fighters_top10_men.csv',
        'data/elo_ratings.csv'
    ]
    
    missing = []
    present = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            present.append(f"    {os.path.basename(file_path)}: {size_kb:.1f} KB")
        else:
            missing.append(file_path)
    
    if missing:
        print(f"  ⚠️  Missing data files: {len(missing)}/{len(required_files)}")
        for file_path in missing:
            print(f"    - {file_path}")
        print("\n  Run data pipeline to generate missing files:")
        print("    python scripts/scrape_rankings.py")
        print("    python scripts/scrape_fighter_stats.py")
        print("    python scripts/scrape_events.py")
        print("    python scripts/build_dataset.py")
        print("    python elo_pipeline.py")
        return False
    else:
        print("  ✅ All data files present")
        for item in present:
            print(item)
        return True

def check_model_artifacts():
    """Check that model artifacts exist"""
    print("\nChecking model artifacts...")
    
    required_artifacts = [
        'artifacts/model.pkl',
        'artifacts/model_metadata.json',
        'artifacts/feature_names.json'
    ]
    
    missing = []
    present = []
    
    for file_path in required_artifacts:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            present.append(f"    {os.path.basename(file_path)}: {size_kb:.1f} KB")
        else:
            missing.append(file_path)
    
    if missing:
        print(f"  ⚠️  Missing model artifacts: {len(missing)}/{len(required_artifacts)}")
        for file_path in missing:
            print(f"    - {file_path}")
        print("\n  Run training to generate missing artifacts:")
        print("    python train.py")
        return False
    else:
        print("  ✅ All model artifacts present")
        for item in present:
            print(item)
        return True

def check_python_packages():
    """Check that required Python packages are installed"""
    print("\nChecking Python packages...")
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'lightgbm',
        'streamlit',
        'plotly',
        'bs4',  # beautifulsoup4 imports as bs4
        'requests',
        'pytest'
    ]
    
    missing = []
    present = []
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            present.append(f"    {package}: {version}")
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"  ❌ Missing packages: {', '.join(missing)}")
        print("\n  Install missing packages:")
        print("    pip install -r requirements.txt")
        return False
    else:
        print("  ✅ All required packages installed")
        for item in present[:5]:  # Show first 5
            print(item)
        print(f"    ... and {len(present) - 5} more")
        return True

def check_imports():
    """Check that project modules can be imported"""
    print("\nChecking project imports...")
    
    modules = [
        'data_processing',
        'elo_system',
        'elo_pipeline',
        'train',
        'infer',
        'dashboard_app'
    ]
    
    failed = []
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except Exception as e:
            print(f"  ❌ {module_name}: {str(e)}")
            failed.append(module_name)
    
    return len(failed) == 0

def run_quick_test():
    """Run a quick prediction test"""
    print("\nRunning quick prediction test...")
    
    try:
        from infer import predict_fight
        
        # Try a prediction with common fighters
        result = predict_fight("Islam Makhachev", "Charles Oliveira")
        
        if result and 'predicted_winner' in result:
            print(f"  ✅ Prediction successful")
            print(f"    Winner: {result['predicted_winner']}")
            print(f"    Confidence: {result['confidence']:.1%}")
            return True
        else:
            print("  ❌ Prediction returned invalid result")
            return False
    
    except Exception as e:
        print(f"  ❌ Prediction failed: {str(e)}")
        return False

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("FightIQ System Verification")
    print("=" * 60)
    
    checks = [
        ("Directories", check_directories),
        ("Data Files", check_data_files),
        ("Model Artifacts", check_model_artifacts),
        ("Python Packages", check_python_packages),
        ("Project Imports", check_imports),
        ("Prediction Test", run_quick_test)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ❌ Error during {name} check: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 System verification complete - All checks passed!")
        print("\nYou can now:")
        print("  1. Run the dashboard: streamlit run dashboard_app.py")
        print("  2. Make predictions: python infer.py 'Fighter1' 'Fighter2'")
        print("  3. Run tests: pytest tests/ -v")
        return 0
    else:
        print("\n⚠️  Some checks failed - please fix the issues above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
