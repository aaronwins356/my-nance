"""
Test Streamlit Import
Validates that dashboard can be imported without errors
"""
import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_streamlit_installed():
    """Test that streamlit is installed"""
    try:
        import streamlit
        assert True
    except ImportError:
        pytest.fail("Streamlit is not installed")

def test_plotly_installed():
    """Test that plotly is installed"""
    try:
        import plotly
        assert True
    except ImportError:
        pytest.fail("Plotly is not installed")

def test_dashboard_imports():
    """Test that dashboard can be imported"""
    try:
        # This will test if all dependencies are available
        import dashboard_app
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import dashboard: {e}")

def test_infer_imports():
    """Test that inference module can be imported"""
    try:
        import infer
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import infer module: {e}")

def test_data_processing_imports():
    """Test that data processing module can be imported"""
    try:
        import data_processing
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import data_processing module: {e}")

def test_elo_system_imports():
    """Test that ELO system can be imported"""
    try:
        import elo_system
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import elo_system module: {e}")

def test_train_imports():
    """Test that training module can be imported"""
    try:
        import train
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import train module: {e}")

def test_required_packages():
    """Test that all required packages are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'streamlit',
        'plotly',
        'xgboost',
        'lightgbm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        pytest.fail(f"Missing packages: {', '.join(missing_packages)}")

def test_dashboard_functions_exist():
    """Test that key dashboard functions exist"""
    import dashboard_app
    
    # Check that main navigation functions exist
    assert hasattr(dashboard_app, 'home_page'), "home_page function missing"
    assert hasattr(dashboard_app, 'fighter_profiles_page'), "fighter_profiles_page missing"
    assert hasattr(dashboard_app, 'rankings_page'), "rankings_page missing"
    assert hasattr(dashboard_app, 'simulator_page'), "simulator_page missing"
    assert hasattr(dashboard_app, 'feature_importance_page'), "feature_importance_page missing"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
