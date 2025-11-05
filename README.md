# FightIQ - MMA Prediction System 🥊

An advanced machine learning system for predicting Mixed Martial Arts (MMA) fight outcomes using comprehensive fighter statistics, ELO ratings, and calibrated probability models.

## Features

### 🎯 Core Capabilities
- **AI-Powered Predictions**: Calibrated win probability predictions using XGBoost, Random Forest, LightGBM, and Decision Tree models
- **ELO Rating System**: Dynamic fighter rankings with inactivity decay and title fight bonuses
- **Interactive Dashboard**: Streamlit-based web interface with fighter profiles, rankings, and fight simulator
- **Automated Updates**: Weekly data pipeline for scraping new events and retraining models
- **Feature Explainability**: SHAP-ready architecture with feature importance analysis
- **Offline Operation**: Runs entirely without external API dependencies using cached/generated data

### 📊 Data Pipeline
1. **Rankings Scraper**: Pulls UFC official rankings (with fallback data)
2. **Fighter Stats Scraper**: Comprehensive fighter statistics with intelligent caching
3. **Events Scraper**: Historical fight results with method and outcome data
4. **Dataset Builder**: Merges and cleans data with quality scoring
5. **ELO Pipeline**: Updates ratings based on fight outcomes with special adjustments

### 🤖 ML Models
- Primary: **XGBoost** (selected based on test accuracy)
- Alternatives: Random Forest, LightGBM, Decision Tree
- Calibrated probability outputs for reliable confidence estimates
- 50+ engineered features including physical attributes, performance metrics, and ELO ratings

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Quick Start

#### Windows (PowerShell)
```powershell
# Clone the repository
git clone https://github.com/aaronwins356/MMA-Predictor.git
cd MMA-Predictor

# Run the launcher (handles setup automatically)
.\scripts\launch_fightiq.ps1
```

#### Windows (Batch)
```cmd
scripts\launch_fightiq.bat
```

#### Linux/Mac
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python scripts/scrape_rankings.py
python scripts/scrape_fighter_stats.py
python scripts/scrape_events.py
python scripts/build_dataset.py

# Update ELO ratings
python elo_pipeline.py

# Train model
python train.py

# Launch dashboard
streamlit run dashboard_app.py
```

## Usage

### Dashboard
The main interface is accessible via the Streamlit dashboard:

```bash
streamlit run dashboard_app.py
```

Navigate to http://localhost:8501 in your browser to access:
- **Home**: System overview and model metrics
- **Fighter Profiles**: Detailed fighter statistics and performance radar charts
- **Rankings**: ELO-based rankings by weight class
- **Fight Simulator**: Predict outcomes between any two fighters
- **Feature Importance**: Understand key factors in predictions

### Command Line Predictions

```bash
# Predict a single fight
python infer.py "Islam Makhachev" "Charles Oliveira" true

# Run from Python
from infer import predict_fight, print_prediction

result = predict_fight("Max Holloway", "Alexander Volkanovski")
print_prediction(result)
```

### Auto-Update System

```bash
# Check if update is needed
python scripts/auto_update.py --check

# Run quick update (events and ELO only)
python scripts/auto_update.py --quick

# Run full update (including model retraining)
python scripts/auto_update.py --full

# Scheduled weekly check
python scripts/auto_update.py --schedule
```

## Project Structure

```
MMA-Predictor/
├── data/                      # Data files (CSV)
│   ├── ufc_rankings.csv
│   ├── fighter_stats.csv
│   ├── fight_results.csv
│   ├── fighters_top10_men.csv
│   └── elo_ratings.csv
├── artifacts/                 # Model artifacts
│   ├── model.pkl
│   ├── model_metadata.json
│   └── feature_names.json
├── cache/                     # Cached HTML/JSON responses
├── scripts/                   # Data pipeline scripts
│   ├── scrape_rankings.py
│   ├── scrape_fighter_stats.py
│   ├── scrape_events.py
│   ├── build_dataset.py
│   ├── auto_update.py
│   ├── launch_fightiq.ps1
│   └── launch_fightiq.bat
├── tests/                     # Test suite
│   ├── test_csv_schema.py
│   ├── test_model_training.py
│   ├── test_scraper.py
│   └── test_streamlit_import.py
├── dashboard_app.py           # Streamlit dashboard
├── data_processing.py         # Feature engineering
├── elo_system.py              # ELO calculation engine
├── elo_pipeline.py            # ELO update orchestrator
├── infer.py                   # Prediction engine
├── train.py                   # Model training pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_model_training.py -v
```

All tests should pass:
- ✅ CSV Schema validation (10 tests)
- ✅ Model training validation (9 tests)
- ✅ Scraper functionality (5 tests)
- ✅ Streamlit imports (9 tests)

## Model Performance

Current model (XGBoost):
- **Test Accuracy**: ~60%
- **Test AUC**: 0.55
- **Cross-Validation**: 5-fold CV for robustness

### Top Features by Importance
1. ELO difference
2. Fighter ELO ratings
3. Experience difference
4. Submission threat differential
5. Finish rate differential
6. Physical advantages (height/reach)
7. Striking metrics
8. Takedown statistics

## Data Sources

### Online Mode (when available)
- UFC.com for official rankings
- UFCStats.com for detailed fighter statistics
- Tapology.com as fallback source

### Offline Mode (default)
- Fallback rankings for 9 men's divisions
- Generated fighter statistics with realistic distributions
- Historical fight data generation

## Configuration

### ELO System Parameters
```python
k_factor = 32              # Rating change multiplier
initial_rating = 1500      # Starting rating for new fighters
decay_threshold = 365      # Days before inactivity decay starts
decay_rate = 0.02          # Monthly decay percentage
```

### Model Hyperparameters
```python
# XGBoost (Best Model)
n_estimators = 100
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
```

## Advanced Features

### Feature Engineering
The system generates 50+ features including:
- Base stats (wins, losses, age, physical attributes)
- Derived metrics (win rate, finish rate, experience)
- Performance indicators (striking, defense, takedowns)
- Comparative features (differences between fighters)
- ELO ratings and differentials

### Model Calibration
Predictions use `CalibratedClassifierCV` for reliable probability estimates:
- Sigmoid calibration method
- 5-fold cross-validation
- Ensures probabilities sum to 1.0

### Data Quality Scoring
Each dataset includes quality metrics:
- Completeness check (missing values)
- Outlier detection (3x IQR threshold)
- Overall quality score (0-1 scale)

## Troubleshooting

### Common Issues

**ModuleNotFoundError**: Install dependencies
```bash
pip install -r requirements.txt
```

**Model file not found**: Train the model first
```bash
python train.py
```

**No data available**: Run the data pipeline
```bash
python scripts/scrape_rankings.py
python scripts/scrape_fighter_stats.py
python scripts/scrape_events.py
python scripts/build_dataset.py
```

**Dashboard won't start**: Check Streamlit installation
```bash
pip install streamlit
streamlit run dashboard_app.py
```

## Future Enhancements

### Planned Features
- [ ] SHAP value visualizations in dashboard
- [ ] Real-time UFC Stats API integration
- [ ] Historical backtest validation
- [ ] More sophisticated feature engineering
- [ ] Ensemble model combinations
- [ ] Fighter style matchup analysis
- [ ] Injury and camp data integration
- [ ] Fight outcome method prediction (not just winner)

### Potential Improvements
- Neural network models (LSTM for fight sequences)
- Betting odds integration for market comparison
- Social media sentiment analysis
- Video/frame analysis for technique detection
- Multi-label classification for finish method

## Contributing

This is a personal project, but suggestions are welcome. Feel free to:
1. Open issues for bugs or feature requests
2. Submit pull requests with improvements
3. Share feedback on model performance
4. Suggest new data sources or features

## License

This project is for educational and research purposes. Fight data is publicly available, but check UFC's terms of service for commercial use restrictions.

## Acknowledgments

- UFC for official rankings data
- UFCStats.com for comprehensive fight statistics
- Scikit-learn, XGBoost, LightGBM for ML tools
- Streamlit for the dashboard framework
- Plotly for interactive visualizations

## Contact

For questions or feedback, open an issue on the GitHub repository.

---

**FightIQ** - Making MMA predictions data-driven and transparent 🥊📊
