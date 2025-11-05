# FightIQ Usage Guide

## Quick Start

### 1. Initial Setup (First Time Only)

```bash
# Windows PowerShell
.\scripts\launch_fightiq.ps1

# Windows Command Prompt
scripts\launch_fightiq.bat

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Data Pipeline

The data pipeline consists of 4 steps that must be run in order:

```bash
# Step 1: Scrape UFC rankings
python scripts/scrape_rankings.py

# Step 2: Scrape fighter statistics
python scripts/scrape_fighter_stats.py

# Step 3: Scrape fight events and results
python scripts/scrape_events.py

# Step 4: Build training dataset
python scripts/build_dataset.py
```

### 3. Update ELO Ratings

```bash
python elo_pipeline.py
```

This will:
- Calculate ELO ratings for all fighters based on fight history
- Apply inactivity decay for fighters who haven't fought recently
- Save ratings to `data/elo_ratings.csv`

### 4. Train the Model

```bash
python train.py
```

This will:
- Load and prepare the training data
- Train multiple models (DecisionTree, RandomForest, XGBoost, LightGBM)
- Compare models and select the best performer
- Calibrate the model for accurate probabilities
- Save the trained model to `artifacts/model.pkl`

Expected output:
- Model accuracy: ~60%
- Training time: 1-2 minutes
- Artifacts saved to `artifacts/` directory

### 5. Launch the Dashboard

```bash
streamlit run dashboard_app.py
```

Then open your browser to: http://localhost:8501

## Dashboard Features

### Home Page
- System overview and statistics
- Model performance metrics
- Key feature highlights

### Fighter Profiles
1. Select a fighter from the dropdown
2. View comprehensive stats including:
   - Record (wins-losses-draws)
   - ELO rating
   - Physical attributes (height, reach, age)
   - Performance metrics (striking, takedowns, submissions)
   - Win methods breakdown
3. Interactive radar chart showing fighter strengths

### Rankings
1. **Overall Rankings**: Top 20 fighters by ELO rating
2. **By Weight Class**: Rankings filtered by division
3. Interactive bar charts for visualization

### Fight Simulator
1. Select Fighter 1 from dropdown
2. Select Fighter 2 from dropdown
3. Check "Title Fight" if applicable
4. Click "Predict Fight Outcome"
5. View:
   - Win probabilities for each fighter
   - Predicted winner with confidence level
   - Visual probability comparison
   - Key insights about the matchup

### Feature Importance
- Bar chart showing top 20 most important features
- Feature descriptions and explanations
- Understand what drives predictions

## Command Line Usage

### Make Single Prediction

```bash
# Basic prediction
python infer.py "Fighter 1 Name" "Fighter 2 Name"

# Title fight prediction
python infer.py "Fighter 1 Name" "Fighter 2 Name" true
```

Example:
```bash
python infer.py "Islam Makhachev" "Charles Oliveira" true
```

Output:
```
============================================================
FIGHT PREDICTION
============================================================

Islam Makhachev vs Charles Oliveira
(Title Fight)

ELO Ratings:
  Islam Makhachev: 1482
  Charles Oliveira: 1497

Win Probabilities:
  Islam Makhachev: 57.5%
  Charles Oliveira: 42.5%

Predicted Winner: Islam Makhachev
Confidence: 57.5%

Key Insights:
  • Close matchup - fight could go either way
============================================================
```

### Python API

```python
from infer import predict_fight, print_prediction

# Make prediction
result = predict_fight("Jon Jones", "Tom Aspinall", is_title_fight=True)

# Print formatted result
print_prediction(result)

# Access raw data
print(f"Fighter 1 win probability: {result['fighter1_win_probability']:.2%}")
print(f"Fighter 2 win probability: {result['fighter2_win_probability']:.2%}")
print(f"Predicted winner: {result['predicted_winner']}")
```

## Auto-Update System

### Manual Update

```bash
# Check if update is needed
python scripts/auto_update.py --check

# Quick update (events and ELO only, ~1 minute)
python scripts/auto_update.py --quick

# Full update (includes model retraining, ~3 minutes)
python scripts/auto_update.py --full
```

### Scheduled Updates

Set up a weekly cron job (Linux/Mac) or Task Scheduler (Windows):

```bash
# Run every Sunday at 2 AM
python scripts/auto_update.py --schedule
```

Windows Task Scheduler:
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Weekly, Sunday, 2:00 AM
4. Action: Start a program
5. Program: `python`
6. Arguments: `scripts/auto_update.py --schedule`
7. Start in: `C:\path\to\MMA-Predictor`

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Tests

```bash
# Test CSV schemas
pytest tests/test_csv_schema.py -v

# Test model training
pytest tests/test_model_training.py -v

# Test scrapers
pytest tests/test_scraper.py -v

# Test dashboard imports
pytest tests/test_streamlit_import.py -v
```

### Test with Coverage

```bash
pytest tests/ -v --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

## Troubleshooting

### Issue: "Model file not found"
**Solution**: Run the training pipeline first
```bash
python train.py
```

### Issue: "No fighter data available"
**Solution**: Run the data pipeline
```bash
python scripts/scrape_rankings.py
python scripts/scrape_fighter_stats.py
python scripts/scrape_events.py
python scripts/build_dataset.py
```

### Issue: "ELO ratings not found"
**Solution**: Run the ELO pipeline
```bash
python elo_pipeline.py
```

### Issue: Dashboard won't start
**Solution**: Check Streamlit installation
```bash
pip install --upgrade streamlit
streamlit run dashboard_app.py
```

### Issue: Import errors
**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt
```

### Issue: Tests failing
**Solution**: Ensure data pipeline has been run
```bash
# Run full pipeline
python scripts/scrape_rankings.py
python scripts/scrape_fighter_stats.py
python scripts/scrape_events.py
python scripts/build_dataset.py
python elo_pipeline.py
python train.py

# Then run tests
pytest tests/ -v
```

## Advanced Usage

### Custom Model Training

Edit `train.py` to customize hyperparameters:

```python
# Decision Tree parameters
max_depth = 10
min_samples_split = 20

# Random Forest parameters
n_estimators = 100

# XGBoost parameters
n_estimators = 100
max_depth = 6
learning_rate = 0.1
```

### Custom ELO Parameters

Edit `elo_system.py`:

```python
# ELO parameters
k_factor = 32              # Rating change per fight
initial_rating = 1500      # Starting rating

# Inactivity decay
decay_threshold_days = 365  # Days before decay starts
decay_rate = 0.02           # Decay per month
```

### Add New Features

Edit `data_processing.py` in the `engineer_features()` function:

```python
def engineer_features(df):
    df = df.copy()
    
    # Your custom features here
    df['custom_feature'] = df['f1_wins'] * df['f2_losses']
    
    return df
```

### Export Predictions

```python
from infer import predict_multiple_fights
import pandas as pd

# List of fights
fights = [
    ("Fighter A", "Fighter B"),
    ("Fighter C", "Fighter D", True),  # Title fight
]

# Get predictions
results = predict_multiple_fights(fights)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('predictions.csv', index=False)
```

## Performance Optimization

### Speed up training
```python
# Use fewer trees
n_estimators = 50  # Instead of 100

# Reduce cross-validation folds
cv = 3  # Instead of 5
```

### Reduce memory usage
```python
# Use float32 instead of float64
X_train = X_train.astype('float32')
```

### Cache management
```bash
# Clear cache to force fresh scraping
rm -rf cache/*

# View cache size
du -sh cache/
```

## Data Management

### Backup your data
```bash
# Backup data directory
cp -r data/ data_backup_$(date +%Y%m%d)/

# Backup model
cp -r artifacts/ artifacts_backup_$(date +%Y%m%d)/
```

### Reset to fresh state
```bash
# Remove generated data (keeps cache)
rm data/*.csv

# Remove model
rm artifacts/*.pkl artifacts/*.json

# Run full pipeline again
python scripts/scrape_rankings.py
# ... (continue with rest of pipeline)
```

## Best Practices

1. **Run updates weekly** to keep data fresh
2. **Retrain model monthly** for best accuracy
3. **Monitor test accuracy** - should be 55-65%
4. **Backup before updates** to preserve historical data
5. **Check data quality scores** in build_dataset.py output
6. **Review feature importance** after retraining
7. **Validate predictions** against actual fight outcomes

## Support

For issues or questions:
1. Check this guide first
2. Review README.md for detailed information
3. Run tests to diagnose problems: `pytest tests/ -v`
4. Check GitHub Issues for similar problems
5. Open a new issue with error details and system info

---

Happy predicting! 🥊
