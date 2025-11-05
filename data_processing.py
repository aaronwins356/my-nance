"""Data processing and simulation utilities for MMA fight prediction.

This module contains functions to simulate synthetic MMA fighter data,
generate fight matchups, and transform raw data into a machine learning
friendly format. If you have access to a real dataset you can adapt
these functions accordingly. Otherwise the provided functions create
realistic‑looking data with reproducible randomness.

Functions
---------
simulate_fighters(num_fighters: int, random_state: int) -> pd.DataFrame
    Generate a pool of fighters with physical, historical and stylistic
    attributes.

simulate_fights(fighters: pd.DataFrame, num_fights: int, start_date: datetime,
                end_date: datetime, random_state: int) -> pd.DataFrame
    Create a list of fights between randomly selected fighters.

prepare_training_data(fights: pd.DataFrame, fighters: pd.DataFrame,
                      elo_ratings: dict) -> Tuple[pd.DataFrame, pd.Series]
    Construct feature matrix X and target vector y for model training.

encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder]
    Perform one‑hot encoding on categorical columns and return the
    transformed feature matrix and the fitted encoder.

scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]
    Standardize numerical features for use with certain models.

Note
----
The synthetic data generator is intentionally simple; feel free to adjust
distributions or add additional attributes as needed. All random operations
use NumPy's random module to ensure reproducibility via a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, Dict


def simulate_fighters(num_fighters: int = 50, random_state: int = 42) -> pd.DataFrame:
    """Simulate a roster of fighters with various attributes.

    Parameters
    ----------
    num_fighters : int, default=50
        Number of fighters to simulate.
    random_state : int, default=42
        Seed for reproducible randomness.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a fighter and columns
        correspond to physical attributes, fight history and stylistic
        information. An internal integer ``fighter_id`` uniquely
        identifies each fighter.

    Example
    -------
    >>> fighters = simulate_fighters(10, random_state=0)
    >>> fighters.head()
       fighter_id  height_cm  weight_kg  reach_cm   age  ...
    0          0        178         77        191  28.5  ...
    """
    rng = np.random.default_rng(random_state)

    # Physical attributes (approximate realistic ranges)
    heights = rng.normal(loc=180, scale=10, size=num_fighters).clip(155, 205)
    weights = rng.normal(loc=80, scale=12, size=num_fighters).clip(50, 120)
    reaches = heights + rng.normal(loc=5, scale=5, size=num_fighters)
    ages = rng.normal(loc=30, scale=4, size=num_fighters).clip(20, 45)

    # Fight history
    total_fights = rng.integers(low=5, high=40, size=num_fighters)
    wins = np.array([rng.integers(low=max(1, t // 2), high=t) for t in total_fights])
    losses = total_fights - wins - rng.integers(low=0, high=2, size=num_fighters)  # some draws
    draws = total_fights - wins - losses
    ko_rate = rng.uniform(0.1, 0.7, size=num_fighters)
    sub_rate = rng.uniform(0.05, 0.5, size=num_fighters)
    avg_fight_time = rng.uniform(5, 20, size=num_fighters)

    # Performance stats
    strikes_per_min = rng.uniform(2, 8, size=num_fighters)
    takedown_acc = rng.uniform(0.3, 0.8, size=num_fighters)
    defense = rng.uniform(0.4, 0.9, size=num_fighters)
    control_time = rng.uniform(1, 5, size=num_fighters)

    # Categorical/stylistic attributes
    stances = rng.choice(['orthodox', 'southpaw', 'switch', 'open'], size=num_fighters)
    genders = rng.choice(['male', 'female'], size=num_fighters, p=[0.8, 0.2])
    camps = rng.choice(['ATT', 'AKA', 'JacksonWink', 'Alliance', 'Nova Uniao'], size=num_fighters)
    weight_classes = rng.choice(['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight',
                                'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight'],
                               size=num_fighters)
    archetypes = rng.choice(['striker', 'grappler', 'well-rounded'], p=[0.45, 0.35, 0.2], size=num_fighters)

    fighters = pd.DataFrame({
        'fighter_id': np.arange(num_fighters),
        'height_cm': heights,
        'weight_kg': weights,
        'reach_cm': reaches,
        'age': ages,
        'total_fights': total_fights,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'ko_rate': ko_rate,
        'sub_rate': sub_rate,
        'avg_fight_time': avg_fight_time,
        'strikes_per_min': strikes_per_min,
        'takedown_acc': takedown_acc,
        'defense': defense,
        'control_time': control_time,
        'stance': stances,
        'gender': genders,
        'camp': camps,
        'weight_class': weight_classes,
        'archetype': archetypes,
    })
    return fighters


def simulate_fights(
    fighters: pd.DataFrame,
    num_fights: int = 300,
    start_date: datetime = datetime(2015, 1, 1),
    end_date: datetime = datetime(2025, 1, 1),
    title_prob: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """Simulate a series of fights between fighters.

    Fights are created by randomly pairing fighters. A hidden skill rating
    is assigned to each fighter; the outcome of a fight is sampled from
    a logistic model based on the difference in skill. This ensures that
    the simulated dataset contains learnable patterns.

    Parameters
    ----------
    fighters : pd.DataFrame
        DataFrame produced by :func:`simulate_fighters`. Must contain
        ``fighter_id`` column.
    num_fights : int, default=300
        Number of fights to simulate.
    start_date : datetime, default=datetime(2015,1,1)
        Earliest possible fight date.
    end_date : datetime, default=datetime(2025,1,1)
        Latest possible fight date.
    title_prob : float, default=0.05
        Probability that a given fight is for a title. Title fights
        receive additional ELO weight (handled in the ELO system).
    random_state : int, default=42
        Seed for randomness.

    Returns
    -------
    pd.DataFrame
        Each row corresponds to a fight with the following columns:
        ``date`` (datetime), ``fighter_a_id``, ``fighter_b_id``,
        ``winner_id`` (ID of winner), ``is_title_fight`` (bool), and
        ``location`` (categorical).
    """
    rng = np.random.default_rng(random_state)
    fighter_ids = fighters['fighter_id'].tolist()
    # Hidden skill rating influences outcome probability
    skill = rng.normal(loc=0.0, scale=1.0, size=len(fighter_ids))
    skill_map = dict(zip(fighter_ids, skill))

    # Generate random fight dates uniformly between start and end
    total_seconds = (end_date - start_date).total_seconds()
    fight_times = rng.uniform(0, total_seconds, size=num_fights)
    fight_dates = [start_date + timedelta(seconds=float(s)) for s in fight_times]

    fights = []
    for dt in fight_dates:
        # random pair of distinct fighters
        a, b = rng.choice(fighter_ids, size=2, replace=False)
        # logistic probability of fighter a winning
        rating_diff = skill_map[a] - skill_map[b]
        p_a = 1 / (1 + np.exp(-rating_diff))
        winner = a if rng.random() < p_a else b
        fights.append({
            'date': dt,
            'fighter_a_id': a,
            'fighter_b_id': b,
            'winner_id': winner,
            'is_title_fight': rng.random() < title_prob,
            'location': rng.choice(['USA', 'Brazil', 'UK', 'UAE', 'Japan', 'Canada'])
        })

    fights_df = pd.DataFrame(fights).sort_values('date').reset_index(drop=True)
    return fights_df


def prepare_training_data(
    fights: pd.DataFrame,
    fighters: pd.DataFrame,
    elo_ratings: Dict[int, float]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct a training matrix and target vector from fight and fighter data.

    The idea is to model the probability that ``fighter_a`` wins a given
    fight. We create features representing the differences between the two
    fighters' attributes (fighter A minus fighter B) and include their
    current ELO ratings. Only numeric features are directly differenced;
    categorical features are handled separately during encoding.

    Parameters
    ----------
    fights : pd.DataFrame
        DataFrame of fights generated by :func:`simulate_fights`. Must
        include ``fighter_a_id``, ``fighter_b_id`` and ``winner_id``.
    fighters : pd.DataFrame
        DataFrame of fighters generated by :func:`simulate_fighters`.
    elo_ratings : dict
        Mapping from fighter ID to their latest ELO rating at the
        time of the fight. In practice this can be computed iteratively
        using the ``EloRatingSystem``; here we assume the ratings are
        supplied by the caller.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with numeric and categorical columns. The
        categorical columns still need to be one‑hot encoded using
        :func:`encode_features`.
    y : pd.Series
        Binary target indicating whether fighter A won (1) or not (0).
    """
    # Merge fighter attributes for both sides
    fighters_a = fighters.add_prefix('a_').rename(columns={'a_fighter_id': 'fighter_a_id'})
    fighters_b = fighters.add_prefix('b_').rename(columns={'b_fighter_id': 'fighter_b_id'})
    fights_df = fights.merge(fighters_a, on='fighter_a_id', how='left')
    fights_df = fights_df.merge(fighters_b, on='fighter_b_id', how='left')

    # Add ELO ratings for both fighters
    fights_df['a_elo'] = fights_df['fighter_a_id'].map(elo_ratings)
    fights_df['b_elo'] = fights_df['fighter_b_id'].map(elo_ratings)
    # If ratings missing for some fighters (e.g., first fights), set to default 1500
    fights_df[['a_elo', 'b_elo']] = fights_df[['a_elo', 'b_elo']].fillna(1500)

    # Target: 1 if fighter A wins
    y = (fights_df['winner_id'] == fights_df['fighter_a_id']).astype(int)

    # Compute numeric differences (fighter A minus fighter B)
    numeric_cols = [
        'height_cm', 'weight_kg', 'reach_cm', 'age', 'total_fights',
        'wins', 'losses', 'draws', 'ko_rate', 'sub_rate', 'avg_fight_time',
        'strikes_per_min', 'takedown_acc', 'defense', 'control_time', 'elo'
    ]
    # Add ELO to differences by renaming columns accordingly
    fights_df = fights_df.rename(columns={'a_elo': 'a_elo_rating', 'b_elo': 'b_elo_rating'})
    fights_df['a_elo'] = fights_df['a_elo_rating']
    fights_df['b_elo'] = fights_df['b_elo_rating']
    # Add combined numeric difference features
    feature_rows = {}
    for col in numeric_cols:
        a_col = 'a_' + col
        b_col = 'b_' + col
        if a_col in fights_df.columns and b_col in fights_df.columns:
            feature_rows[col + '_diff'] = fights_df[a_col] - fights_df[b_col]
    X_numeric = pd.DataFrame(feature_rows)

    # Categorical columns: difference not meaningful; we keep both and encode later
    # Location is associated with the fight rather than a specific fighter, so
    # include it separately instead of prefixing with 'a_' or 'b_'.
    categorical_cols = ['stance', 'gender', 'camp', 'weight_class', 'archetype']
    # Extract fighter A and B categorical attributes
    cat_features = fights_df[['a_' + c for c in categorical_cols] + ['b_' + c for c in categorical_cols]].reset_index(drop=True)
    # Add fight location as its own categorical feature
    location_col = fights_df[['location']].reset_index(drop=True)

    X = pd.concat([X_numeric.reset_index(drop=True), cat_features, location_col], axis=1)
    return X, y


def encode_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """One‑hot encode categorical columns in the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing both numeric and categorical features.

    Returns
    -------
    X_enc : pd.DataFrame
        Feature matrix with categorical variables expanded into binary
        indicator columns. Numeric columns are untouched.
    encoder : OneHotEncoder
        Fitted encoder that can be used to transform new data during
        inference.
    """
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Instantiate the encoder with appropriate argument depending on scikit‑learn version
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        # Older versions use `sparse` instead of `sparse_output`
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    if categorical_cols:
        X_cat = encoder.fit_transform(X[categorical_cols])
        cat_cols = encoder.get_feature_names_out(categorical_cols)
        X_cat_df = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)
        X_numeric_df = X[numeric_cols].reset_index(drop=True)
        X_enc = pd.concat([X_numeric_df, X_cat_df.reset_index(drop=True)], axis=1)
    else:
        X_enc = X.copy()
    return X_enc, encoder


def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize numerical columns of the feature matrix.

    Only numerical columns are scaled; one‑hot encoded categorical columns
    remain unchanged. Many tree‑based models (e.g., random forests) do
    not require feature scaling, but scaling can be useful for models
    such as logistic regression or gradient boosting. The function
    returns the scaled matrix and the fitted scaler.

    Parameters
    ----------
    X : pd.DataFrame
        Matrix with numeric and possibly categorical columns.

    Returns
    -------
    X_scaled : pd.DataFrame
        DataFrame with numeric features standardized to zero mean and
        unit variance. Categorical columns are preserved.
    scaler : StandardScaler
        Fitted scaler instance.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_scaled_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)
    X_non_numeric = X.drop(columns=numeric_cols)
    X_scaled = pd.concat([X_scaled_numeric, X_non_numeric.reset_index(drop=True)], axis=1)
    return X_scaled, scaler
