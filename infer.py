"""Inference CLI for FightIQ.

This script loads the trained model and associated feature metadata
and computes the probability that fighter A defeats fighter B. It
takes fighter names as command line arguments and looks up their
statistics from the canonical dataset.

Example::

    python -m mma_project.infer --fighter_a "Islam Makhachev" --fighter_b "Charles Oliveira"

"""

from __future__ import annotations

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')

FIGHTERS_CSV = os.path.join(DATA_DIR, 'fighters_top10_men.csv')
ELO_CSV = os.path.join(DATA_DIR, 'elo_ratings.csv')


def load_model() -> tuple[object, list[str]]:
    model = joblib.load(os.path.join(ARTIFACTS_DIR, 'model.pkl'))
    with open(os.path.join(ARTIFACTS_DIR, 'feature_names.json')) as f:
        feature_names = json.load(f)
    return model, feature_names


def load_fighters_data() -> pd.DataFrame:
    df = pd.read_csv(FIGHTERS_CSV)
    elos = pd.read_csv(ELO_CSV)
    df = df.merge(elos, on='fighter_id', how='left')
    df['p4p_rank'] = pd.to_numeric(df['p4p_rank'], errors='coerce').fillna(100)
    df['elo'] = pd.to_numeric(df['elo'], errors='coerce').fillna(1500.0)
    return df


def get_fighter_row(df: pd.DataFrame, name: str) -> pd.Series:
    """Return the fighter row matching the given name (case-insensitive)."""
    slug = name.lower().replace("'", '').replace(' ', '-')
    row = df[df['fighter_id'] == slug]
    if row.empty:
        raise ValueError(f'Fighter "{name}" not found in dataset')
    return row.iloc[0]


def compute_features(a_row: pd.Series, b_row: pd.Series) -> np.ndarray:
    """Compute feature vector [rank_diff, p4p_diff, elo_diff]."""
    rank_diff = a_row['rank'] - b_row['rank']
    p4p_diff = a_row['p4p_rank'] - b_row['p4p_rank']
    elo_diff = a_row['elo'] - b_row['elo']
    return np.array([[rank_diff, p4p_diff, elo_diff]])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Predict outcome probability for a matchup between two fighters')
    parser.add_argument('--fighter_a', required=True, help='Name of fighter A')
    parser.add_argument('--fighter_b', required=True, help='Name of fighter B')
    args = parser.parse_args(argv)
    model, feature_names = load_model()
    df = load_fighters_data()
    a_row = get_fighter_row(df, args.fighter_a)
    b_row = get_fighter_row(df, args.fighter_b)
    X = compute_features(a_row, b_row)
    proba = model.predict_proba(X)[0, 1]
    print(f"Probability that {args.fighter_a} defeats {args.fighter_b}: {proba:.3f}")


if __name__ == '__main__':
    main()