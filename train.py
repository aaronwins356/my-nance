"""Training script for FightIQ models.

This module builds a simple feature matrix from the canonical dataset
(`data/fighters_top10_men.csv`) and associated ELO ratings
(`data/elo_ratings.csv`), constructs pairwise training examples and
trains a decision tree classifier (and optional ensembles) to predict
fight outcomes. The trained model and metadata are saved to the
`artifacts/` directory.

Usage::

    python -m mma_project.train

"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
import joblib


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')

FIGHTERS_CSV = os.path.join(DATA_DIR, 'fighters_top10_men.csv')
ELO_CSV = os.path.join(DATA_DIR, 'elo_ratings.csv')

def load_data() -> pd.DataFrame:
    """Load fighters dataset and merge ELO ratings.

    Returns
    -------
    pd.DataFrame
        DataFrame with fighter_id, division, rank, p4p_rank and elo.
    """
    fighters = pd.read_csv(FIGHTERS_CSV)
    elos = pd.read_csv(ELO_CSV)
    df = fighters.merge(elos, on='fighter_id', how='left')
    # Convert p4p_rank to numeric, fill missing with a large number (for fighters not in P4P list)
    df['p4p_rank'] = pd.to_numeric(df['p4p_rank'], errors='coerce')
    df['p4p_rank'] = df['p4p_rank'].fillna(100)
    return df


def build_pairwise_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construct pairwise feature matrix and labels.

    For each division, generate synthetic matchups: champion vs each ranked fighter
    (champion assumed to win) and ranked fighter i vs fighter j (lower rank wins).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fighters with columns ``division``, ``rank``,
        ``p4p_rank`` and ``elo``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns [rank_diff, p4p_diff, elo_diff].
    y : pd.Series
        Binary labels where 1 indicates fighter A wins, 0 indicates fighter B wins.
    """
    rows = []
    labels = []
    # iterate by division
    for division, group in df.groupby('division'):
        # identify champion row (rank 0)
        champ_row = group[group['rank'] == 0]
        if not champ_row.empty:
            champ = champ_row.iloc[0]
            others = group[group['rank'] > 0]
            for _, b in others.iterrows():
                # champion vs other
                a = champ
                # compute features as difference A - B
                rank_diff = a['rank'] - b['rank']
                p4p_diff = a['p4p_rank'] - b['p4p_rank']
                elo_diff = a['elo'] - b['elo']
                rows.append({'rank_diff': rank_diff, 'p4p_diff': p4p_diff, 'elo_diff': elo_diff})
                labels.append(1)  # champion wins
                # reverse matchup (other vs champ)
                rows.append({'rank_diff': -rank_diff, 'p4p_diff': -p4p_diff, 'elo_diff': -elo_diff})
                labels.append(0)  # champion loses (should rarely happen)
        # matchups among ranked fighters
        ranked = group[group['rank'] > 0].sort_values('rank')
        for i, a in ranked.iterrows():
            for j, b in ranked.iterrows():
                if a['rank'] < b['rank']:
                    # higher ranked (lower number) vs lower ranked
                    rank_diff = a['rank'] - b['rank']
                    p4p_diff = a['p4p_rank'] - b['p4p_rank']
                    elo_diff = a['elo'] - b['elo']
                    rows.append({'rank_diff': rank_diff, 'p4p_diff': p4p_diff, 'elo_diff': elo_diff})
                    labels.append(1)  # a wins
                    # reverse matchup: lower ranked fighter vs higher ranked fighter
                    rows.append({'rank_diff': -rank_diff, 'p4p_diff': -p4p_diff, 'elo_diff': -elo_diff})
                    labels.append(0)
    X = pd.DataFrame(rows)
    y = pd.Series(labels)
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict:
    """Train multiple models and return their performance metrics.

    Returns a dictionary keyed by model name with metrics dict as value. Metrics
    include accuracy, f1 and ROC-AUC.
    """
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }
    models = {
        'decision_tree': DecisionTreeClassifier(random_state=random_state, class_weight='balanced'),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced'),
        'gradient_boosting': GradientBoostingClassifier(random_state=random_state)
    }
    results = {}
    for name, model in models.items():
        cv = cross_validate(model, X, y, cv=3, scoring=scoring, n_jobs=-1)
        metrics = {m: np.mean(cv[f'test_{m}']) for m in scoring}
        results[name] = metrics
    return results


def select_best_model(metrics: dict) -> str:
    """Select the best model based on ROC-AUC, then F1, then accuracy.

    Parameters
    ----------
    metrics : dict
        Mapping from model name to metrics dict.

    Returns
    -------
    str
        Name of the best model.
    """
    best_name: str | None = None
    best_key: tuple[float, float, float] | None = None
    for name, m in metrics.items():
        key = (m['roc_auc'], m['f1'], m['accuracy'])
        if best_key is None or key > best_key:
            best_name = name
            best_key = key
    # Fallback: return first model name if metrics dict is empty or best not assigned
    if best_name is None and metrics:
        best_name = next(iter(metrics.keys()))
    return best_name or ''


def save_artifacts(model, feature_names: list[str]) -> None:
    """Persist the trained model and metadata to ``artifacts/``.

    Parameters
    ----------
    model : object
        Fitted scikit-learn model.
    feature_names : list of str
        Names of features used during training.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'model.pkl'))
    with open(os.path.join(ARTIFACTS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)


def main() -> None:
    df = load_data()
    X, y = build_pairwise_dataset(df)
    metrics = train_and_evaluate(X, y, random_state=42)
    print('Cross-validated metrics:')
    for name, m in metrics.items():
        print(f"  {name}: accuracy={m['accuracy']:.3f}, f1={m['f1']:.3f}, roc_auc={m['roc_auc']:.3f}")
    best_name = select_best_model(metrics)
    print(f'Best model: {best_name}')
    # Fit best model on full data
    models = {
        'decision_tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }
    model = models[best_name]
    model.fit(X, y)
    save_artifacts(model, list(X.columns))
    print(f'Saved trained {best_name} model and feature names to {ARTIFACTS_DIR}')


if __name__ == '__main__':
    main()