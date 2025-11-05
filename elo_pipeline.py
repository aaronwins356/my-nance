"""ELO rating pipeline for FightIQ.

This module provides a simple command line utility to compute and update
ELO ratings for the fighters contained in ``data/fighters_top10_men.csv``.
It reads recent fight results from ``data/events_latest.csv`` and applies
the update rules defined in :class:`mma_project.elo_system.EloRatingSystem`.
If no events are present, the ratings remain unchanged (default 1500).

The pipeline produces ``data/elo_ratings.csv`` and optionally an
``data/elo_history.parquet`` file with the full rating history. It is
intended to be run after scraping new events (see ``scripts/scrape_events.py``)
and before training models.

Usage::

    python -m mma_project.elo_pipeline

"""

from __future__ import annotations

import csv
import os
import pandas as pd
from typing import Dict

from .elo_system import EloRatingSystem


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

FIGHTERS_CSV = os.path.join(DATA_DIR, 'fighters_top10_men.csv')
EVENTS_CSV = os.path.join(DATA_DIR, 'events_latest.csv')
ELO_CSV = os.path.join(DATA_DIR, 'elo_ratings.csv')


def load_fighters() -> pd.DataFrame:
    """Load the fighters dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing fighter identifiers. Must include ``fighter_id``.
    """
    return pd.read_csv(FIGHTERS_CSV)


def load_events() -> pd.DataFrame:
    """Load the latest events.

    Returns
    -------
    pd.DataFrame
        DataFrame of events with columns ``fighter_a``, ``fighter_b`` and ``winner``.
        If the file is empty or missing, an empty DataFrame is returned.
    """
    if not os.path.exists(EVENTS_CSV):
        return pd.DataFrame(columns=['event_id','date','fighter_a','fighter_b','winner','method','round','time','is_title_fight','division'])
    events = pd.read_csv(EVENTS_CSV)
    if events.empty:
        return events
    return events


def compute_ratings() -> Dict[str, float]:
    """Compute ELO ratings for all fighters.

    Ratings are initialized to 1500 for each fighter. For each fight in
    ``events_latest.csv``, ratings are updated sequentially by date using
    :class:`mma_project.elo_system.EloRatingSystem`. Events referencing
    fighters outside the dataset are ignored.

    Returns
    -------
    Dict[str, float]
        Mapping from fighter_id to their final rating.
    """
    fighters = load_fighters()
    # Initialize ratings
    ratings: Dict[str, float] = {fid: 1500.0 for fid in fighters['fighter_id']}
    events = load_events()
    # If there are no events, return defaults
    if events.empty:
        return ratings
    # Instantiate rating system
    elo_system = EloRatingSystem(initial_rating=1500, k=32, title_bonus=16, decay_rate=0.001, inactivity_threshold=365)
    # Pre-sort by date if date column exists
    if 'date' in events.columns:
        events = events.sort_values('date')
    for _, row in events.iterrows():
        a_name = row.get('fighter_a')
        b_name = row.get('fighter_b')
        winner_name = row.get('winner')
        # Map names to fighter_id (slug) by matching case-insensitive name in dataset
        def to_id(name: str) -> str | None:
            if not isinstance(name, str):
                return None
            # normalize: lower-case and replace spaces and apostrophes
            slug = name.lower().replace("'", '').replace(' ', '-')
            return slug if slug in ratings else None
        a_id = to_id(a_name)
        b_id = to_id(b_name)
        winner_id = to_id(winner_name)
        if a_id is None or b_id is None or winner_id is None:
            # skip if any fighter not in dataset
            continue
        # current ratings
        ra, rb = ratings.get(a_id, 1500.0), ratings.get(b_id, 1500.0)
        # Determine outcome: 1 if A wins, else 0
        outcome = 1 if winner_id == a_id else 0
        new_ra, new_rb = elo_system.update_ratings(ra, rb, outcome, title_fight=bool(row.get('is_title_fight', False)))
        ratings[a_id] = new_ra
        ratings[b_id] = new_rb
    return ratings


def save_ratings(ratings: Dict[str, float]) -> None:
    """Persist ratings to ``data/elo_ratings.csv``.

    Parameters
    ----------
    ratings : dict
        Mapping from fighter_id to rating.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ELO_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fighter_id','elo'])
        for fid, elo in ratings.items():
            writer.writerow([fid, elo])


def main() -> None:
    ratings = compute_ratings()
    save_ratings(ratings)
    print(f"Computed ratings for {len(ratings)} fighters and saved to {ELO_CSV}")


if __name__ == '__main__':
    main()