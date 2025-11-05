"""Custom ELO rating system implementation.

This module defines a class for computing and updating ELO ratings for
MMA fighters. The ELO algorithm assigns a numeric rating to each fighter
and updates their ratings after each fight based on the expected vs
actual outcome. Additional features include:

* Title fight bonus: important fights carry additional weight.
* Inactivity decay: fighters who have not competed for a while have
  their ratings slightly decayed to reflect potential ring rust.

Usage
-----
>>> from mma_project.elo_system import EloRatingSystem
>>> system = EloRatingSystem(initial_rating=1500, k=32)
>>> new_r_a, new_r_b = system.update_ratings(1500, 1500, outcome=1)
    # fighter A beats fighter B
>>> print(new_r_a, new_r_b)

When computing ratings for an entire dataset of fights, call
``update_ratings`` sequentially in date order and store the ratings per
fighter after each fight. See the notebook or dashboard for examples.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Tuple, Dict


class EloRatingSystem:
    """A flexible ELO rating system for head‑to‑head sports.

    Parameters
    ----------
    initial_rating : float, default=1500.0
        Starting rating assigned to fighters who have not competed before.
    k : float, default=32.0
        K‑factor controlling the maximum change per fight. Higher values
        cause ratings to adjust more quickly.
    title_bonus : float, default=16.0
        Additional K‑factor applied to title fights, making them more
        impactful on ratings.
    decay_rate : float, default=0.0
        Daily decay applied to ratings for inactive fighters. A value of
        0.001 implies ratings drop by 0.1% per day without a fight.
    inactivity_threshold : int, default=365
        Number of days after which inactivity decay begins applying.
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k: float = 32.0,
        title_bonus: float = 16.0,
        decay_rate: float = 0.0,
        inactivity_threshold: int = 365
    ) -> None:
        self.initial_rating = initial_rating
        self.k = k
        self.title_bonus = title_bonus
        self.decay_rate = decay_rate
        self.inactivity_threshold = inactivity_threshold
        # Track last active date for each fighter to compute decay
        self.last_active: Dict[int, datetime] = {}

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Return the expected probability of fighter A defeating fighter B."""
        exp = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return exp

    def apply_inactivity_decay(self, rating: float, days_inactive: int) -> float:
        """Apply exponential decay to a fighter's rating based on inactivity.

        The rating decays proportionally to the number of days beyond the
        inactivity threshold. If ``days_inactive`` is below the threshold
        no decay is applied.

        Parameters
        ----------
        rating : float
            Current rating before decay.
        days_inactive : int
            Days since the fighter last fought.

        Returns
        -------
        float
            Decayed rating.
        """
        if days_inactive <= self.inactivity_threshold or self.decay_rate <= 0:
            return rating
        # Days beyond threshold
        extra_days = days_inactive - self.inactivity_threshold
        # Exponential decay: rating *= (1 - decay_rate)**extra_days
        decayed_rating = rating * ((1 - self.decay_rate) ** extra_days)
        return decayed_rating

    def update_ratings(
        self,
        rating_a: float,
        rating_b: float,
        outcome: int,
        title_fight: bool = False
    ) -> Tuple[float, float]:
        """Update ratings for a single fight and return new ratings.

        Parameters
        ----------
        rating_a : float
            Pre‑fight rating for fighter A.
        rating_b : float
            Pre‑fight rating for fighter B.
        outcome : int
            1 if fighter A wins, 0 if fighter B wins. Draws are not
            supported here but could be incorporated by using 0.5.
        title_fight : bool, default=False
            If true, apply the title bonus to the K‑factor.

        Returns
        -------
        (new_rating_a, new_rating_b) : Tuple[float, float]
            Updated ratings for fighters A and B.
        """
        expected_a = self.expected_score(rating_a, rating_b)
        # Determine K‑factor for this fight
        k_factor = self.k + (self.title_bonus if title_fight else 0)
        # Score is 1 for a win, 0 for a loss
        score_a = outcome
        score_b = 1 - outcome
        new_rating_a = rating_a + k_factor * (score_a - expected_a)
        new_rating_b = rating_b + k_factor * (score_b - (1 - expected_a))
        return new_rating_a, new_rating_b

    def compute_elo_history(
        self,
        fights_df: 'pd.DataFrame',
        fighters_df: 'pd.DataFrame',
        date_col: str = 'date'
    ) -> Dict[int, float]:
        """Compute and return final ELO ratings for all fighters.

        This method iterates through fights in chronological order,
        applying inactivity decay where appropriate and updating ratings
        based on fight outcomes. It keeps track of each fighter's last
        fight date to compute inactivity.

        Parameters
        ----------
        fights_df : pd.DataFrame
            DataFrame of fights. Must include columns ``fighter_a_id``,
            ``fighter_b_id``, ``winner_id``, ``is_title_fight`` and a
            date column specified by ``date_col``.
        fighters_df : pd.DataFrame
            DataFrame containing fighter IDs. Used only to initialize
            ratings for fighters who never appear in the fights list.
        date_col : str, default='date'
            Column in ``fights_df`` representing the fight date.

        Returns
        -------
        Dict[int, float]
            Final ELO rating for each fighter after processing all fights.
        """
        # Lazy import to avoid heavy dependency unless needed
        import pandas as pd

        ratings: Dict[int, float] = {}
        # Initialize ratings for all fighters
        for fid in fighters_df['fighter_id']:
            ratings[fid] = self.initial_rating
            self.last_active[fid] = None

        # Sort fights chronologically
        fights_sorted = fights_df.sort_values(date_col)

        for _, fight in fights_sorted.iterrows():
            a_id = int(fight['fighter_a_id'])
            b_id = int(fight['fighter_b_id'])
            winner_id = int(fight['winner_id'])
            is_title = bool(fight.get('is_title_fight', False))
            date = fight[date_col]

            # Apply inactivity decay
            for fid in (a_id, b_id):
                last_date = self.last_active.get(fid)
                if last_date is not None:
                    days_inactive = (date - last_date).days
                    ratings[fid] = self.apply_inactivity_decay(ratings[fid], days_inactive)
                # update last active date to current fight date
                self.last_active[fid] = date

            # Current ratings after decay
            ra = ratings.get(a_id, self.initial_rating)
            rb = ratings.get(b_id, self.initial_rating)

            outcome = 1 if winner_id == a_id else 0
            new_ra, new_rb = self.update_ratings(ra, rb, outcome, title_fight=is_title)
            ratings[a_id], ratings[b_id] = new_ra, new_rb

        return ratings
