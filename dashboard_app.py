"""FightIQ dashboard built on real UFC rankings and ELO data.

This Streamlit application loads fighter information from the canonical
dataset (``data/fighters_top10_men.csv``) merged with ELO ratings
(``data/elo_ratings.csv``) and displays rankings, fighter profiles, a
head‑to‑head fight simulator and feature importance for a decision‑tree
model trained on simple features.

Run this app with:

    streamlit run mma_project/dashboard_app.py

The layout and styling follow a dark, terminal‑like theme inspired by
the Poly‑Trader project.
"""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Tuple, Dict

# Import utilities from training and inference modules. These functions
# reside in separate files to keep concerns modular.
from mma_project import train as train_mod  # type: ignore
from mma_project import infer as infer_mod  # type: ignore

@st.cache_data(show_spinner=True, suppress_st_warning=True)
def load_fightiq_data() -> Tuple[pd.DataFrame, Dict[str, float], object, list[str]]:
    """Load fighter data, compute model performance and load the model.

    This helper reads the canonical fighters dataset and ELO ratings,
    evaluates available models on the training data and returns the
    dataset along with metrics for the best model. The trained model
    and its feature names are loaded from the ``artifacts`` folder. If
    artifacts are missing, a fallback dummy model is created which will
    predict a uniform probability for either fighter.

    Returns
    -------
    fighters : pd.DataFrame
        Dataframe containing fighters with columns ``fighter_id``,
        ``name``, ``division``, ``rank``, ``p4p_rank`` and ``elo``.
    metrics : dict
        Performance metrics (accuracy, F1, ROC‑AUC) for the best model.
    model : object
        Trained classification model implementing ``predict_proba``.
    feature_names : list of str
        Names of features used by the model.
    """
    # Load fighters with ELO ratings via inference utility
    fighters = infer_mod.load_fighters_data()
    # Compute training metrics using the training module
    df_train = train_mod.load_data()
    X, y = train_mod.build_pairwise_dataset(df_train)
    metrics_all = train_mod.train_and_evaluate(X, y, random_state=42)
    best_name = train_mod.select_best_model(metrics_all)
    metrics = metrics_all[best_name]
    # Attempt to load trained model and feature names
    try:
        model, feature_names = infer_mod.load_model()
    except FileNotFoundError:
        # Provide a fallback dummy model if artifacts are missing
        import numpy as np
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='uniform')
        model.fit(X, y)
        feature_names = list(X.columns)
    return fighters, metrics, model, feature_names




def render_home(metrics: dict, fighters: pd.DataFrame) -> None:
    """Display overview page with dataset summaries and model metrics."""
    st.header("FightIQ Prediction Dashboard")
    st.markdown(
        """
        Welcome to **FightIQ**! This dashboard uses a decision tree–based
        model trained on real UFC rankings and ELO ratings to forecast
        fight outcomes. Explore fighter profiles, simulate matchups, view
        rankings by division and inspect how the model makes decisions.
        """
    )

    st.subheader("Model Performance (3‑fold CV)")
    perf_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'ROC‑AUC'],
        'Score': [metrics['accuracy'], metrics['f1'], metrics['roc_auc']]
    })
    st.table(perf_df.style.format({'Score': '{:.3f}'}))

    st.subheader("Dataset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Fighters", fighters.shape[0])
        st.metric("Number of Divisions", fighters['division'].nunique())
    with col2:
        # Show count of fighters in P4P list
        p4p_count = fighters[fighters['p4p_rank'] < 100].shape[0]
        st.metric("Fighters in P4P Top‑10", p4p_count)
        # Average ELO rating
        st.metric("Average ELO", f"{fighters['elo'].mean():.1f}")


def render_fighter_profiles(fighters: pd.DataFrame) -> None:
    """Render page showing individual fighter profiles.

    The fighters dataframe is expected to contain columns ``fighter_id``,
    ``name``, ``division``, ``rank``, ``p4p_rank``, ``country``, ``camp``
    and ``elo``. Additional columns will be displayed automatically.
    """
    st.header("Fighter Profiles")
    # Map slug to display name with division and rank
    fighter_map = {row['fighter_id']: f"{row['name']} ({row['division']} # {int(row['rank'])})" for _, row in fighters.iterrows()}
    selected_id = st.selectbox("Select a fighter", options=list(fighter_map.keys()), format_func=lambda x: fighter_map[x])

    fighter = fighters.loc[fighters['fighter_id'] == selected_id].iloc[0]
    # Display basic details
    st.subheader(fighter['name'])
    info_cols = st.columns(3)
    info_cols[0].metric("Division", fighter['division'])
    info_cols[1].metric("Rank", int(fighter['rank']))
    if pd.notnull(fighter.get('p4p_rank')) and fighter['p4p_rank'] < 100:
        info_cols[2].metric("P4P Rank", int(fighter['p4p_rank']))
    else:
        info_cols[2].metric("P4P Rank", "–")
    # Additional metadata
    st.write(f"**Country:** {fighter.get('country', 'N/A')}")
    st.write(f"**Camp:** {fighter.get('camp', 'N/A')}")
    # Show ELO rating
    st.write(f"**Current ELO Rating:** {fighter.get('elo', 1500):.1f}")
    # Show numeric attributes available
    numeric_cols = [col for col in fighters.columns if col not in {'fighter_id', 'name', 'division', 'rank', 'p4p_rank', 'country', 'camp', 'elo'}]
    if numeric_cols:
        numeric_df = fighter[numeric_cols].to_frame().reset_index()
        numeric_df.columns = ['Attribute', 'Value']
        st.dataframe(numeric_df)


def render_fight_simulator(
    fighters: pd.DataFrame,
    model: object,
    feature_names: list[str]
) -> None:
    """Provide an interface to compare two fighters and predict outcomes.

    The simulator computes simple difference features (rank, p4p rank,
    ELO) between two fighters and feeds them to the loaded model. If
    the model lacks ``predict_proba``, a warning is displayed.
    """
    st.header("Fight Simulator")
    # Build mapping for selection: show name and division
    fighter_map = {row['fighter_id']: f"{row['name']} ({row['division']} # {int(row['rank'])})" for _, row in fighters.iterrows()}
    fighter_ids = list(fighter_map.keys())
    if not fighter_ids:
        st.write("No fighter data available.")
        return
    fighter_a_id = st.selectbox("Select Fighter A", options=fighter_ids, format_func=lambda x: fighter_map[x], index=0)
    fighter_b_id = st.selectbox("Select Fighter B", options=[fid for fid in fighter_ids if fid != fighter_a_id], format_func=lambda x: fighter_map[x], index=1)

    if fighter_a_id and fighter_b_id:
        a_row = fighters.loc[fighters['fighter_id'] == fighter_a_id].iloc[0]
        b_row = fighters.loc[fighters['fighter_id'] == fighter_b_id].iloc[0]
        # Compute features expected by the model
        # The feature order is defined in feature_names
        diff_map = {
            'rank_diff': a_row['rank'] - b_row['rank'],
            'p4p_diff': a_row['p4p_rank'] - b_row['p4p_rank'],
            'elo_diff': a_row['elo'] - b_row['elo'],
        }
        X_input = pd.DataFrame([[diff_map.get(name, 0.0) for name in feature_names]], columns=feature_names)
        try:
            proba = model.predict_proba(X_input)[0, 1]
            st.subheader("Predicted Outcome")
            st.write(f"Probability that {a_row['name']} defeats {b_row['name']}: **{proba:.2f}**")
        except AttributeError:
            st.warning("Loaded model does not support probability predictions. Please train the model via 'python -m mma_project.train'.")
        # Display side‑by‑side comparison of basic attributes
        st.subheader("Fighter Comparison")
        comp_df = pd.DataFrame({
            'Attribute': ['Rank', 'P4P Rank', 'ELO'],
            a_row['name']: [int(a_row['rank']), int(a_row['p4p_rank']), f"{a_row['elo']:.1f}"],
            b_row['name']: [int(b_row['rank']), int(b_row['p4p_rank']), f"{b_row['elo']:.1f}"]
        })
        st.table(comp_df)


def render_rankings(fighters: pd.DataFrame) -> None:
    """Display current rankings per division and pound‑for‑pound list."""
    st.header("Rankings & ELO")
    # Select division or P4P
    divisions = sorted(fighters['division'].unique().tolist())
    divisions.append('Pound‑for‑Pound')
    selected_div = st.selectbox("Select division", options=divisions)
    if selected_div == 'Pound‑for‑Pound':
        # Filter fighters with p4p_rank < 100 and sort ascending by p4p_rank
        p4p_df = fighters[fighters['p4p_rank'] < 100].sort_values(['p4p_rank'])
        display_cols = ['p4p_rank', 'name', 'division', 'elo']
        st.table(p4p_df[display_cols].reset_index(drop=True).rename(columns={'p4p_rank': 'Rank'}))
    else:
        div_df = fighters[fighters['division'] == selected_div].sort_values('rank')
        display_cols = ['rank', 'name', 'elo']
        st.table(div_df[display_cols].reset_index(drop=True).rename(columns={'rank': 'Rank'}))


def render_feature_importance(model: object, feature_names: list[str]) -> None:
    """Display a bar chart of feature importances for tree‑based models."""
    st.header("Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        importance_df = importance_df.sort_values('importance', ascending=True)
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Feature Importances')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("The selected model does not provide feature importances.")


def main() -> None:
    """Entry point for the Streamlit dashboard.

    This function configures the page, loads data and model artifacts,
    and dispatches rendering of pages based on user selection.
    """
    st.set_page_config(page_title="FightIQ Dashboard", layout="wide")
    fighters, metrics, model, feature_names = load_fightiq_data()
    # Sidebar navigation
    pages = {
        'Home': render_home,
        'Fighter Profiles': render_fighter_profiles,
        'Fight Simulator': render_fight_simulator,
        'Rankings': render_rankings,
        'Feature Importance': render_feature_importance,
    }
    page = st.sidebar.selectbox("Navigate", list(pages.keys()))
    if page == 'Home':
        render_home(metrics, fighters)
    elif page == 'Fighter Profiles':
        render_fighter_profiles(fighters)
    elif page == 'Fight Simulator':
        render_fight_simulator(fighters, model, feature_names)
    elif page == 'Rankings':
        render_rankings(fighters)
    elif page == 'Feature Importance':
        render_feature_importance(model, feature_names)


if __name__ == '__main__':
    main()
