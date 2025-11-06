"""
FightIQ Dashboard
Interactive Streamlit dashboard for MMA fight predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# Import project modules
from infer import (
    load_model, load_fighter_stats, load_elo_ratings, 
    predict_fight, explain_prediction_simple, load_fighter_styles
)
from data_processing import load_feature_names
from fight_history import load_fight_history, get_fighter_history, format_fight_duration

# Page config
st.set_page_config(
    page_title="FightIQ - MMA Prediction System",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #262730;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .winner {
        color: #00CC00;
        font-weight: bold;
    }
    .fighter-profile {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_resource
def load_model_cached():
    """Load model with caching"""
    try:
        return load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_fighter_stats_cached():
    """Load fighter stats with caching"""
    try:
        return load_fighter_stats()
    except Exception as e:
        st.error(f"Error loading fighter stats: {e}")
        return pd.DataFrame()

@st.cache_data
def load_elo_ratings_cached():
    """Load ELO ratings with caching"""
    try:
        return load_elo_ratings()
    except Exception as e:
        st.warning(f"Error loading ELO ratings: {e}")
        return {}

@st.cache_data
def load_fighter_styles_cached():
    """Load fighter styles with caching"""
    try:
        return load_fighter_styles()
    except Exception as e:
        st.warning(f"Error loading fighter styles: {e}")
        return {}

@st.cache_data
def load_fight_history_cached():
    """Load fight history with caching"""
    try:
        return load_fight_history()
    except Exception as e:
        st.warning(f"Error loading fight history: {e}")
        return pd.DataFrame()

@st.cache_data
def load_model_metadata_cached():
    """Load model metadata with caching"""
    try:
        artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
        metadata_file = os.path.join(artifacts_dir, 'model_metadata.json')
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Error loading metadata: {e}")
    
    return None

def home_page():
    """Home page with system overview"""
    st.markdown('<div class="main-header">🥊 FightIQ</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced MMA Fight Prediction System</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System overview
    col1, col2, col3, col4 = st.columns(4)
    
    fighter_stats = load_fighter_stats_cached()
    elo_ratings = load_elo_ratings_cached()
    metadata = load_model_metadata_cached()
    
    with col1:
        st.metric("Fighters Tracked", len(fighter_stats) if not fighter_stats.empty else 0)
    
    with col2:
        st.metric("Active Fighters", len(elo_ratings))
    
    with col3:
        if metadata and 'metrics' in metadata:
            accuracy = metadata['metrics'].get('test_accuracy', 0) * 100
            st.metric("Model Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("Model Accuracy", "N/A")
    
    with col4:
        if metadata:
            model_name = metadata.get('model_name', 'Unknown')
            st.metric("Model Type", model_name)
        else:
            st.metric("Model Type", "N/A")
    
    st.markdown("---")
    
    # Features
    st.markdown('<div class="sub-header">✨ Features</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Fight Predictions**
        - AI-powered win probability predictions
        - Calibrated probability estimates
        - ELO rating integration
        
        **📊 Fighter Analytics**
        - Comprehensive fighter profiles
        - Performance statistics
        - Historical fight records
        """)
    
    with col2:
        st.markdown("""
        **🏆 Rankings**
        - Live UFC rankings
        - ELO-based rankings
        - Weight class breakdowns
        
        **🔍 Explainability**
        - Feature importance analysis
        - Key factors in predictions
        - Transparent decision-making
        """)
    
    st.markdown("---")
    
    # Model info
    if metadata:
        st.markdown('<div class="sub-header">🤖 Model Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {metadata.get('model_name', 'Unknown')}")
            st.write(f"**Training Date:** {metadata.get('trained_at', 'Unknown')[:10]}")
            st.write(f"**Number of Features:** {metadata.get('num_features', 'Unknown')}")
        
        with col2:
            if 'metrics' in metadata:
                metrics = metadata['metrics']
                st.write(f"**Test Accuracy:** {metrics.get('test_accuracy', 0):.2%}")
                st.write(f"**Test AUC:** {metrics.get('test_auc', 0):.4f}")
                st.write(f"**Test F1 Score:** {metrics.get('test_f1', 0):.4f}")

def fighter_profiles_page():
    """Fighter profiles page with enhanced fight history"""
    st.markdown('<div class="main-header">👤 Fighter Profiles</div>', unsafe_allow_html=True)
    
    fighter_stats = load_fighter_stats_cached()
    elo_ratings = load_elo_ratings_cached()
    fighter_styles = load_fighter_styles_cached()
    fight_history = load_fight_history_cached()
    
    if fighter_stats.empty:
        st.error("No fighter data available. Please run the data pipeline first.")
        return
    
    # Fighter selection
    fighter_names = sorted(fighter_stats['name'].unique())
    selected_fighter = st.selectbox("Select a Fighter", fighter_names)
    
    if selected_fighter:
        fighter_data = fighter_stats[fighter_stats['name'] == selected_fighter].iloc[0]
        
        st.markdown("---")
        
        # Fighter header
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f'<div class="sub-header">{selected_fighter}</div>', unsafe_allow_html=True)
            st.write(f"**Weight Class:** {fighter_data.get('weight_class', 'N/A')}")
            st.write(f"**Rank:** {fighter_data.get('rank', 'N/A')}")
        
        with col2:
            record = f"{fighter_data['wins']}-{fighter_data['losses']}-{fighter_data['draws']}"
            st.metric("Record", record)
        
        with col3:
            elo_rating = elo_ratings.get(selected_fighter, 1500)
            st.metric("ELO Rating", f"{elo_rating:.0f}")
        
        with col4:
            # Display fighting style
            fighting_style = fighter_styles.get(selected_fighter, 'Unknown')
            st.metric("Fighting Style", fighting_style)
        
        st.markdown("---")
        
        # Stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Physical Attributes**")
            st.write(f"Height: {fighter_data['height']} inches")
            st.write(f"Reach: {fighter_data['reach']} inches")
            st.write(f"Age: {fighter_data['age']} years")
            
            st.markdown("**Fight Record**")
            st.write(f"Wins: {fighter_data['wins']}")
            st.write(f"Losses: {fighter_data['losses']}")
            st.write(f"Win Rate: {fighter_data['wins'] / max(fighter_data['wins'] + fighter_data['losses'], 1):.1%}")
        
        with col2:
            st.markdown("**Win Methods**")
            st.write(f"KO/TKO Wins: {fighter_data['wins_by_ko']}")
            st.write(f"Submission Wins: {fighter_data['wins_by_submission']}")
            st.write(f"Decision Wins: {fighter_data['wins_by_decision']}")
            
            st.markdown("**Performance Metrics**")
            st.write(f"Sig. Strikes/Min: {fighter_data['sig_strikes_per_min']:.2f}")
            st.write(f"Takedown Avg: {fighter_data['takedown_avg_per_15min']:.2f}")
            st.write(f"Striking Accuracy: {fighter_data['striking_accuracy_pct']:.0f}%")
        
        # Fighting Style Description
        st.markdown("---")
        st.markdown("**Fighting Style Analysis** 🥋")
        
        style_descriptions = {
            'Boxer': '🥊 Striker specializing in hands, head movement, and precise punching',
            'Kickboxer': '🦵 Striker using full-body striking with punches and kicks',
            'Muay Thai': '🇹🇭 Striker emphasizing elbows, knees, and clinch work',
            'Karate': '🥋 Striker with movement, timing, and unorthodox techniques',
            'Taekwondo': '🥋 Striker focusing on dynamic kicks and agility',
            'Wrestler': '🤼 Grappler controlling with takedowns and ground dominance',
            'BJJ': '🟦 Grappler hunting for submissions from various positions',
            'Judoka': '🥋 Grappler using throws, trips, and positional control',
            'Sambo': '🇷🇺 Grappler combining throws, leg locks, and ground control',
            'Wrestle-Boxer': '🥊🤼 Hybrid blending wrestling control with boxing strikes',
            'Striker-Grappler': '⚔️ Hybrid with balanced striking and grappling skills',
            'All-Rounder': '🌟 Hybrid with elite-level skills in all areas'
        }
        
        style_desc = style_descriptions.get(fighting_style, 'Versatile mixed martial artist')
        st.info(style_desc)
        
        # Visualization
        st.markdown("---")
        st.markdown("**Performance Radar Chart**")
        
        # Create radar chart
        categories = ['Striking', 'Defense', 'Takedown', 'Submission', 'Accuracy']
        values = [
            fighter_data['sig_strikes_per_min'] / 7 * 100,  # Normalize to 0-100
            fighter_data['striking_defense_pct'],
            fighter_data['takedown_avg_per_15min'] / 5 * 100,
            fighter_data['submission_avg_per_15min'] / 2 * 100,
            fighter_data['striking_accuracy_pct']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=selected_fighter
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Complete Fight History Section
        st.markdown("---")
        st.markdown("**Complete Fight History** 📋")
        
        # Get fighter's complete history
        fighter_history = get_fighter_history(selected_fighter, fight_history)
        
        if not fighter_history.empty:
            # Format the display data
            display_history = fighter_history.copy()
            
            # Format date
            if 'event_date' in display_history.columns:
                display_history['Date'] = pd.to_datetime(display_history['event_date']).dt.strftime('%Y-%m-%d')
            else:
                display_history['Date'] = 'N/A'
            
            # Create display columns
            display_columns = {
                'Date': display_history.get('Date', 'N/A'),
                'Opponent': display_history.get('opponent_name', 'N/A'),
                'Event': display_history.get('event_name', 'N/A'),
                'Result': display_history.get('result', 'N/A'),
                'Method': display_history.get('win_method', 'N/A'),
                'Round': display_history.get('round_number', 'N/A'),
                'Time': display_history['fight_duration_seconds'].apply(format_fight_duration) if 'fight_duration_seconds' in display_history.columns else 'N/A',
                'Sig. Strikes': display_history.get('sig_strikes_landed', 'N/A').astype(str) + ' / ' + display_history.get('sig_strikes_received', 'N/A').astype(str) if 'sig_strikes_landed' in display_history.columns else 'N/A',
                'Finish Technique': display_history.get('fight_ending_technique', 'N/A')
            }
            
            display_df = pd.DataFrame(display_columns)
            
            # Apply styling based on result
            def highlight_result(row):
                if row['Result'] == 'Win':
                    return ['background-color: #d4edda'] * len(row)
                elif row['Result'] == 'Loss':
                    return ['background-color: #f8d7da'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(
                display_df.style.apply(highlight_result, axis=1),
                use_container_width=True,
                height=400
            )
            
            st.caption(f"Showing {len(display_df)} fights from {selected_fighter}'s professional career")
        else:
            st.warning(f"No detailed fight history available for {selected_fighter}. Basic statistics are shown above.")
            st.info("Fight history is currently being expanded. Check back soon for complete career records.")


def rankings_page():
    """Rankings page"""
    st.markdown('<div class="main-header">🏆 Rankings</div>', unsafe_allow_html=True)
    
    elo_ratings = load_elo_ratings_cached()
    fighter_stats = load_fighter_stats_cached()
    
    if not elo_ratings:
        st.error("No ELO ratings available. Please run the ELO pipeline first.")
        return
    
    # Create rankings dataframe
    rankings_data = []
    
    for fighter, rating in elo_ratings.items():
        fighter_info = fighter_stats[fighter_stats['name'] == fighter]
        
        if not fighter_info.empty:
            fighter_row = fighter_info.iloc[0]
            rankings_data.append({
                'Rank': 0,  # Will be set after sorting
                'Fighter': fighter,
                'ELO Rating': rating,
                'Record': f"{fighter_row['wins']}-{fighter_row['losses']}-{fighter_row['draws']}",
                'Weight Class': fighter_row.get('weight_class', 'Unknown')
            })
    
    rankings_df = pd.DataFrame(rankings_data)
    rankings_df = rankings_df.sort_values('ELO Rating', ascending=False)
    rankings_df['Rank'] = range(1, len(rankings_df) + 1)
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["Overall Rankings", "By Weight Class"])
    
    with tab1:
        st.markdown("**Top 20 Fighters by ELO Rating**")
        
        top_20 = rankings_df.head(20)
        
        # Display as styled dataframe
        st.dataframe(
            top_20[['Rank', 'Fighter', 'ELO Rating', 'Record', 'Weight Class']],
            use_container_width=True,
            hide_index=True
        )
        
        # Bar chart
        fig = px.bar(
            top_20,
            x='ELO Rating',
            y='Fighter',
            orientation='h',
            title='Top 20 Fighters by ELO Rating',
            color='ELO Rating',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Filter by weight class
        weight_classes = sorted(rankings_df['Weight Class'].unique())
        selected_class = st.selectbox("Select Weight Class", weight_classes)
        
        if selected_class:
            class_rankings = rankings_df[rankings_df['Weight Class'] == selected_class]
            class_rankings = class_rankings.reset_index(drop=True)
            class_rankings['Class Rank'] = range(1, len(class_rankings) + 1)
            
            st.markdown(f"**{selected_class} Rankings**")
            
            st.dataframe(
                class_rankings[['Class Rank', 'Fighter', 'ELO Rating', 'Record']],
                use_container_width=True,
                hide_index=True
            )

def simulator_page():
    """Fight simulator page"""
    st.markdown('<div class="main-header">⚔️ Fight Simulator</div>', unsafe_allow_html=True)
    
    model = load_model_cached()
    fighter_stats = load_fighter_stats_cached()
    
    if model is None or fighter_stats.empty:
        st.error("Model or fighter data not available. Please run the training pipeline first.")
        return
    
    st.markdown("Select two fighters to predict the outcome of their matchup.")
    
    fighter_names = sorted(fighter_stats['name'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        fighter1 = st.selectbox("Fighter 1", fighter_names, key='fighter1')
    
    with col2:
        # Exclude fighter 1 from fighter 2 options
        fighter2_options = [f for f in fighter_names if f != fighter1]
        fighter2 = st.selectbox("Fighter 2", fighter2_options, key='fighter2')
    
    is_title_fight = st.checkbox("Title Fight")
    
    if st.button("🎯 Predict Fight Outcome", type="primary"):
        if fighter1 and fighter2 and fighter1 != fighter2:
            with st.spinner("Analyzing matchup..."):
                try:
                    result = predict_fight(fighter1, fighter2, model, is_title_fight)
                    
                    st.markdown("---")
                    
                    # Display prediction
                    st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
                    
                    # Win probabilities
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {fighter1}")
                        st.metric("Win Probability", f"{result['fighter1_win_probability']:.1%}")
                        st.metric("ELO Rating", f"{result['fighter1_elo']:.0f}")
                    
                    with col2:
                        st.markdown(f"### {fighter2}")
                        st.metric("Win Probability", f"{result['fighter2_win_probability']:.1%}")
                        st.metric("ELO Rating", f"{result['fighter2_elo']:.0f}")
                    
                    # Predicted winner
                    st.markdown("---")
                    
                    winner_col1, winner_col2 = st.columns([2, 1])
                    
                    with winner_col1:
                        st.markdown(f'<div class="sub-header">Predicted Winner: <span class="winner">{result["predicted_winner"]}</span></div>', unsafe_allow_html=True)
                    
                    with winner_col2:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    
                    # Visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=[result['fighter1_win_probability'], result['fighter2_win_probability']],
                        y=[fighter1, fighter2],
                        orientation='h',
                        marker=dict(
                            color=[result['fighter1_win_probability'], result['fighter2_win_probability']],
                            colorscale='RdYlGn',
                            cmin=0,
                            cmax=1
                        ),
                        text=[f"{result['fighter1_win_probability']:.1%}", f"{result['fighter2_win_probability']:.1%}"],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title='Win Probability Comparison',
                        xaxis_title='Win Probability',
                        yaxis_title='Fighter',
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation
                    st.markdown("---")
                    st.markdown('<div class="sub-header">Key Insights</div>', unsafe_allow_html=True)
                    
                    explanation = explain_prediction_simple(result)
                    
                    if 'key_insights' in explanation:
                        for insight in explanation['key_insights']:
                            st.info(f"💡 {insight}")
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Please select two different fighters.")

def feature_importance_page():
    """Feature importance page"""
    st.markdown('<div class="main-header">📊 Feature Importance</div>', unsafe_allow_html=True)
    
    metadata = load_model_metadata_cached()
    
    if not metadata or 'feature_importance' not in metadata:
        st.error("Feature importance data not available.")
        return
    
    st.markdown("Understanding which factors most influence fight predictions.")
    
    # Get top features
    feature_importance = metadata['feature_importance']
    
    # Convert to dataframe
    importance_df = pd.DataFrame([
        {'Feature': feature, 'Importance': importance}
        for feature, importance in feature_importance.items()
    ])
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Display top 20
    top_20 = importance_df.head(20)
    
    # Bar chart
    fig = px.bar(
        top_20,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 20 Most Important Features',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature descriptions
    st.markdown("---")
    st.markdown('<div class="sub-header">Feature Descriptions</div>', unsafe_allow_html=True)
    
    feature_descriptions = {
        'elo_diff': 'Difference in ELO ratings between fighters',
        'win_rate_diff': 'Difference in career win rates',
        'experience_diff': 'Difference in total fights',
        'striking_diff': 'Difference in significant strikes per minute',
        'height_diff': 'Height difference in inches',
        'reach_diff': 'Reach difference in inches',
        'age_diff': 'Age difference in years',
        'finish_rate_diff': 'Difference in finish rate (KO + Submission)',
        'takedown_advantage': 'Combined takedown offense and defense advantage'
    }
    
    for feature, description in feature_descriptions.items():
        if feature in top_20['Feature'].values:
            importance = top_20[top_20['Feature'] == feature]['Importance'].values[0]
            st.markdown(f"**{feature}** (Importance: {importance:.4f})")
            st.write(f"  {description}")
            st.markdown("")

def main():
    """Main application"""
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/200x100/FF4B4B/FFFFFF?text=FightIQ", use_column_width=True)
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "👤 Fighter Profiles", "🏆 Rankings", "⚔️ Fight Simulator", "📊 Feature Importance"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "FightIQ is an advanced MMA fight prediction system using "
        "machine learning, ELO ratings, and comprehensive fighter statistics."
    )
    
    # Route to pages
    if page == "🏠 Home":
        home_page()
    elif page == "👤 Fighter Profiles":
        fighter_profiles_page()
    elif page == "🏆 Rankings":
        rankings_page()
    elif page == "⚔️ Fight Simulator":
        simulator_page()
    elif page == "📊 Feature Importance":
        feature_importance_page()

if __name__ == '__main__':
    main()
