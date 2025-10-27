import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="StormGate Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_process_data():
    """Load and process data"""
    # Load version update data
    patches_data = pd.read_csv('StormGatePatches_fixed.csv')
    patches_data['Release Date'] = pd.to_datetime(patches_data['Release Date'])
    
    # Load review data
    reviews_data = pd.read_csv('aspect_sentiment_results_OneByOne_700_multiaspect.csv')
    reviews_data['date_time'] = pd.to_datetime(reviews_data['date_time'])
    
    # Process review data
    processed_reviews = []
    for idx, row in reviews_data.iterrows():
        try:
            if pd.notna(row['predicted_aspects']) and row['predicted_aspects'] != '[]':
                aspects = json.loads(row['predicted_aspects'])
                for aspect_data in aspects:
                    processed_reviews.append({
                        'date': row['date_time'],
                        'aspect': aspect_data['aspect'],
                        'sentiment': aspect_data['sentiment'],
                        'score': aspect_data['score'],
                        'snippet': aspect_data['snippet']
                    })
        except:
            continue
    
    processed_data = pd.DataFrame(processed_reviews)
    return patches_data, processed_data

def create_simple_timeline_chart(patches_data, processed_data):
    """Create simplified timeline chart"""
    # Get main aspects
    top_aspects = processed_data['aspect'].value_counts().head(6).index.tolist()
    
    # Create monthly data
    monthly_data = processed_data.copy()
    monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
    
    # Calculate monthly aspect statistics
    monthly_stats = monthly_data.groupby(['year_month', 'aspect']).agg({
        'sentiment': lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0,
        'score': 'mean'
    }).reset_index()
    monthly_stats.columns = ['year_month', 'aspect', 'positive_rate', 'avg_score']
    monthly_stats['year_month'] = monthly_stats['year_month'].astype(str)
    
    # Create chart
    fig = go.Figure()
    
    # Add line for each aspect
    for aspect in top_aspects:
        aspect_data = monthly_stats[monthly_stats['aspect'] == aspect]
        if not aspect_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=aspect_data['year_month'],
                    y=aspect_data['positive_rate'],
                    mode='lines+markers',
                    name=aspect,
                    line=dict(width=3),
                    marker=dict(size=8)
                )
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'StormGate Positive Rate Trends by Aspect',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Time",
        yaxis_title="Positive Rate",
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_version_impact_chart(patches_data, processed_data):
    """Create version update impact chart"""
    version_impacts = []
    for idx, patch in patches_data.iterrows():
        patch_date = patch['Release Date']
        before_period = (processed_data['date'] >= patch_date - timedelta(days=30)) & \
                       (processed_data['date'] < patch_date)
        after_period = (processed_data['date'] >= patch_date) & \
                      (processed_data['date'] <= patch_date + timedelta(days=30))
        
        before_positive = (processed_data[before_period]['sentiment'] == 'Positive').sum() / \
                         max(1, before_period.sum())
        after_positive = (processed_data[after_period]['sentiment'] == 'Positive').sum() / \
                        max(1, after_period.sum())
        
        version_impacts.append({
            'version': patch['Version'],
            'change': after_positive - before_positive,
            'before': before_positive,
            'after': after_positive
        })
    
    if version_impacts:
        impact_df = pd.DataFrame(version_impacts)
        colors = ['#2E8B57' if x > 0 else '#DC143C' for x in impact_df['change']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=impact_df['version'],
                y=impact_df['change'],
                marker_color=colors,
                text=[f"{x:.1%}" for x in impact_df['change']],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Version Update Impact on Positive Rate',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Version",
            yaxis_title="Change in Positive Rate",
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    return None

def create_aspect_ranking_chart(processed_data):
    """Create aspect ranking chart"""
    top_aspects = processed_data['aspect'].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_aspects.values,
            y=top_aspects.index,
            orientation='h',
            marker_color='#1f77b4'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Top 10 Most Discussed Aspects',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Number of Reviews",
        yaxis_title="Aspect",
        height=500,
        template='plotly_white'
    )
    
    return fig

def main():
    """Main function"""
    # Title
    st.title("🎮 StormGate Analytics Dashboard")
    st.markdown("Real-time analysis of player sentiment and version updates")
    
    # Load data
    with st.spinner('Loading data...'):
        patches_data, processed_data = load_and_process_data()
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Reviews",
            value=f"{len(processed_data):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Unique Aspects",
            value=f"{processed_data['aspect'].nunique()}",
            delta=None
        )
    
    with col3:
        overall_positive_rate = (processed_data['sentiment'] == 'Positive').sum() / len(processed_data)
        st.metric(
            label="Overall Positive Rate",
            value=f"{overall_positive_rate:.1%}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Version Updates",
            value=f"{len(patches_data)}",
            delta=None
        )
    
    # Timeline analysis
    st.header("📈 Positive Rate Trends")
    timeline_fig = create_simple_timeline_chart(patches_data, processed_data)
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Version update impact
    st.header("🚀 Version Update Impact")
    impact_fig = create_version_impact_chart(patches_data, processed_data)
    if impact_fig:
        st.plotly_chart(impact_fig, use_container_width=True)
    
    # Aspect ranking
    st.header("🏆 Most Discussed Aspects")
    ranking_fig = create_aspect_ranking_chart(processed_data)
    st.plotly_chart(ranking_fig, use_container_width=True)
    
    # Version update impact table
    st.header("📋 Version Update Impact Details")
    
    version_impacts = []
    for idx, patch in patches_data.iterrows():
        patch_date = patch['Release Date']
        before_period = (processed_data['date'] >= patch_date - timedelta(days=30)) & \
                       (processed_data['date'] < patch_date)
        after_period = (processed_data['date'] >= patch_date) & \
                      (processed_data['date'] <= patch_date + timedelta(days=30))
        
        before_positive = (processed_data[before_period]['sentiment'] == 'Positive').sum() / \
                         max(1, before_period.sum())
        after_positive = (processed_data[after_period]['sentiment'] == 'Positive').sum() / \
                        max(1, after_period.sum())
        
        version_impacts.append({
            'Version': patch['Version'],
            'Release Date': patch_date.strftime('%Y-%m-%d'),
            'Before Update': f"{before_positive:.1%}",
            'After Update': f"{after_positive:.1%}",
            'Change': f"{after_positive - before_positive:+.1%}",
            'Impact': 'Positive' if after_positive > before_positive else 'Negative'
        })
    
    impact_df = pd.DataFrame(version_impacts)
    st.dataframe(impact_df, use_container_width=True)

if __name__ == "__main__":
    main()
