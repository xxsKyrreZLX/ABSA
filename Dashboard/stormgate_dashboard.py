import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class StormGateDashboard:
    def __init__(self, patches_file, reviews_file):
        """Initialize dashboard"""
        self.patches_file = patches_file
        self.reviews_file = reviews_file
        self.patches_data = None
        self.reviews_data = None
        self.processed_data = None
        
    def load_data(self):
        """Load data"""
        print("Loading data...")
        
        # Load version update data
        self.patches_data = pd.read_csv(self.patches_file)
        self.patches_data['Release Date'] = pd.to_datetime(self.patches_data['Release Date'])
        
        # Load review data
        self.reviews_data = pd.read_csv(self.reviews_file)
        self.reviews_data['date_time'] = pd.to_datetime(self.reviews_data['date_time'])
        
        print(f"Version update data: {len(self.patches_data)} records")
        print(f"Review data: {len(self.reviews_data)} records")
        
    def parse_aspects(self, aspect_str):
        """Parse aspect data"""
        try:
            if pd.isna(aspect_str) or aspect_str == '[]':
                return []
            aspects = json.loads(aspect_str)
            return aspects
        except:
            return []
    
    def process_reviews(self):
        """Process review data"""
        print("Processing review data...")
        
        processed_reviews = []
        
        for idx, row in self.reviews_data.iterrows():
            if idx % 1000 == 0:
                print(f"Processing progress: {idx}/{len(self.reviews_data)}")
                
            aspects = self.parse_aspects(row['predicted_aspects'])
            
            for aspect_data in aspects:
                processed_reviews.append({
                    'date': row['date_time'],
                    'aspect': aspect_data['aspect'],
                    'sentiment': aspect_data['sentiment'],
                    'score': aspect_data['score'],
                    'snippet': aspect_data['snippet']
                })
        
        self.processed_data = pd.DataFrame(processed_reviews)
        print(f"Processed data: {len(self.processed_data)} records")
        
    def create_timeline_chart(self):
        """Create timeline chart"""
        # Get main aspects
        top_aspects = self.processed_data['aspect'].value_counts().head(8).index.tolist()
        
        # Create monthly data
        monthly_data = self.processed_data.copy()
        monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
        
        # Calculate monthly aspect statistics
        monthly_stats = monthly_data.groupby(['year_month', 'aspect']).agg({
            'sentiment': lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0,
            'score': 'mean'
        }).reset_index()
        monthly_stats.columns = ['year_month', 'aspect', 'positive_rate', 'avg_score']
        monthly_stats['year_month'] = monthly_stats['year_month'].astype(str)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Review Count Over Time', 'Positive Rate Trends', 
                          'Version Update Impact', 'Aspect Sentiment Heatmap'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "heatmap"}]]
        )
        
        # 1. Review count changes
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
                    ),
                    row=1, col=1
                )
        
        # 2. Positive rate trends
        overall_sentiment = monthly_data.groupby('year_month').agg({
            'sentiment': lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        overall_sentiment.columns = ['year_month', 'positive_rate']
        overall_sentiment['year_month'] = overall_sentiment['year_month'].astype(str)
        
        fig.add_trace(
            go.Scatter(
                x=overall_sentiment['year_month'],
                y=overall_sentiment['positive_rate'],
                mode='lines+markers',
                name='Overall Positive Rate',
                line=dict(color='#2E8B57', width=4),
                marker=dict(size=10, color='#2E8B57')
            ),
            row=1, col=2
        )
        
        # 3. Version update impact
        version_impacts = []
        for idx, patch in self.patches_data.iterrows():
            patch_date = patch['Release Date']
            before_period = (self.processed_data['date'] >= patch_date - timedelta(days=30)) & \
                           (self.processed_data['date'] < patch_date)
            after_period = (self.processed_data['date'] >= patch_date) & \
                          (self.processed_data['date'] <= patch_date + timedelta(days=30))
            
            before_positive = (self.processed_data[before_period]['sentiment'] == 'Positive').sum() / \
                             max(1, before_period.sum())
            after_positive = (self.processed_data[after_period]['sentiment'] == 'Positive').sum() / \
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
            
            fig.add_trace(
                go.Bar(
                    x=impact_df['version'],
                    y=impact_df['change'],
                    marker_color=colors,
                    name='Version Impact',
                    text=[f"{x:.1%}" for x in impact_df['change']],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # 4. Sentiment heatmap
        sentiment_heatmap = monthly_data.groupby(['year_month', 'aspect']).agg({
            'sentiment': lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        sentiment_heatmap.columns = ['year_month', 'aspect', 'positive_rate']
        sentiment_heatmap['year_month'] = sentiment_heatmap['year_month'].astype(str)
        
        # Create pivot table
        pivot_data = sentiment_heatmap.pivot(index='aspect', columns='year_month', values='positive_rate')
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn',
                showscale=True,
                name='Sentiment Heatmap'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'StormGate Timeline Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#1f77b4'}
            },
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Add version update lines
        for patch_date in self.patches_data['Release Date']:
            patch_date_str = patch_date.strftime('%Y-%m')
            fig.add_vline(
                x=patch_date_str,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                annotation_text=f"Update: {patch_date.strftime('%Y-%m-%d')}"
            )
        
        return fig
    
    def create_aspect_analysis_chart(self):
        """Create aspect analysis chart"""
        # Get main aspects
        top_aspects = self.processed_data['aspect'].value_counts().head(6).index.tolist()
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'{aspect} Analysis' for aspect in top_aspects],
            specs=[[{"secondary_y": True} for _ in range(3)],
                   [{"secondary_y": True} for _ in range(3)]]
        )
        
        for idx, aspect in enumerate(top_aspects):
            row = (idx // 3) + 1
            col = (idx % 3) + 1
            
            # Get data for this aspect
            aspect_data = self.processed_data[self.processed_data['aspect'] == aspect]
            monthly_data = aspect_data.groupby(aspect_data['date'].dt.to_period('M')).agg({
                'sentiment': lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0,
                'score': 'mean'
            }).reset_index()
            monthly_data.columns = ['year_month', 'positive_rate', 'avg_score']
            monthly_data['year_month'] = monthly_data['year_month'].astype(str)
            
            # Plot positive rate
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['year_month'],
                    y=monthly_data['positive_rate'],
                    mode='lines+markers',
                    name='Positive Rate',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ),
                row=row, col=col, secondary_y=False
            )
            
            # Plot average score
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['year_month'],
                    y=monthly_data['avg_score'],
                    mode='lines+markers',
                    name='Average Score',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8)
                ),
                row=row, col=col, secondary_y=True
            )
            
            # Add version update lines
            for patch_date in self.patches_data['Release Date']:
                patch_date_str = patch_date.strftime('%Y-%m')
                fig.add_vline(
                    x=patch_date_str,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.5,
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'StormGate Detailed Aspect Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#1f77b4'}
            },
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_summary_cards(self):
        """Create summary cards"""
        # Basic statistics
        total_reviews = len(self.processed_data)
        unique_aspects = self.processed_data['aspect'].nunique()
        overall_positive_rate = (self.processed_data['sentiment'] == 'Positive').sum() / len(self.processed_data)
        
        # Most popular aspects
        top_aspects = self.processed_data['aspect'].value_counts().head(3)
        
        cards = [
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{total_reviews:,}", className="card-title text-primary"),
                    html.P("Total Reviews", className="card-text")
                ])
            ], className="text-center"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{unique_aspects}", className="card-title text-success"),
                    html.P("Unique Aspects", className="card-text")
                ])
            ], className="text-center"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{overall_positive_rate:.1%}", className="card-title text-warning"),
                    html.P("Overall Positive Rate", className="card-text")
                ])
            ], className="text-center"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(self.patches_data)}", className="card-title text-info"),
                    html.P("Version Updates", className="card-text")
                ])
            ], className="text-center")
        ]
        
        return cards
    
    def create_dashboard(self):
        """Create complete dashboard"""
        # Load and process data
        self.load_data()
        self.process_reviews()
        
        # Create charts
        timeline_fig = self.create_timeline_chart()
        aspect_fig = self.create_aspect_analysis_chart()
        
        # Create summary cards
        summary_cards = self.create_summary_cards()
        
        # Create Dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("StormGate Analytics Dashboard", 
                           className="text-center mb-4 text-primary"),
                    html.P("Real-time analysis of player sentiment and version updates",
                          className="text-center text-muted mb-4")
                ])
            ]),
            
            # Summary cards
            dbc.Row([
                dbc.Col(card, width=3) for card in summary_cards
            ], className="mb-4"),
            
            # Timeline analysis
            dbc.Row([
                dbc.Col([
                    html.H3("Timeline Analysis", className="mb-3"),
                    dcc.Graph(figure=timeline_fig, id='timeline-chart')
                ])
            ], className="mb-4"),
            
            # Aspect analysis
            dbc.Row([
                dbc.Col([
                    html.H3("Detailed Aspect Analysis", className="mb-3"),
                    dcc.Graph(figure=aspect_fig, id='aspect-chart')
                ])
            ]),
            
            # Data table
            dbc.Row([
                dbc.Col([
                    html.H3("Version Update Impact", className="mb-3"),
                    self.create_impact_table()
                ])
            ], className="mt-4")
            
        ], fluid=True)
        
        return app
    
    def create_impact_table(self):
        """Create impact table"""
        version_impacts = []
        for idx, patch in self.patches_data.iterrows():
            patch_date = patch['Release Date']
            before_period = (self.processed_data['date'] >= patch_date - timedelta(days=30)) & \
                           (self.processed_data['date'] < patch_date)
            after_period = (self.processed_data['date'] >= patch_date) & \
                          (self.processed_data['date'] <= patch_date + timedelta(days=30))
            
            before_positive = (self.processed_data[before_period]['sentiment'] == 'Positive').sum() / \
                             max(1, before_period.sum())
            after_positive = (self.processed_data[after_period]['sentiment'] == 'Positive').sum() / \
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
        
        return dbc.Table.from_dataframe(
            impact_df,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="table-striped"
        )

def create_dashboard():
    """Create and run dashboard"""
    dashboard = StormGateDashboard('StormGatePatches_fixed.csv', 'aspect_sentiment_results_OneByOne_700_multiaspect.csv')
    app = dashboard.create_dashboard()
    return app

if __name__ == "__main__":
    app = create_dashboard()
    app.run_server(debug=True, host='0.0.0.0', port=8050)
