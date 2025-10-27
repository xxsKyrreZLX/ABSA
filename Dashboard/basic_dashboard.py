import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import webbrowser
import os

def load_and_process_data():
    """Load and process data"""
    print("Loading data...")
    
    # Load version update data
    patches_data = pd.read_csv('StormGatePatches_fixed.csv')
    patches_data['Release Date'] = pd.to_datetime(patches_data['Release Date'])
    
    # Load review data
    reviews_data = pd.read_csv('aspect_sentiment_results_OneByOne_700_multiaspect.csv')
    reviews_data['date_time'] = pd.to_datetime(reviews_data['date_time'])
    
    # Process review data
    processed_reviews = []
    for idx, row in reviews_data.iterrows():
        if idx % 1000 == 0:
            print(f"Processing progress: {idx}/{len(reviews_data)}")
            
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
    print(f"Processed data: {len(processed_data)} records")
    
    return patches_data, processed_data

def create_timeline_chart(patches_data, processed_data):
    """Create timeline chart"""
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

def generate_html_dashboard():
    """Generate HTML dashboard"""
    print("Generating HTML dashboard...")
    
    # Load data
    patches_data, processed_data = load_and_process_data()
    
    # Create chart
    timeline_fig = create_timeline_chart(patches_data, processed_data)
    impact_fig = create_version_impact_chart(patches_data, processed_data)
    ranking_fig = create_aspect_ranking_chart(processed_data)
    
    # Basic statistics
    total_reviews = len(processed_data)
    unique_aspects = processed_data['aspect'].nunique()
    overall_positive_rate = (processed_data['sentiment'] == 'Positive').sum() / len(processed_data)
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>StormGate Analytics Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
                min-width: 150px;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }}
            .chart-container {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .chart-title {{
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>StormGate Analytics Dashboard</h1>
            <p>Real-time analysis of player sentiment and version updates</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{total_reviews:,}</div>
                <div>Total Reviews</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{unique_aspects}</div>
                <div>Unique Aspects</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{overall_positive_rate:.1%}</div>
                <div>Positive Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(patches_data)}</div>
                <div>Version Updates</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Positive Rate Trends by Aspect</div>
            <div id="timeline-chart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Version Update Impact</div>
            <div id="impact-chart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Most Discussed Aspects</div>
            <div id="ranking-chart"></div>
        </div>
        
        <script>
            // Timeline chart
            var timelineData = {timeline_fig.to_json()};
            Plotly.newPlot('timeline-chart', timelineData.data, timelineData.layout);
            
            // Version impact chart
            var impactData = {impact_fig.to_json() if impact_fig else 'null'};
            if (impactData) {{
                Plotly.newPlot('impact-chart', impactData.data, impactData.layout);
            }}
            
            // Ranking chart
            var rankingData = {ranking_fig.to_json()};
            Plotly.newPlot('ranking-chart', rankingData.data, rankingData.layout);
        </script>
    </body>
    </html>
    """
    
    # Save HTML file
    with open('stormgate_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("HTML dashboard generated: stormgate_dashboard.html")
    return 'stormgate_dashboard.html'

def main():
    """Main function"""
    print("StormGate Analytics Dashboard")
    print("=" * 50)
    
    # Generate HTML dashboard
    html_file = generate_html_dashboard()
    
    # Auto-open browser
    try:
        webbrowser.open(f'file://{os.path.abspath(html_file)}')
        print(f"Dashboard opened in browser: {os.path.abspath(html_file)}")
    except:
        print(f"Please open the dashboard manually: {os.path.abspath(html_file)}")
    
    print("\nDashboard generation completed!")
    print("The dashboard is saved as 'stormgate_dashboard.html'")
    print("You can open it in any web browser.")

if __name__ == "__main__":
    main()
