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
    reviews_data = pd.read_csv('../ABSA/aspect_sentiment_results_OneByOne_700_multiaspect.csv')
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

def create_timeline_with_patches(processed_data, patches_data):
    """Create timeline chart with version update annotations"""
    # Get top aspects
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
    
    # Add lines for each aspect
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, aspect in enumerate(top_aspects):
        aspect_data = monthly_stats[monthly_stats['aspect'] == aspect]
        if not aspect_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=aspect_data['year_month'],
                    y=aspect_data['positive_rate'],
                    mode='lines+markers',
                    name=aspect,
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8, color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{aspect}</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Positive Rate: %{y:.2%}<br>' +
                                 '<extra></extra>'
                )
            )
    
    # Add version update vertical lines and annotations
    for idx, patch in patches_data.iterrows():
        patch_date = patch['Release Date']
        patch_date_str = patch_date.strftime('%Y-%m')
        
        # Add vertical line
        fig.add_vline(
            x=patch_date_str,
            line_dash="dash",
            line_color="red",
            line_width=2,
            opacity=0.7
        )
        
        # Add version update annotation
        fig.add_annotation(
            x=patch_date_str,
            y=0.9,
            text=f"<b>{patch['Version']}</b><br>{patch_date.strftime('%Y-%m-%d')}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=0,
            ay=-40,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            font=dict(size=10, color="red")
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'StormGate Positive Rate Trends with Version Updates',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Time",
        yaxis_title="Positive Rate",
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_aspect_sentiment_with_patches(processed_data, patches_data):
    """Create aspect sentiment analysis chart with version update annotations"""
    # Get all aspects
    all_aspects = processed_data['aspect'].unique()
    
    # Create monthly data
    monthly_data = processed_data.copy()
    monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
    
    # Calculate monthly aspect statistics
    aspect_stats = []
    for aspect in all_aspects:
        aspect_data = monthly_data[monthly_data['aspect'] == aspect]
        if len(aspect_data) > 0:
            monthly_stats = aspect_data.groupby('year_month').agg({
                'sentiment': lambda x: {
                    'positive': (x == 'Positive').sum(),
                    'negative': (x == 'Negative').sum(),
                    'neutral': (x == 'Neutral').sum(),
                    'total': len(x)
                }
            }).reset_index()
            
            for _, row in monthly_stats.iterrows():
                stats = row['sentiment']
                aspect_stats.append({
                    'aspect': aspect,
                    'year_month': str(row['year_month']),
                    'positive_count': stats['positive'],
                    'negative_count': stats['negative'],
                    'neutral_count': stats['neutral'],
                    'total_count': stats['total'],
                    'positive_rate': stats['positive'] / stats['total'] if stats['total'] > 0 else 0,
                    'negative_rate': stats['negative'] / stats['total'] if stats['total'] > 0 else 0,
                    'neutral_rate': stats['neutral'] / stats['total'] if stats['total'] > 0 else 0
                })
    
    aspect_df = pd.DataFrame(aspect_stats)
    
    # Create positive rate heatmap
    pivot_positive = aspect_df.pivot(index='aspect', columns='year_month', values='positive_rate')
    
    fig_positive = go.Figure(data=go.Heatmap(
        z=pivot_positive.values,
        x=pivot_positive.columns,
        y=pivot_positive.index,
        colorscale='RdYlGn',
        showscale=True,
        name='Positive Rate',
        hovertemplate='<b>%{y}</b><br>' +
                      'Time: %{x}<br>' +
                      'Positive Rate: %{z:.2%}<br>' +
                      '<extra></extra>'
    ))
    
    # Add version update vertical lines
    for idx, patch in patches_data.iterrows():
        patch_date = patch['Release Date']
        patch_date_str = patch_date.strftime('%Y-%m')
        
        fig_positive.add_vline(
            x=patch_date_str,
            line_dash="dash",
            line_color="red",
            line_width=2,
            opacity=0.8
        )
        
        # Add version update annotation
        fig_positive.add_annotation(
            x=patch_date_str,
            y=len(pivot_positive.index) - 0.5,
            text=f"<b>{patch['Version']}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=0,
            ay=20,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="red",
            borderwidth=1,
            font=dict(size=9, color="red")
        )
    
    fig_positive.update_layout(
        title='Aspect Positive Rate Heatmap with Version Updates',
        xaxis_title='Time',
        yaxis_title='Aspect',
        height=700
    )
    
    # Create negative rate heatmap
    pivot_negative = aspect_df.pivot(index='aspect', columns='year_month', values='negative_rate')
    
    fig_negative = go.Figure(data=go.Heatmap(
        z=pivot_negative.values,
        x=pivot_negative.columns,
        y=pivot_negative.index,
        colorscale='Reds',
        showscale=True,
        name='Negative Rate',
        hovertemplate='<b>%{y}</b><br>' +
                      'Time: %{x}<br>' +
                      'Negative Rate: %{z:.2%}<br>' +
                      '<extra></extra>'
    ))
    
    # Add version update vertical lines
    for idx, patch in patches_data.iterrows():
        patch_date = patch['Release Date']
        patch_date_str = patch_date.strftime('%Y-%m')
        
        fig_negative.add_vline(
            x=patch_date_str,
            line_dash="dash",
            line_color="red",
            line_width=2,
            opacity=0.8
        )
        
        # Add version update annotation
        fig_negative.add_annotation(
            x=patch_date_str,
            y=len(pivot_negative.index) - 0.5,
            text=f"<b>{patch['Version']}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=0,
            ay=20,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="red",
            borderwidth=1,
            font=dict(size=9, color="red")
        )
    
    fig_negative.update_layout(
        title='Aspect Negative Rate Heatmap with Version Updates',
        xaxis_title='Time',
        yaxis_title='Aspect',
        height=700
    )
    
    # Create discussion frequency heatmap
    pivot_frequency = aspect_df.pivot(index='aspect', columns='year_month', values='total_count')
    
    fig_frequency = go.Figure(data=go.Heatmap(
        z=pivot_frequency.values,
        x=pivot_frequency.columns,
        y=pivot_frequency.index,
        colorscale='Blues',
        showscale=True,
        name='Discussion Frequency',
        hovertemplate='<b>%{y}</b><br>' +
                      'Time: %{x}<br>' +
                      'Discussion Count: %{z}<br>' +
                      '<extra></extra>'
    ))
    
    # Add version update vertical lines
    for idx, patch in patches_data.iterrows():
        patch_date = patch['Release Date']
        patch_date_str = patch_date.strftime('%Y-%m')
        
        fig_frequency.add_vline(
            x=patch_date_str,
            line_dash="dash",
            line_color="red",
            line_width=2,
            opacity=0.8
        )
        
        # Add version update annotation
        fig_frequency.add_annotation(
            x=patch_date_str,
            y=len(pivot_frequency.index) - 0.5,
            text=f"<b>{patch['Version']}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=0,
            ay=20,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="red",
            borderwidth=1,
            font=dict(size=9, color="red")
        )
    
    fig_frequency.update_layout(
        title='Aspect Discussion Frequency with Version Updates',
        xaxis_title='Time',
        yaxis_title='Aspect',
        height=700
    )
    
    return fig_positive, fig_negative, fig_frequency, aspect_df

def create_version_impact_analysis(patches_data, processed_data):
    """Create version update impact analysis"""
    version_impacts = []
    
    for idx, patch in patches_data.iterrows():
        patch_date = patch['Release Date']
        before_period = (processed_data['date'] >= patch_date - timedelta(days=30)) & \
                       (processed_data['date'] < patch_date)
        after_period = (processed_data['date'] >= patch_date) & \
                      (processed_data['date'] <= patch_date + timedelta(days=30))
        
        # Overall impact
        before_positive = (processed_data[before_period]['sentiment'] == 'Positive').sum() / \
                         max(1, before_period.sum())
        after_positive = (processed_data[after_period]['sentiment'] == 'Positive').sum() / \
                        max(1, after_period.sum())
        
        # Analyze by aspect
        aspect_impacts = []
        for aspect in processed_data['aspect'].unique():
            aspect_before = processed_data[before_period & (processed_data['aspect'] == aspect)]
            aspect_after = processed_data[after_period & (processed_data['aspect'] == aspect)]
            
            if len(aspect_before) > 0 and len(aspect_after) > 0:
                before_rate = (aspect_before['sentiment'] == 'Positive').sum() / len(aspect_before)
                after_rate = (aspect_after['sentiment'] == 'Positive').sum() / len(aspect_after)
                
                aspect_impacts.append({
                    'aspect': aspect,
                    'before_rate': before_rate,
                    'after_rate': after_rate,
                    'change': after_rate - before_rate
                })
        
        version_impacts.append({
            'version': patch['Version'],
            'date': patch_date,
            'overall_change': after_positive - before_positive,
            'aspect_impacts': aspect_impacts,
            'major_changes': patch['Major Changes']
        })
    
    return version_impacts

def create_sentiment_distribution_chart(processed_data):
    """Create sentiment distribution chart"""
    # Overall sentiment distribution
    sentiment_counts = processed_data['sentiment'].value_counts()
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.3,
        hovertemplate='<b>%{label}</b><br>' +
                      'Count: %{value}<br>' +
                      'Percentage: %{percent}<br>' +
                      '<extra></extra>'
    )])
    
    fig_pie.update_layout(
        title='Overall Sentiment Distribution',
        height=400
    )
    
    # Sentiment distribution by aspect
    aspect_sentiment = processed_data.groupby(['aspect', 'sentiment']).size().reset_index(name='count')
    aspect_sentiment_pivot = aspect_sentiment.pivot(index='aspect', columns='sentiment', values='count').fillna(0)
    
    fig_bar = go.Figure()
    
    colors = {'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#FFA500'}
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment in aspect_sentiment_pivot.columns:
            fig_bar.add_trace(go.Bar(
                name=sentiment,
                x=aspect_sentiment_pivot.index,
                y=aspect_sentiment_pivot[sentiment],
                text=aspect_sentiment_pivot[sentiment],
                textposition='auto',
                marker_color=colors[sentiment],
                hovertemplate=f'<b>%{{x}}</b><br>' +
                             f'{sentiment}: %{{y}}<br>' +
                             '<extra></extra>'
            ))
    
    fig_bar.update_layout(
        title='Sentiment Distribution by Aspect',
        xaxis_title='Aspect',
        yaxis_title='Count',
        barmode='group',
        height=500
    )
    
    return fig_pie, fig_bar

def create_score_analysis_chart(processed_data):
    """Create score analysis chart"""
    # Average score by aspect
    aspect_scores = processed_data.groupby('aspect')['score'].agg(['mean', 'std', 'count']).reset_index()
    aspect_scores = aspect_scores.sort_values('mean', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=aspect_scores['aspect'],
        y=aspect_scores['mean'],
        error_y=dict(type='data', array=aspect_scores['std']),
        text=aspect_scores['mean'].round(3),
        textposition='auto',
        name='Average Score',
        hovertemplate='<b>%{x}</b><br>' +
                      'Average Score: %{y:.3f}<br>' +
                      'Standard Deviation: %{error_y.array:.3f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Average Score by Aspect (with Standard Deviation)',
        xaxis_title='Aspect',
        yaxis_title='Average Score',
        height=500
    )
    
    return fig

def create_trend_analysis_chart(processed_data, patches_data):
    """Create trend analysis chart"""
    # Create monthly data
    monthly_data = processed_data.copy()
    monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
    
    # Calculate monthly trends
    monthly_trends = monthly_data.groupby('year_month').agg({
        'sentiment': lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0,
        'score': 'mean'
    }).reset_index()
    monthly_trends.columns = ['year_month', 'positive_rate', 'avg_score']
    monthly_trends['year_month'] = monthly_trends['year_month'].astype(str)
    
    # Create dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Positive rate
    fig.add_trace(
        go.Scatter(
            x=monthly_trends['year_month'],
            y=monthly_trends['positive_rate'],
            mode='lines+markers',
            name='Positive Rate',
            line=dict(color='#2E8B57', width=3),
            hovertemplate='<b>Positive Rate</b><br>' +
                          'Time: %{x}<br>' +
                          'Rate: %{y:.2%}<br>' +
                          '<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Average score
    fig.add_trace(
        go.Scatter(
            x=monthly_trends['year_month'],
            y=monthly_trends['avg_score'],
            mode='lines+markers',
            name='Average Score',
            line=dict(color='#FF6B6B', width=3),
            hovertemplate='<b>Average Score</b><br>' +
                          'Time: %{x}<br>' +
                          'Score: %{y:.3f}<br>' +
                          '<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add version update vertical lines
    for idx, patch in patches_data.iterrows():
        patch_date = patch['Release Date']
        patch_date_str = patch_date.strftime('%Y-%m')
        
        fig.add_vline(
            x=patch_date_str,
            line_dash="dash",
            line_color="red",
            line_width=2,
            opacity=0.7
        )
        
        # Add version update annotation
        fig.add_annotation(
            x=patch_date_str,
            y=0.9,
            text=f"<b>{patch['Version']}</b><br>{patch_date.strftime('%Y-%m-%d')}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=0,
            ay=-40,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            font=dict(size=10, color="red")
        )
    
    fig.update_layout(
        title='Monthly Trends: Positive Rate vs Average Score with Version Updates',
        xaxis_title='Time',
        height=600
    )
    
    fig.update_yaxes(title_text="Positive Rate", secondary_y=False)
    fig.update_yaxes(title_text="Average Score", secondary_y=True)
    
    return fig

def generate_enhanced_html_dashboard():
    """Generate enhanced HTML dashboard"""
    print("Generating enhanced HTML dashboard with version updates...")
    
    # Load data
    patches_data, processed_data = load_and_process_data()
    
    # Create various charts
    timeline_fig = create_timeline_with_patches(processed_data, patches_data)
    fig_positive, fig_negative, fig_frequency, aspect_df = create_aspect_sentiment_with_patches(processed_data, patches_data)
    version_impacts = create_version_impact_analysis(patches_data, processed_data)
    fig_pie, fig_bar = create_sentiment_distribution_chart(processed_data)
    fig_score = create_score_analysis_chart(processed_data)
    fig_trend = create_trend_analysis_chart(processed_data, patches_data)
    
    # Basic statistics
    total_reviews = len(processed_data)
    unique_aspects = processed_data['aspect'].nunique()
    overall_positive_rate = (processed_data['sentiment'] == 'Positive').sum() / len(processed_data)
    overall_negative_rate = (processed_data['sentiment'] == 'Negative').sum() / len(processed_data)
    avg_score = processed_data['score'].mean()
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced StormGate Analytics Dashboard with Version Updates</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f8f9fa;
            }}
            .stat-card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            .stat-card:hover {{
                transform: translateY(-5px);
            }}
            .stat-number {{
                font-size: 2.5em;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 10px;
            }}
            .stat-label {{
                color: #666;
                font-size: 1.1em;
            }}
            .chart-section {{
                padding: 30px;
                border-bottom: 1px solid #eee;
            }}
            .chart-section:last-child {{
                border-bottom: none;
            }}
            .chart-title {{
                font-size: 1.8em;
                font-weight: 600;
                margin-bottom: 20px;
                color: #333;
                border-left: 4px solid #667eea;
                padding-left: 15px;
            }}
            .chart-container {{
                background: white;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            .tabs {{
                display: flex;
                background: #f8f9fa;
                border-radius: 10px;
                margin-bottom: 20px;
                overflow: hidden;
            }}
            .tab {{
                flex: 1;
                padding: 15px;
                text-align: center;
                background: #e9ecef;
                cursor: pointer;
                transition: all 0.3s ease;
                border: none;
                font-size: 1.1em;
            }}
            .tab.active {{
                background: #667eea;
                color: white;
            }}
            .tab:hover {{
                background: #5a6fd8;
                color: white;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
            .version-info {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
            }}
            .version-item {{
                background: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .version-title {{
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }}
            .version-date {{
                color: #666;
                font-size: 0.9em;
                margin-bottom: 10px;
            }}
            .version-changes {{
                color: #333;
                line-height: 1.5;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Enhanced StormGate Analytics Dashboard</h1>
                <p>Comprehensive analysis with version update annotations</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{total_reviews:,}</div>
                    <div class="stat-label">Total Reviews</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{unique_aspects}</div>
                    <div class="stat-label">Unique Aspects</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{overall_positive_rate:.1%}</div>
                    <div class="stat-label">Positive Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{overall_negative_rate:.1%}</div>
                    <div class="stat-label">Negative Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{avg_score:.2f}</div>
                    <div class="stat-label">Average Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(patches_data)}</div>
                    <div class="stat-label">Version Updates</div>
                </div>
            </div>
            
            <div class="chart-section">
                <div class="chart-title">Overview Analysis</div>
                <div class="tabs">
                    <button class="tab active" onclick="showTab('overview-sentiment')">Overall Sentiment</button>
                    <button class="tab" onclick="showTab('overview-aspects')">Aspect Sentiment</button>
                </div>
                
                <div id="overview-sentiment" class="tab-content active">
                    <div class="chart-container">
                        <div id="overview-pie"></div>
                    </div>
                </div>
                
                <div id="overview-aspects" class="tab-content">
                    <div class="chart-container">
                        <div id="overview-bar"></div>
                    </div>
                </div>
            </div>
            
            <div class="chart-section">
                <div class="chart-title">Timeline Analysis with Version Updates</div>
                <div class="chart-container">
                    <div id="timeline-chart"></div>
                </div>
            </div>
            
            <div class="chart-section">
                <div class="chart-title">Aspect Analysis Over Time</div>
                <div class="tabs">
                    <button class="tab active" onclick="showTab('aspect-positive')">Positive Rate</button>
                    <button class="tab" onclick="showTab('aspect-negative')">Negative Rate</button>
                    <button class="tab" onclick="showTab('aspect-frequency')">Discussion Frequency</button>
                </div>
                
                <div id="aspect-positive" class="tab-content active">
                    <div class="chart-container">
                        <div id="aspect-positive-heatmap"></div>
                    </div>
                </div>
                
                <div id="aspect-negative" class="tab-content">
                    <div class="chart-container">
                        <div id="aspect-negative-heatmap"></div>
                    </div>
                </div>
                
                <div id="aspect-frequency" class="tab-content">
                    <div class="chart-container">
                        <div id="aspect-frequency-heatmap"></div>
                    </div>
                </div>
            </div>
            
            <div class="chart-section">
                <div class="chart-title">Sentiment Analysis</div>
                <div class="tabs">
                    <button class="tab active" onclick="showTab('sentiment-overview')">Overview</button>
                    <button class="tab" onclick="showTab('sentiment-distribution')">Distribution</button>
                </div>
                
                <div id="sentiment-overview" class="tab-content active">
                    <div class="chart-container">
                        <div id="sentiment-pie"></div>
                    </div>
                </div>
                
                <div id="sentiment-distribution" class="tab-content">
                    <div class="chart-container">
                        <div id="sentiment-bar"></div>
                    </div>
                </div>
            </div>
            
            <div class="chart-section">
                <div class="chart-title">Score Analysis</div>
                <div class="chart-container">
                    <div id="score-analysis"></div>
                </div>
            </div>
            
            <div class="chart-section">
                <div class="chart-title">Trend Analysis with Version Updates</div>
                <div class="chart-container">
                    <div id="trend-analysis"></div>
                </div>
            </div>
            
            <div class="chart-section">
                <div class="chart-title">Version Update Information</div>
                <div class="version-info">
                    {create_version_info_html(patches_data)}
                </div>
            </div>
        </div>
        
        <script>
            // Tab functionality
            function showTab(tabId) {{
                // Hide all tab contents
                var contents = document.querySelectorAll('.tab-content');
                contents.forEach(function(content) {{
                    content.classList.remove('active');
                }});
                
                // Remove active class from all tabs
                var tabs = document.querySelectorAll('.tab');
                tabs.forEach(function(tab) {{
                    tab.classList.remove('active');
                }});
                
                // Show selected tab content
                document.getElementById(tabId).classList.add('active');
                
                // Add active class to clicked tab
                event.target.classList.add('active');
            }}
            
            // Load charts
            var overviewPieData = {fig_pie.to_json()};
            Plotly.newPlot('overview-pie', overviewPieData.data, overviewPieData.layout);
            
            var overviewBarData = {fig_bar.to_json()};
            Plotly.newPlot('overview-bar', overviewBarData.data, overviewBarData.layout);
            
            var timelineData = {timeline_fig.to_json()};
            Plotly.newPlot('timeline-chart', timelineData.data, timelineData.layout);
            
            var aspectPositiveData = {fig_positive.to_json()};
            Plotly.newPlot('aspect-positive-heatmap', aspectPositiveData.data, aspectPositiveData.layout);
            
            var aspectNegativeData = {fig_negative.to_json()};
            Plotly.newPlot('aspect-negative-heatmap', aspectNegativeData.data, aspectNegativeData.layout);
            
            var aspectFrequencyData = {fig_frequency.to_json()};
            Plotly.newPlot('aspect-frequency-heatmap', aspectFrequencyData.data, aspectFrequencyData.layout);
            
            var sentimentPieData = {fig_pie.to_json()};
            Plotly.newPlot('sentiment-pie', sentimentPieData.data, sentimentPieData.layout);
            
            var sentimentBarData = {fig_bar.to_json()};
            Plotly.newPlot('sentiment-bar', sentimentBarData.data, sentimentBarData.layout);
            
            var scoreData = {fig_score.to_json()};
            Plotly.newPlot('score-analysis', scoreData.data, scoreData.layout);
            
            var trendData = {fig_trend.to_json()};
            Plotly.newPlot('trend-analysis', trendData.data, trendData.layout);
        </script>
    </body>
    </html>
    """
    
    # Save HTML file
    with open('enhanced_stormgate_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Enhanced HTML dashboard generated: enhanced_stormgate_dashboard.html")
    return 'enhanced_stormgate_dashboard.html'

def create_version_info_html(patches_data):
    """Create version information HTML"""
    html = ""
    for idx, patch in patches_data.iterrows():
        html += f"""
        <div class="version-item">
            <div class="version-title">{patch['Version']}</div>
            <div class="version-date">Release Date: {patch['Release Date'].strftime('%Y-%m-%d')}</div>
            <div class="version-changes">{patch['Major Changes']}</div>
        </div>
        """
    return html

def main():
    """Main function"""
    print("Enhanced StormGate Analytics Dashboard with Version Updates")
    print("=" * 60)
    
    # Generate enhanced HTML dashboard
    html_file = generate_enhanced_html_dashboard()
    
    # Automatically open browser
    try:
        webbrowser.open(f'file://{os.path.abspath(html_file)}')
        print(f"Dashboard opened in browser: {os.path.abspath(html_file)}")
    except:
        print(f"Please open the dashboard manually: {os.path.abspath(html_file)}")
    
    print("\nEnhanced dashboard generation completed!")
    print("The enhanced dashboard is saved as 'enhanced_stormgate_dashboard.html'")
    print("New features included:")
    print("- Version update annotations on all timeline charts")
    print("- Hover information for version updates")
    print("- Version update information panel")
    print("- Interactive version update markers")
    print("- Enhanced hover tooltips with version context")

if __name__ == "__main__":
    main()