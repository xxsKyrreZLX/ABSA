import pandas as pd
import json
from datetime import datetime, timedelta

def test_data_loading():
    """Test data loading"""
    print("Testing data loading...")
    
    try:
        # Load version update data
        patches_data = pd.read_csv('StormGatePatches_fixed.csv')
        patches_data['Release Date'] = pd.to_datetime(patches_data['Release Date'])
        print(f"Patches data loaded: {len(patches_data)} records")
        
        # Load review data
        reviews_data = pd.read_csv('aspect_sentiment_results_OneByOne_700_multiaspect.csv')
        reviews_data['date_time'] = pd.to_datetime(reviews_data['date_time'])
        print(f"Reviews data loaded: {len(reviews_data)} records")
        
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
        
        # Basic statistics
        print(f"\nData Summary:")
        print(f"Total Reviews: {len(processed_data):,}")
        print(f"Unique Aspects: {processed_data['aspect'].nunique()}")
        print(f"Date Range: {processed_data['date'].min()} to {processed_data['date'].max()}")
        
        overall_positive_rate = (processed_data['sentiment'] == 'Positive').sum() / len(processed_data)
        print(f"Overall Positive Rate: {overall_positive_rate:.1%}")
        
        # Most popular aspects
        top_aspects = processed_data['aspect'].value_counts().head(5)
        print(f"\nTop 5 Aspects:")
        for aspect, count in top_aspects.items():
            print(f"  {aspect}: {count:,} reviews")
        
        # Version update impact analysis
        print(f"\nVersion Update Impact:")
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
            
            change = after_positive - before_positive
            change_str = f"+{change:.1%}" if change > 0 else f"{change:.1%}"
            print(f"  {patch['Version']}: {change_str} (Post-update: {after_positive:.1%})")
        
        print("\nData loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_data_loading()
