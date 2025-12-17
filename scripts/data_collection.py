"""
Collect training data from Garmin Connect
"""
import json
import pandas as pd
from garminconnect import Garmin
from datetime import datetime, timedelta
import time

# Load credentials
with open('config.json', 'r') as f:
    config = json.load(f)

def connect_garmin():
    """Authenticate with Garmin Connect"""
    try:
        client = Garmin(config['garmin_email'], config['garmin_password'])
        client.login()
        print("✓ Connected to Garmin Connect successfully!")
        return client
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None

def get_activities(client, num_activities=100):
    """Download recent activities"""
    try:
        activities = client.get_activities(0, num_activities)
        df = pd.DataFrame(activities)
        
        # Keep relevant columns
        cols_to_keep = [
            'activityId', 'activityName', 'activityType', 'startTimeLocal',
            'distance', 'duration', 'averageHR', 'maxHR', 'calories',
            'elevationGain', 'averageSpeed', 'maxSpeed'
        ]
        # Only keep columns that exist
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        df = df[cols_to_keep]
        
        print(f"✓ Downloaded {len(df)} activities")
        return df
    except Exception as e:
        print(f"✗ Failed to get activities: {e}")
        return None

def get_daily_health(client, start_date, end_date):
    """Get health metrics for a date range"""
    health_data = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        try:
            # Get various health metrics
            daily_stats = {
                'date': date_str,
                'resting_hr': None,
                'hrv_rmssd': None,
                'sleep_hours': None,
                'stress_avg': None,
                'steps': None,
                'calories': None
            }
            
            # Try to get each metric
            try:
                hrv = client.get_hrv_data(date_str)
                if hrv and 'hrvSummary' in hrv:
                    daily_stats['hrv_rmssd'] = hrv['hrvSummary'].get('lastNightAvg')
            except:
                pass
            
            try:
                sleep = client.get_sleep_data(date_str)
                if sleep and 'dailySleepDTO' in sleep:
                    sleep_seconds = sleep['dailySleepDTO'].get('sleepTimeSeconds', 0)
                    daily_stats['sleep_hours'] = sleep_seconds / 3600
            except:
                pass
            
            try:
                stress = client.get_stress_data(date_str)
                if stress and 'avgStressLevel' in stress:
                    daily_stats['stress_avg'] = stress['avgStressLevel']
            except:
                pass
            
            try:
                stats = client.get_stats(date_str)
                if stats:
                    daily_stats['resting_hr'] = stats.get('restingHeartRate')
                    daily_stats['steps'] = stats.get('totalSteps')
                    daily_stats['calories'] = stats.get('totalKilocalories')
            except:
                pass
            
            health_data.append(daily_stats)
            print(f"✓ Health data for {date_str}")
            
        except Exception as e:
            print(f"✗ Failed for {date_str}: {e}")
        
        current_date += timedelta(days=1)
        time.sleep(1)  # Be nice to Garmin servers
    
    return pd.DataFrame(health_data)

if __name__ == "__main__":
    # Connect
    client = connect_garmin()
    
    if client:
        print("\n=== Collecting Activities ===")
        activities_df = get_activities(client, 100)
        
        if activities_df is not None:
            activities_df.to_csv('data/raw/activities.csv', index=False)
            print(f"✓ Saved {len(activities_df)} activities to data/raw/activities.csv")
            
            print("\nRecent activities:")
            print(activities_df[['activityName', 'distance', 'duration', 'averageHR']].head(10))
        
        print("\n=== Collecting Health Data ===")
        # Get last 30 days of health data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        health_df = get_daily_health(client, start_date, end_date)
        
        if not health_df.empty:
            health_df.to_csv('data/raw/health_metrics.csv', index=False)
            print(f"\n✓ Saved {len(health_df)} days of health data to data/raw/health_metrics.csv")
            print("\nRecent health metrics:")
            print(health_df.tail(10))
