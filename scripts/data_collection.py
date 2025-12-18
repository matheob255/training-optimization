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


def get_activities(client, num_activities=365):
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
    
    total_days = (end_date - start_date).days + 1
    print(f"Fetching health data for {total_days} days...")
    
    day_count = 0
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        day_count += 1
        
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
            
            # Progress indicator
            if day_count % 10 == 0:
                print(f"✓ Progress: {day_count}/{total_days} days ({day_count/total_days*100:.0f}%)")
            
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
        # CHANGED: Fetch 365 activities instead of 100
        activities_df = get_activities(client, 365)
        
        if activities_df is not None:
            activities_df.to_csv('data/raw/activities.csv', index=False)
            print(f"✓ Saved {len(activities_df)} activities to data/raw/activities.csv")
            
            print("\nRecent activities:")
            print(activities_df[['activityName', 'distance', 'duration', 'averageHR']].head(10))
        
        print("\n=== Collecting Health Data ===")
        # CHANGED: Get last 180 days (6 months) of health data instead of 30
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        
        print(f"Fetching health data from {start_date} to {end_date}")
        health_df = get_daily_health(client, start_date, end_date)
        
        if not health_df.empty:
            health_df.to_csv('data/raw/health_metrics.csv', index=False)
            print(f"\n✓ Saved {len(health_df)} days of health data to data/raw/health_metrics.csv")
            print("\nRecent health metrics:")
            print(health_df.tail(10))
            
            # Show data quality summary
            print("\n=== Data Quality Summary ===")
            print(f"HRV data: {health_df['hrv_rmssd'].notna().sum()} days")
            print(f"Sleep data: {health_df['sleep_hours'].notna().sum()} days")
            print(f"Resting HR data: {health_df['resting_hr'].notna().sum()} days")
            print(f"Stress data: {health_df['stress_avg'].notna().sum()} days")
