"""
Collect training data from Garmin Connect
"""
import json
import pandas as pd
from garminconnect import Garmin
from datetime import datetime, timedelta

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
        print(f"✓ Downloaded {len(df)} activities")
        return df
    except Exception as e:
        print(f"✗ Failed to get activities: {e}")
        return None

def get_daily_health(client, date):
    """Get health metrics for a specific date"""
    try:
        # Convert date to string format 'YYYY-MM-DD'
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        sleep = client.get_sleep_data(date_str)
        hrv = client.get_hrv_data(date_str)
        stress = client.get_stress_data(date_str)
        
        return {
            'date': date_str,
            'sleep': sleep,
            'hrv': hrv,
            'stress': stress
        }
    except Exception as e:
        print(f"✗ Failed to get health data for {date_str}: {e}")
        return None


if __name__ == "__main__":
    # Test connection
    client = connect_garmin()
    
    if client:
        # Get recent activities
        activities_df = get_activities(client, 50)
        
        if activities_df is not None:
            # Save to CSV
            activities_df.to_csv('data/raw/activities.csv', index=False)
            print(f"✓ Saved to data/raw/activities.csv")
            
            # Show summary
            print("\nActivity summary:")
            print(activities_df[['activityName', 'distance', 'duration', 'averageHR']].head(10))
        
        # Get today's health data
        today = datetime.now().date()
        health = get_daily_health(client, today)
        if health:
            print(f"✓ Retrieved health data for {today}")
