"""
Preprocess Garmin data and create ML-ready features
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_merge_data():
    """Load activities and health data, merge on date"""
    
    # Load raw data
    activities = pd.read_csv('data/raw/activities.csv')
    health = pd.read_csv('data/raw/health_metrics.csv')
    
    # Parse dates
    activities['date'] = pd.to_datetime(activities['startTimeLocal']).dt.date
    health['date'] = pd.to_datetime(health['date']).dt.date
    
    # Aggregate activities by date (in case multiple per day)
    daily_activities = activities.groupby('date').agg({
        'distance': 'sum',
        'duration': 'sum',
        'averageHR': 'mean',
        'maxHR': 'max',
        'calories': 'sum',
        'elevationGain': 'sum'
    }).reset_index()
    
    # Convert to km and hours
    daily_activities['distance_km'] = daily_activities['distance'] / 1000
    daily_activities['duration_hours'] = daily_activities['duration'] / 3600
    daily_activities['elevation_gain_m'] = daily_activities['elevationGain']
    
    # Merge with health data
    df = pd.merge(health, daily_activities, on='date', how='outer')
    df = df.sort_values('date').reset_index(drop=True)
    
    # Fill NaN for rest days
    df['distance_km'] = df['distance_km'].fillna(0)
    df['duration_hours'] = df['duration_hours'].fillna(0)
    df['averageHR'] = df['averageHR'].fillna(0)
    df['elevation_gain_m'] = df['elevation_gain_m'].fillna(0)
    
    return df

def create_training_load_features(df):
    """Calculate acute/chronic load ratios and rolling metrics"""
    
    # Acute load (last 7 days)
    df['acute_load_km'] = df['distance_km'].rolling(7, min_periods=1).sum()
    
    # Chronic load (last 28 days)
    df['chronic_load_km'] = df['distance_km'].rolling(28, min_periods=7).sum()
    
    # ACWR (Acute:Chronic Workload Ratio)
    df['acwr'] = df['acute_load_km'] / (df['chronic_load_km'] / 4)
    df['acwr'] = df['acwr'].replace([np.inf, -np.inf], np.nan)
    df['acwr'] = df['acwr'].fillna(1.0)
    
    # 7-day rolling averages
    df['avg_hr_7d'] = df['averageHR'].rolling(7, min_periods=1).mean()
    df['avg_km_7d'] = df['distance_km'].rolling(7, min_periods=1).mean()
    
    # Training monotony (std dev of load)
    df['load_monotony'] = df['acute_load_km'] / (df['distance_km'].rolling(7, min_periods=3).std() + 1)
    
    return df

def create_recovery_features(df):
    """Features related to recovery status"""
    
    # HRV baseline and deviation
    df['hrv_baseline_28d'] = df['hrv_rmssd'].rolling(28, min_periods=7).mean()
    df['hrv_deviation'] = (df['hrv_rmssd'] - df['hrv_baseline_28d']) / df['hrv_baseline_28d']
    
    # Resting HR baseline and deviation
    df['rhr_baseline_28d'] = df['resting_hr'].rolling(28, min_periods=7).mean()
    df['rhr_deviation'] = (df['resting_hr'] - df['rhr_baseline_28d']) / df['rhr_baseline_28d']
    
    # Sleep metrics
    df['sleep_deficit'] = 7.5 - df['sleep_hours']  # Assuming 7.5h target
    df['sleep_debt_7d'] = df['sleep_deficit'].rolling(7, min_periods=1).sum()
    
    # Consecutive hard days (HR > 150 or duration > 1h)
    df['hard_day'] = ((df['averageHR'] > 150) | (df['duration_hours'] > 1.0)).astype(int)
    df['consecutive_hard_days'] = df['hard_day'].rolling(3, min_periods=1).sum()
    
    # Recovery score (composite metric)
    df['recovery_score'] = (
        (1 - df['hrv_deviation'].fillna(0).clip(-0.5, 0.5)) * 0.4 +
        (df['sleep_hours'].fillna(7) / 8) * 0.3 +
        ((100 - df['stress_avg'].fillna(30)) / 100) * 0.2 +
        (1 - df['rhr_deviation'].fillna(0).clip(-0.2, 0.2)) * 0.1
    )
    
    return df

def create_lagged_features(df):
    """Create yesterday's metrics as features for tomorrow's prediction"""
    
    # Yesterday's metrics
    df['hrv_yesterday'] = df['hrv_rmssd'].shift(1)
    df['sleep_yesterday'] = df['sleep_hours'].shift(1)
    df['rhr_yesterday'] = df['resting_hr'].shift(1)
    df['km_yesterday'] = df['distance_km'].shift(1)
    df['recovery_yesterday'] = df['recovery_score'].shift(1)
    
    # Tomorrow's metrics (for training targets)
    df['hrv_tomorrow'] = df['hrv_rmssd'].shift(-1)
    df['need_rest_tomorrow'] = ((df['recovery_score'].shift(-1) < 0.6) | 
                                 (df['distance_km'].shift(-1) == 0)).astype(int)
    
    return df

def create_performance_features(df):
    """Features for performance modeling"""
    
    # Running efficiency (pace per HR beat)
    df['pace_min_per_km'] = df['duration_hours'] * 60 / (df['distance_km'] + 0.1)
    df['hr_efficiency'] = df['pace_min_per_km'] / (df['averageHR'] + 1)
    
    # VO2max proxy (very rough estimate from HR and pace)
    # Better: use Garmin's VO2max if available
    df['vo2max_proxy'] = 15.3 * (df['maxHR'] / df['resting_hr'].fillna(50))
    
    # Fitness trend (7-day rolling efficiency)
    df['fitness_trend'] = df['hr_efficiency'].rolling(7, min_periods=3).mean()
    
    return df

def prepare_dataset():
    """Main preprocessing pipeline"""
    
    print("Loading data...")
    df = load_and_merge_data()
    print(f"✓ Loaded {len(df)} days of data")
    
    print("Creating features...")
    df = create_training_load_features(df)
    df = create_recovery_features(df)
    df = create_lagged_features(df)
    df = create_performance_features(df)
    
    # Drop rows with too many NaNs (early rows with insufficient rolling history)
    df_clean = df.dropna(subset=['acwr', 'hrv_baseline_28d', 'recovery_score'])
    
    print(f"✓ Feature engineering complete: {len(df_clean)} usable days")
    
    # Save processed data
    df.to_csv('data/processed/master_dataset.csv', index=False)
    df_clean.to_csv('data/processed/training_dataset.csv', index=False)
    
    print("✓ Saved to data/processed/")
    
    # Summary
    print("\n=== Dataset Summary ===")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total training days: {(df['distance_km'] > 0).sum()}")
    print(f"Total distance: {df['distance_km'].sum():.1f} km")
    print(f"Avg weekly volume: {df['distance_km'].sum() / (len(df) / 7):.1f} km")
    print(f"\nKey features created: {len(df.columns)}")
    
    return df_clean

if __name__ == "__main__":
    df = prepare_dataset()
    print("\n✓ Preprocessing complete!")
    print("\nSample of processed data:")
    print(df[['date', 'distance_km', 'hrv_rmssd', 'recovery_score', 'acwr']].tail(10))
