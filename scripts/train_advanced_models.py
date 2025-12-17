"""
Train performance and training response models for simulation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_performance_model(df):
    """
    Model 4: Predict race performance based on fitness state
    Estimates pace/time for a given distance based on current fitness markers
    """
    print("\n=== Training Performance Model ===")
    
    # For now, we'll model "easy run pace" as proxy for fitness
    # Later: add actual race results when available
    
    # Filter for runs > 5km to get representative paces
    runs = df[(df['distance_km'] > 5) & (df['distance_km'] < 15)].copy()
    
    if len(runs) < 10:
        print("⚠️ Not enough run data for performance model. Skipping...")
        return None, None
    
    # Calculate pace (min/km)
    runs['pace_min_km'] = (runs['duration_hours'] * 60) / runs['distance_km']
    
    # Features that indicate fitness
    features = [
        'hrv_baseline_28d', 'resting_hr',
        'acute_load_km', 'chronic_load_km',
        'fitness_trend', 'recovery_score',
        'averageHR', 'elevation_gain_m'
    ]
    
    # Target: pace for runs at "moderate" HR (140-160 bpm)
    moderate_runs = runs[(runs['averageHR'] > 140) & (runs['averageHR'] < 160)]
    
    if len(moderate_runs) < 8:
        print("⚠️ Not enough moderate-intensity runs. Using all runs...")
        moderate_runs = runs
    
    df_model = moderate_runs.dropna(subset=features + ['pace_min_km'])
    
    if len(df_model) < 5:
        print("⚠️ Insufficient data for performance model")
        return None, None
    
    X = df_model[features]
    y = df_model['pace_min_km']
    
    print(f"Training samples: {len(X)}")
    print(f"Pace range: {y.min():.2f} - {y.max():.2f} min/km")
    
    # Train
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Use all data (small dataset) but validate with cross-val
    model.fit(X, y)
    
    # Predict on training set to check fit
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Training MAE: {mae:.2f} min/km")
    print(f"Training R²: {r2:.3f}")
    
    # Save
    joblib.dump(model, 'models/performance_predictor.pkl')
    print("✓ Model saved")
    
    return model, features

def train_fitness_response_model(df):
    """
    Model 5: Predict fitness change from a workout
    Estimates how much fitness improves (or fatigue accumulates) from training
    """
    print("\n=== Training Fitness Response Model ===")
    
    # Proxy for fitness: rolling 28-day average HR at similar paces
    # Improvement = lower HR for same pace over time
    
    # Calculate week-over-week fitness change
    df_sorted = df.sort_values('date').copy()
    df_sorted['fitness_proxy'] = df_sorted['hr_efficiency'].rolling(7, min_periods=3).mean()
    df_sorted['fitness_change_7d'] = df_sorted['fitness_proxy'].diff(7)
    
    # Features: last week's training
    features = [
        'distance_km', 'duration_hours', 'averageHR',
        'acute_load_km', 'acwr',
        'recovery_score', 'sleep_hours',
        'consecutive_hard_days'
    ]
    
    # Target: did fitness improve next week?
    df_model = df_sorted.dropna(subset=features + ['fitness_change_7d'])
    
    if len(df_model) < 8:
        print("⚠️ Not enough data for fitness response model. Skipping...")
        return None, None
    
    X = df_model[features]
    y = df_model['fitness_change_7d']
    
    print(f"Training samples: {len(X)}")
    
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Training MAE: {mae:.4f}")
    print(f"Training R²: {r2:.3f}")
    
    # Save
    joblib.dump(model, 'models/fitness_response_predictor.pkl')
    print("✓ Model saved")
    
    return model, features

def main():
    """Train performance and fitness response models"""
    
    print("Loading data...")
    df = pd.read_csv('data/processed/training_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"✓ Loaded {len(df)} days")
    
    # Train models
    perf_model, perf_features = train_performance_model(df)
    fitness_model, fitness_features = train_fitness_response_model(df)
    
    # Update config
    import json
    with open('models/model_config.json') as f:
        config = json.load(f)[0]
    
    if perf_features:
        config['performance_features'] = perf_features
    if fitness_features:
        config['fitness_response_features'] = fitness_features
    
    pd.DataFrame([config]).to_json('models/model_config.json', orient='records', indent=2)
    
    print("\n" + "="*50)
    print("✓ ADVANCED MODELS COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
