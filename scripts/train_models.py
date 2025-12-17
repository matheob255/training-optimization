"""
Train foundational ML models for training simulation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_recovery_model(df):
    """
    Model 1: Predict if tomorrow will be a good training day
    Binary classification: 1 = ready for hard workout, 0 = need easy/rest
    """
    print("\n=== Training Recovery Model ===")
    
    # Define features
    features = [
        'hrv_rmssd', 'hrv_deviation', 'hrv_baseline_28d',
        'resting_hr', 'rhr_deviation',
        'sleep_hours', 'sleep_debt_7d',
        'stress_avg',
        'acute_load_km', 'acwr', 'load_monotony',
        'consecutive_hard_days',
        'recovery_score'
    ]
    
    # Target: is tomorrow a good day for hard training?
    # Define as: recovery_score > 0.7 AND HRV within 10% of baseline
    df['ready_for_hard_workout'] = (
        (df['recovery_score'] > 0.7) & 
        (df['hrv_deviation'] > -0.10)
    ).astype(int)
    
    # Prepare data
    df_model = df.dropna(subset=features + ['ready_for_hard_workout'])
    X = df_model[features]
    y = df_model['ready_for_hard_workout']
    
    print(f"Training samples: {len(X)}")
    print(f"Positive class (ready): {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, target_names=['Rest/Easy', 'Ready for Hard']))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"Cross-val F1 score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(importance_df.head())
    
    # Save model
    joblib.dump(model, 'models/recovery_classifier.pkl')
    importance_df.to_csv('models/recovery_feature_importance.csv', index=False)
    print("✓ Model saved to models/recovery_classifier.pkl")
    
    return model, features

def train_hrv_predictor(df):
    """
    Model 2: Predict tomorrow's HRV (regression)
    Used to forecast recovery trajectory
    """
    print("\n=== Training HRV Predictor ===")
    
    features = [
        'hrv_yesterday', 'hrv_baseline_28d',
        'sleep_yesterday', 'sleep_debt_7d',
        'km_yesterday', 'acute_load_km', 'acwr',
        'resting_hr', 'stress_avg'
    ]
    
    df_model = df.dropna(subset=features + ['hrv_tomorrow'])
    X = df_model[features]
    y = df_model['hrv_tomorrow']
    
    print(f"Training samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MAE: {mae:.2f} ms")
    print(f"Test R²: {r2:.3f}")
    
    # Save
    joblib.dump(model, 'models/hrv_predictor.pkl')
    print("✓ Model saved to models/hrv_predictor.pkl")
    
    return model, features

def train_injury_risk_model(df):
    """
    Model 3: Predict injury risk (binary: high risk yes/no)
    """
    print("\n=== Training Injury Risk Model ===")
    
    features = [
        'acwr', 'load_monotony',
        'consecutive_hard_days',
        'hrv_deviation', 'rhr_deviation',
        'sleep_debt_7d',
        'acute_load_km'
    ]
    
    # Define high injury risk criteria (proxy, since we don't have actual injury data yet)
    # Risk indicators: ACWR > 1.3, HRV drop > 15%, consecutive hard days > 2, sleep debt > 5h
    df['high_injury_risk'] = (
        (df['acwr'] > 1.3) | 
        (df['hrv_deviation'] < -0.15) |
        (df['consecutive_hard_days'] >= 3) |
        (df['sleep_debt_7d'] > 5)
    ).astype(int)
    
    df_model = df.dropna(subset=features + ['high_injury_risk'])
    X = df_model[features]
    y = df_model['high_injury_risk']
    
    print(f"Training samples: {len(X)}")
    print(f"High risk days: {y.sum()} ({y.mean()*100:.1f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nTest Performance:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    # Save
    joblib.dump(model, 'models/injury_risk_classifier.pkl')
    print("✓ Model saved to models/injury_risk_classifier.pkl")
    
    return model, features

def main():
    """Train all foundational models"""
    
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('data/processed/training_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"✓ Loaded {len(df)} days")
    
    # Train models
    recovery_model, recovery_features = train_recovery_model(df)
    hrv_model, hrv_features = train_hrv_predictor(df)
    injury_model, injury_features = train_injury_risk_model(df)
    
    # Save feature lists
    model_config = {
        'recovery_features': recovery_features,
        'hrv_features': hrv_features,
        'injury_features': injury_features
    }
    
    pd.DataFrame([model_config]).to_json('models/model_config.json', orient='records', indent=2)
    
    print("\n" + "="*50)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*50)
    print("\nModels saved in models/:")
    print("  - recovery_classifier.pkl")
    print("  - hrv_predictor.pkl")
    print("  - injury_risk_classifier.pkl")
    print("  - model_config.json")

if __name__ == "__main__":
    main()
