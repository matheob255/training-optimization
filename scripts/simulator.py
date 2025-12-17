"""
Training plan simulator - Enhanced version with better physiological modeling
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

class TrainingSimulator:
    """Simulate training plans and predict outcomes"""
    
    def __init__(self):
        """Load all trained models"""
        self.recovery_model = joblib.load('models/recovery_classifier.pkl')
        self.hrv_model = joblib.load('models/hrv_predictor.pkl')
        self.injury_model = joblib.load('models/injury_risk_classifier.pkl')
        
        try:
            self.performance_model = joblib.load('models/performance_predictor.pkl')
            self.fitness_model = joblib.load('models/fitness_response_predictor.pkl')
        except:
            self.performance_model = None
            self.fitness_model = None
        
        with open('models/model_config.json') as f:
            self.config = json.load(f)[0]
    
    def simulate_week(self, current_state, weekly_km, intensity_dist, week_num):
        """Simulate one week with BOTH ML predictions AND rule-based checks"""
        warnings = []
        daily_states = []
        
        daily_km = self.distribute_weekly_volume(weekly_km, week_num)
        
        state = current_state.copy()
        prev_week_km = sum([s.get('last_week_km', weekly_km * 0.9) for s in [state]]) / len([state])
        
        for day in range(7):
            workout_km = daily_km[day]
            
            # HRV variation
            hrv_noise = np.random.normal(0, 3)
            state['hrv_rmssd'] = max(40, state.get('hrv_rmssd', 80) + hrv_noise)
            
            # === ML-BASED PREDICTIONS ===
            try:
                X_recovery = pd.DataFrame([{f: state.get(f, 0) for f in self.config['recovery_features']}])
                ready = self.recovery_model.predict(X_recovery)[0]
                ready_proba = self.recovery_model.predict_proba(X_recovery)[0][1]
            except:
                ready = 1
                ready_proba = 0.7
            
            try:
                X_injury = pd.DataFrame([{f: state.get(f, 0) for f in self.config['injury_features']}])
                injury_risk_ml = self.injury_model.predict_proba(X_injury)[0][1]
            except:
                injury_risk_ml = 0.1
            
            # === RULE-BASED RISK DETECTION (override ML when needed) ===
            acwr = state.get('acwr', 1.0)
            hrv_drop = (state.get('hrv_rmssd', 80) - state.get('hrv_baseline_28d', 85)) / state.get('hrv_baseline_28d', 85)
            consecutive_hard = state.get('consecutive_hard_days', 0)
            sleep_debt = state.get('sleep_debt_7d', 0)
            
            # Calculate injury risk from rules
            injury_risk_rules = 0.0
            
            if acwr > 1.5:
                injury_risk_rules += 0.50
            elif acwr > 1.3:
                injury_risk_rules += 0.25
            elif acwr > 1.1:
                injury_risk_rules += 0.10
            
            if hrv_drop < -0.15:
                injury_risk_rules += 0.20
            elif hrv_drop < -0.10:
                injury_risk_rules += 0.10
            
            if consecutive_hard >= 5:
                injury_risk_rules += 0.25
            elif consecutive_hard >= 3:
                injury_risk_rules += 0.10
            
            if sleep_debt > 7:
                injury_risk_rules += 0.15
            elif sleep_debt > 4:
                injury_risk_rules += 0.08
            
            # Use higher of ML or rules
            injury_risk = max(injury_risk_ml, min(injury_risk_rules, 0.95))
            
            # Generate warnings
            if injury_risk > 0.30:
                reasons = []
                if acwr > 1.3:
                    reasons.append(f"ACWR={acwr:.2f}")
                if hrv_drop < -0.10:
                    reasons.append(f"HRV {abs(hrv_drop)*100:.0f}% below baseline")
                if consecutive_hard >= 3:
                    reasons.append(f"{consecutive_hard} consecutive hard days")
                if sleep_debt > 4:
                    reasons.append(f"{sleep_debt:.1f}h sleep debt")
                
                reason_str = ", ".join(reasons) if reasons else "cumulative load"
                
                warnings.append({
                    'day': day + 1,
                    'type': 'injury_risk',
                    'value': injury_risk,
                    'severity': 'critical' if injury_risk > 0.5 else 'high' if injury_risk > 0.4 else 'medium',
                    'message': f'Injury risk {injury_risk*100:.0f}% ({reason_str})'
                })
            
            if not ready and workout_km > 10:
                warnings.append({
                    'day': day + 1,
                    'type': 'readiness',
                    'value': ready_proba,
                    'severity': 'medium',
                    'message': f'Low readiness for hard workout ({ready_proba*100:.0f}% ready)'
                })
            
            # Update state
            state = self.update_state_after_workout(state, workout_km, day)
            
            # Predict HRV
            try:
                X_hrv = pd.DataFrame([{f: state.get(f, 0) for f in self.config['hrv_features']}])
                predicted_hrv = self.hrv_model.predict(X_hrv)[0]
                state['hrv_rmssd'] = predicted_hrv
            except:
                pass
            
            daily_states.append(state.copy())
        
        # === WEEK-LEVEL CHECKS ===
        week_acwr = state.get('acwr', 1.0)
        week_increase_pct = (weekly_km - prev_week_km) / prev_week_km if prev_week_km > 0 else 0
        
        # ACWR warnings
        if week_acwr > 1.5:
            warnings.append({
                'type': 'acwr_critical',
                'value': week_acwr,
                'severity': 'critical',
                'message': f'🚨 ACWR {week_acwr:.2f} CRITICAL - very high injury risk'
            })
        elif week_acwr > 1.3:
            warnings.append({
                'type': 'acwr_high',
                'value': week_acwr,
                'severity': 'high',
                'message': f'ACWR {week_acwr:.2f} exceeds safe 1.3 threshold'
            })
        
        # Volume jump warning (>15% increase)
        if week_increase_pct > 0.15:
            warnings.append({
                'type': 'volume_jump',
                'value': week_increase_pct,
                'severity': 'high',
                'message': f'Volume increased {week_increase_pct*100:.0f}% (>15% unsafe)'
            })
        
        # Overreaching
        if state.get('consecutive_hard_days', 0) >= 5:
            warnings.append({
                'type': 'overreaching',
                'value': state.get('consecutive_hard_days', 0),
                'severity': 'high',
                'message': f'{state.get("consecutive_hard_days")} consecutive hard days - overtraining risk'
            })
        
        # Poor recovery trend
        if state.get('recovery_score', 0.7) < 0.4:
            warnings.append({
                'type': 'low_recovery',
                'value': state.get('recovery_score', 0.7),
                'severity': 'high',
                'message': f'Recovery critically low ({state.get("recovery_score", 0.7)*100:.0f}%)'
            })
        
        state['last_week_km'] = weekly_km
        
        return state, warnings, daily_states


    
    def distribute_weekly_volume(self, weekly_km, week_num):
        """Realistic volume distribution with variation"""
        # Week structure: Long run on Sunday, 2 quality days, easy days, 1-2 rest
        
        long_run_pct = 0.28 + np.random.uniform(-0.03, 0.03)
        quality1_pct = 0.18 + np.random.uniform(-0.02, 0.02)
        quality2_pct = 0.18 + np.random.uniform(-0.02, 0.02)
        easy1_pct = 0.14
        easy2_pct = 0.12
        easy3_pct = 0.10
        
        total_pct = long_run_pct + quality1_pct + quality2_pct + easy1_pct + easy2_pct + easy3_pct
        
        # Normalize
        long_run_pct /= total_pct
        quality1_pct /= total_pct
        quality2_pct /= total_pct
        easy1_pct /= total_pct
        easy2_pct /= total_pct
        easy3_pct /= total_pct
        
        # Mon: easy, Tue: quality, Wed: easy, Thu: quality, Fri: easy/rest, Sat: easy, Sun: long
        return [
            weekly_km * easy1_pct,      # Monday
            weekly_km * quality1_pct,    # Tuesday
            weekly_km * easy2_pct,       # Wednesday
            weekly_km * quality2_pct,    # Thursday
            weekly_km * easy3_pct,       # Friday
            0 if weekly_km > 50 else weekly_km * 0.1,  # Saturday - rest if high volume
            weekly_km * long_run_pct     # Sunday
        ]
    
    def update_state_after_workout(self, state, workout_km, day_of_week):
        """Enhanced state update with cumulative fatigue"""
        new_state = state.copy()
        
        # Update loads with proper rolling windows
        prev_acute = state.get('acute_load_km', 30)
        prev_chronic = state.get('chronic_load_km', 150)
        
        new_state['acute_load_km'] = prev_acute * 0.857 + workout_km  # Decays 1/7 per day
        new_state['chronic_load_km'] = prev_chronic * 0.964 + workout_km  # Decays 1/28 per day
        new_state['acwr'] = new_state['acute_load_km'] / (new_state['chronic_load_km'] / 4 + 1)
        
        # Fatigue accumulation (more realistic)
        current_recovery = state.get('recovery_score', 0.7)
        
        if workout_km > 15:  # Long run
            fatigue_hit = 0.20
            new_state['consecutive_hard_days'] = state.get('consecutive_hard_days', 0) + 1
        elif workout_km > 10:  # Quality/medium
            fatigue_hit = 0.12
            new_state['consecutive_hard_days'] = state.get('consecutive_hard_days', 0) + 1
        elif workout_km > 5:  # Easy
            fatigue_hit = 0.05
            if state.get('consecutive_hard_days', 0) > 0:
                new_state['consecutive_hard_days'] = state.get('consecutive_hard_days', 0) + 0.5
            else:
                new_state['consecutive_hard_days'] = 0
        else:  # Rest
            fatigue_hit = -0.15  # Recovery
            new_state['consecutive_hard_days'] = 0
        
        new_state['recovery_score'] = np.clip(current_recovery - fatigue_hit, 0.2, 1.0)
        
        # HRV response to load
        hrv_baseline = state.get('hrv_baseline_28d', 85)
        if workout_km > 12:
            new_state['hrv_rmssd'] = max(50, state.get('hrv_rmssd', 80) - np.random.uniform(3, 8))
        elif workout_km < 3:
            new_state['hrv_rmssd'] = min(110, state.get('hrv_rmssd', 80) + np.random.uniform(2, 6))
        
        # Sleep debt (simplified)
        new_state['sleep_hours'] = np.random.normal(7.2, 0.5)
        new_state['sleep_debt_7d'] = state.get('sleep_debt_7d', 0) * 0.857 + max(0, 7.5 - new_state['sleep_hours'])
        
        # Stress (higher after hard workouts)
        if workout_km > 12:
            new_state['stress_avg'] = min(60, state.get('stress_avg', 30) + 8)
        else:
            new_state['stress_avg'] = max(20, state.get('stress_avg', 30) - 2)
        
        return new_state
    
    def simulate_plan(self, start_state, weekly_km_plan, race_date=None):
        """Simulate entire training plan with detailed tracking"""
        state = start_state.copy()
        all_warnings = []
        weekly_states = []
        
        # Initialize tracking
        state['hrv_baseline_28d'] = start_state.get('hrv_rmssd', 80)
        state['rhr_baseline_28d'] = start_state.get('resting_hr', 45)
        
        print(f"\n🏃 Simulating {len(weekly_km_plan)}-week training plan...")
        print(f"Starting state: ACWR={state.get('acwr', 1.0):.2f}, HRV={state.get('hrv_rmssd', 80):.0f}ms, Recovery={state.get('recovery_score', 0.7)*100:.0f}%\n")
        
        for week_num, weekly_km in enumerate(weekly_km_plan, 1):
            print(f"  Week {week_num:2d}: {weekly_km:3.0f} km...", end='')
            
            state, warnings, daily = self.simulate_week(
                state, weekly_km, {'z2': 0.75, 'tempo': 0.15, 'intervals': 0.10}, week_num
            )
            
            week_warnings = len(warnings)
            critical = sum(1 for w in warnings if w.get('severity') == 'critical')
            high = sum(1 for w in warnings if w.get('severity') == 'high')
            
            if critical > 0:
                print(f" 🚨 {critical} CRITICAL warnings")
            elif high > 0:
                print(f" ⚠️  {high} high-severity warnings")
            elif week_warnings > 0:
                print(f" ⚡ {week_warnings} warnings")
            else:
                print(" ✓")
            
            all_warnings.extend([{**w, 'week': week_num} for w in warnings])
            weekly_states.append(state.copy())
        
        # Summary statistics
        injury_warnings = [w for w in all_warnings if w.get('type') == 'injury_risk']
        acwr_warnings = [w for w in all_warnings if 'acwr' in w.get('type', '')]
        
        max_injury_risk = max([w.get('value', 0) for w in injury_warnings] or [0])
        max_acwr = max([w.get('value', 0) for w in acwr_warnings] or [state.get('acwr', 1.0)])
        
        # Simple race prediction (placeholder until performance model has data)
        final_fitness = state.get('recovery_score', 0.7)
        predicted_time_seconds = 39 * 60 + 30 - (final_fitness - 0.7) * 180  # Very rough
        predicted_time = f"{int(predicted_time_seconds // 60)}:{int(predicted_time_seconds % 60):02d}"
        
        results = {
            'weekly_states': weekly_states,
            'warnings': all_warnings,
            'injury_warnings': len(injury_warnings),
            'acwr_warnings': len(acwr_warnings),
            'final_fitness': final_fitness,
            'final_acwr': state.get('acwr', 1.0),
            'max_injury_risk': max_injury_risk,
            'max_acwr': max_acwr,
            'predicted_race_time': predicted_time,
            'final_hrv': state.get('hrv_rmssd', 80),
            'peak_fitness_week': self.find_peak_week(weekly_states)
        }
        
        return results
    
    def find_peak_week(self, weekly_states):
        """Find week with best fitness:fatigue ratio"""
        fitness_scores = [s.get('recovery_score', 0.5) * (1 + s.get('acute_load_km', 30)/60) 
                          for s in weekly_states]
        return int(np.argmax(fitness_scores)) + 1

# Example usage
if __name__ == "__main__":
    sim = TrainingSimulator()
    
    # Current state
    start_state = {
        'hrv_rmssd': 79,
        'resting_hr': 43,
        'acute_load_km': 30,
        'chronic_load_km': 180,
        'acwr': 0.67,
        'recovery_score': 0.65,
        'consecutive_hard_days': 0,
        'sleep_hours': 7.0,
        'stress_avg': 37,
        'sleep_debt_7d': 2.0
    }
    
    # Test 1: Conservative plan
    print("\n" + "="*60)
    print("TEST 1: CONSERVATIVE PLAN (gradual build)")
    print("="*60)
    conservative_plan = [50, 52, 54, 56, 58, 58, 60, 58, 55, 52, 38, 22]
    results1 = sim.simulate_plan(start_state.copy(), conservative_plan)
    
    print("\n" + "="*60)
    print("RESULTS - Conservative Plan")
    print("="*60)
    print(f"Total warnings: {len(results1['warnings'])}")
    print(f"  - Injury risk warnings: {results1['injury_warnings']}")
    print(f"  - ACWR warnings: {results1['acwr_warnings']}")
    print(f"Max injury risk: {results1['max_injury_risk']*100:.1f}%")
    print(f"Max ACWR: {results1['max_acwr']:.2f}")
    print(f"Peak fitness week: {results1['peak_fitness_week']}")
    print(f"Final fitness: {results1['final_fitness']*100:.0f}%")
    print(f"Final HRV: {results1['final_hrv']:.0f} ms")
    print(f"Predicted 10k time: {results1['predicted_race_time']}")
    
    # Test 2: Aggressive plan
    print("\n\n" + "="*60)
    print("TEST 2: AGGRESSIVE PLAN (rapid build)")
    print("="*60)
    aggressive_plan = [50, 58, 65, 70, 72, 75, 72, 68, 62, 55, 42, 25]
    results2 = sim.simulate_plan(start_state.copy(), aggressive_plan)
    
    print("\n" + "="*60)
    print("RESULTS - Aggressive Plan")
    print("="*60)
    print(f"Total warnings: {len(results2['warnings'])}")
    print(f"  - Injury risk warnings: {results2['injury_warnings']}")
    print(f"  - ACWR warnings: {results2['acwr_warnings']}")
    print(f"Max injury risk: {results2['max_injury_risk']*100:.1f}%")
    print(f"Max ACWR: {results2['max_acwr']:.2f}")
    print(f"Peak fitness week: {results2['peak_fitness_week']}")
    print(f"Final fitness: {results2['final_fitness']*100:.0f}%")
    print(f"Final HRV: {results2['final_hrv']:.0f} ms")
    print(f"Predicted 10k time: {results2['predicted_race_time']}")
    
    # Show sample warnings
    if results2['warnings']:
        print("\n⚠️ Sample warnings from aggressive plan:")
        for w in results2['warnings'][:8]:
            severity_emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '⚡'}.get(w.get('severity'), '•')
            print(f"  {severity_emoji} Week {w['week']}: {w['message']}")
