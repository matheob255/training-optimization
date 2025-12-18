"""
Race day optimization with Monte Carlo simulation
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

class RaceOptimizer:
    """Optimize race day nutrition, pacing, and logistics"""
    
    def __init__(self):
        pass
    
    def calculate_wake_time(self, race_start_time, pre_race_routine_min=90):
        """Calculate optimal wake-up time"""
        # Reverse from race start
        race_dt = datetime.strptime(race_start_time, '%H:%M')
        wake_time = race_dt - timedelta(minutes=pre_race_routine_min)
        
        routine = {
            'wake': wake_time.strftime('%H:%M'),
            'breakfast': (wake_time + timedelta(minutes=15)).strftime('%H:%M'),
            'digestion_buffer': '2.5-3 hours',
            'arrive_venue': (race_dt - timedelta(minutes=45)).strftime('%H:%M'),
            'warmup_start': (race_dt - timedelta(minutes=25)).strftime('%H:%M'),
            'race_start': race_start_time
        }
        
        return routine
    
    def estimate_energy_needs(self, distance_km, duration_min, body_weight_kg=67):
        """Estimate calorie burn and glycogen needs"""
        # Rough estimate: ~1 kcal/kg/km for running
        calories_burned = body_weight_kg * distance_km
        
        # Glycogen stores: ~2000 kcal for trained athlete
        # Need fueling if race > 90 min or distance > 15km
        needs_fueling = duration_min > 90 or distance_km > 15
        
        if needs_fueling:
            # 30-60g carbs per hour
            hours = duration_min / 60
            carbs_needed_g = int(45 * hours)  # 45g/h middle ground
            gels_needed = int(carbs_needed_g / 25)  # ~25g per gel
        else:
            carbs_needed_g = 0
            gels_needed = 0
        
        return {
            'calories_burned': int(calories_burned),
            'needs_fueling': needs_fueling,
            'carbs_per_hour_g': int(carbs_needed_g / (duration_min/60)) if needs_fueling else 0,
            'total_gels': gels_needed,
            'gel_timing_min': [int((i+1) * (duration_min / (gels_needed + 1))) for i in range(gels_needed)] if gels_needed > 0 else []
        }
    
    def calculate_pacing_strategy(self, distance_km, target_time_min, elevation_gain_m=0):
        """Calculate optimal pacing (even, negative split, or elevation-adjusted)"""
        avg_pace_min_km = target_time_min / distance_km
        
        # Determine strategy based on elevation
        if elevation_gain_m > distance_km * 20:  # Hilly (>20m/km)
            strategy = 'elevation_adjusted'
            # Start conservative, faster on flats/downs
            first_half_pace = avg_pace_min_km * 1.05
            second_half_pace = avg_pace_min_km * 0.95
        elif distance_km >= 21:  # Long races: negative split
            strategy = 'negative_split'
            first_half_pace = avg_pace_min_km * 1.03
            second_half_pace = avg_pace_min_km * 0.97
        else:  # Short races: even pace
            strategy = 'even_pace'
            first_half_pace = avg_pace_min_km
            second_half_pace = avg_pace_min_km
        
        # Split by quarters for detail
        pacing_plan = {
            'strategy': strategy,
            'avg_pace_min_km': round(avg_pace_min_km, 2),
            'quarters': [
                {'portion': '0-25%', 'pace_min_km': round(first_half_pace * 1.02, 2), 'note': 'Settle in, controlled start'},
                {'portion': '25-50%', 'pace_min_km': round(first_half_pace, 2), 'note': 'Find rhythm'},
                {'portion': '50-75%', 'pace_min_km': round(second_half_pace, 2), 'note': 'Push effort'},
                {'portion': '75-100%', 'pace_min_km': round(second_half_pace * 0.98, 2), 'note': 'Empty the tank'}
            ]
        }
        
        return pacing_plan
    
    def warmup_protocol(self, distance_km):
        """Generate warmup routine based on race distance"""
        if distance_km <= 10:
            # Short races need more warmup
            warmup = {
                'duration_min': 20,
                'structure': [
                    '10 min easy jog',
                    '5 min dynamic stretches (leg swings, lunges)',
                    '4×100m strides @ race pace',
                    '1 min rest before start'
                ]
            }
        elif distance_km <= 21:
            warmup = {
                'duration_min': 15,
                'structure': [
                    '10 min easy jog',
                    '3 min dynamic stretches',
                    '2×100m strides'
                ]
            }
        else:
            # Long races: minimal warmup
            warmup = {
                'duration_min': 10,
                'structure': [
                    '8 min very easy jog',
                    '2 min dynamic stretches'
                ]
            }
        
        return warmup
    
    def monte_carlo_simulation(self, distance_km, target_time_min, elevation_gain_m, 
                                breakfast_time_before_h, gel_count, n_simulations=1000):
        """
        Monte Carlo simulation for race outcome
        
        Variables randomized:
        - Temperature (affects hydration needs)
        - Digestion time (individual variability)
        - Pacing execution (±3% variability)
        - Energy availability (glycogen stores 90-110%)
        - Hydration status (affects performance 0-3%)
        """
        results = {
            'finish_times': [],
            'bonk_risk': 0,
            'gi_distress': 0,
            'dehydration': 0,
            'success': 0  # Within ±2% of target time
        }
        
        for _ in range(n_simulations):
            # Randomize variables
            temp_c = np.random.uniform(8, 20)  # Temperature
            digestion_hours = np.random.uniform(2.0, 3.5)
            pacing_error_pct = np.random.normal(0, 3)  # Mean 0, std 3%
            glycogen_pct = np.random.uniform(90, 110)
            hydration_status = np.random.uniform(0, 3)  # 0-3% performance impact
            
            # Calculate finish time with variability
            base_time = target_time_min
            
            # Pacing error impact
            time_with_pacing = base_time * (1 + pacing_error_pct / 100)
            
            # Hydration impact (more in heat)
            if temp_c > 15:
                time_with_hydration = time_with_pacing * (1 + hydration_status / 100)
            else:
                time_with_hydration = time_with_pacing * (1 + hydration_status / 200)
            
            # Energy availability
            if distance_km > 15:  # Longer races affected by glycogen
                if glycogen_pct < 95 and gel_count == 0:
                    # Risk of bonking
                    results['bonk_risk'] += 1
                    time_with_energy = time_with_hydration * 1.15  # Slow down significantly
                else:
                    time_with_energy = time_with_hydration * (110 / glycogen_pct)
            else:
                time_with_energy = time_with_hydration
            
            # GI distress risk
            if breakfast_time_before_h < digestion_hours and distance_km > 10:
                if np.random.random() < 0.3:  # 30% chance if not enough time
                    results['gi_distress'] += 1
                    time_with_gi = time_with_energy * 1.08
                else:
                    time_with_gi = time_with_energy
            else:
                time_with_gi = time_with_energy
            
            # Dehydration check (simple)
            if temp_c > 18 and distance_km > 15:
                if np.random.random() < 0.2:
                    results['dehydration'] += 1
            
            final_time = time_with_gi
            results['finish_times'].append(final_time)
            
            # Success if within ±2% of target
            if abs(final_time - target_time_min) / target_time_min <= 0.02:
                results['success'] += 1
        
        # Convert to percentages
        results['bonk_risk_pct'] = (results['bonk_risk'] / n_simulations) * 100
        results['gi_distress_pct'] = (results['gi_distress'] / n_simulations) * 100
        results['dehydration_pct'] = (results['dehydration'] / n_simulations) * 100
        results['success_rate_pct'] = (results['success'] / n_simulations) * 100
        
        # Statistics
        results['mean_time'] = np.mean(results['finish_times'])
        results['std_time'] = np.std(results['finish_times'])
        results['p10_time'] = np.percentile(results['finish_times'], 10)
        results['p50_time'] = np.percentile(results['finish_times'], 50)
        results['p90_time'] = np.percentile(results['finish_times'], 90)
        
        return results
    
    def plot_monte_carlo_results(self, results):
        """Plot distribution of finish times from Monte Carlo"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=results['finish_times'],
            nbinsx=50,
            name='Finish Time Distribution',
            marker_color='lightblue'
        ))
        
        # Add median line
        fig.add_vline(
            x=results['p50_time'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {results['p50_time']:.1f} min"
        )
        
        fig.update_layout(
            title='Monte Carlo: Predicted Finish Time Distribution',
            xaxis_title='Finish Time (minutes)',
            yaxis_title='Frequency',
            height=350
        )
        
        return fig
