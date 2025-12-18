"""
Comprehensive Garmin data analysis and visualization
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class GarminAnalyzer:
    """Analyze and visualize Garmin training data"""
    
    def __init__(self, df):
        """Initialize with processed Garmin dataframe"""
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
    
    def filter_data(self, start_date, end_date, metrics):
        """Filter data by date range and selected metrics"""
        mask = (self.df['date'] >= pd.to_datetime(start_date)) & \
               (self.df['date'] <= pd.to_datetime(end_date))
        filtered_df = self.df[mask].copy()
        
        # Drop unselected metrics
        all_metrics = {
            'hrv': ['hrv_rmssd', 'hrv_baseline_28d', 'hrv_deviation'],
            'sleep': ['sleep_hours', 'sleep_debt_7d'],
            'rhr': ['resting_hr', 'rhr_baseline_28d', 'rhr_deviation'],
            'stress': ['stress_avg'],
            'training': ['distance_km', 'duration_hours', 'averageHR', 'elevation_gain_m']
        }
        
        cols_to_keep = ['date']
        for metric_type, enabled in metrics.items():
            if enabled and metric_type in all_metrics:
                cols_to_keep.extend(all_metrics[metric_type])
        
        # Keep only available columns
        cols_to_keep = [c for c in cols_to_keep if c in filtered_df.columns]
        
        return filtered_df[cols_to_keep]
    
    def get_volume_stats(self, df=None):
        """Calculate training volume statistics"""
        if df is None:
            df = self.df
        
        stats = {
            'total_distance_km': df['distance_km'].sum(),
            'total_duration_hours': df['duration_hours'].sum() if 'duration_hours' in df else 0,
            'total_runs': (df['distance_km'] > 0).sum(),
            'avg_weekly_volume_4w': df.tail(28)['distance_km'].sum() / 4,
            'avg_weekly_volume_8w': df.tail(56)['distance_km'].sum() / 8,
            'avg_weekly_volume_12w': df.tail(84)['distance_km'].sum() / 12,
            'avg_weekly_volume_all': df['distance_km'].sum() / (len(df) / 7),
            'longest_run_km': df['distance_km'].max(),
            'avg_run_distance_km': df[df['distance_km'] > 0]['distance_km'].mean(),
            'runs_per_week': (df['distance_km'] > 0).sum() / (len(df) / 7)
        }
        
        return stats
    
    def get_intensity_distribution(self, df=None):
        """Calculate time in HR zones (approximation)"""
        if df is None:
            df = self.df
        
        # Rough estimation based on average HR
        # Z1: <60%, Z2: 60-70%, Z3: 70-80%, Z4: 80-90%, Z5: 90%+
        # Assume max HR ~200 for estimation
        
        zones = {
            'Z1_recovery': 0,
            'Z2_easy': 0,
            'Z3_tempo': 0,
            'Z4_threshold': 0,
            'Z5_max': 0
        }
        
        for _, row in df[df['distance_km'] > 0].iterrows():
            hr = row.get('averageHR', 0)
            duration = row.get('duration_hours', 0)
            
            if hr < 120:
                zones['Z1_recovery'] += duration
            elif hr < 140:
                zones['Z2_easy'] += duration
            elif hr < 165:
                zones['Z3_tempo'] += duration
            elif hr < 180:
                zones['Z4_threshold'] += duration
            else:
                zones['Z5_max'] += duration
        
        total = sum(zones.values())
        if total > 0:
            zones = {k: (v/total)*100 for k, v in zones.items()}
        
        return zones
    
    def get_physiological_trends(self, df=None):
        """Get trends in HRV, RHR, sleep, stress"""
        if df is None:
            df = self.df
        
        trends = {
            'hrv': {
                'mean': df['hrv_rmssd'].mean() if 'hrv_rmssd' in df else None,
                'std': df['hrv_rmssd'].std() if 'hrv_rmssd' in df else None,
                'min': df['hrv_rmssd'].min() if 'hrv_rmssd' in df else None,
                'max': df['hrv_rmssd'].max() if 'hrv_rmssd' in df else None,
                'current': df['hrv_rmssd'].iloc[-1] if 'hrv_rmssd' in df and len(df) > 0 else None
            },
            'rhr': {
                'mean': df['resting_hr'].mean() if 'resting_hr' in df else None,
                'current': df['resting_hr'].iloc[-1] if 'resting_hr' in df and len(df) > 0 else None
            },
            'sleep': {
                'mean': df['sleep_hours'].mean() if 'sleep_hours' in df else None,
                'debt_avg': df['sleep_debt_7d'].mean() if 'sleep_debt_7d' in df else None
            },
            'stress': {
                'mean': df['stress_avg'].mean() if 'stress_avg' in df else None
            }
        }
        
        return trends
    
    def plot_volume_trend(self, df=None):
        """Plot weekly volume trend"""
        if df is None:
            df = self.df
        
        # Aggregate by week
        df['week'] = df['date'].dt.to_period('W').dt.start_time
        weekly = df.groupby('week')['distance_km'].sum().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=weekly['week'],
            y=weekly['distance_km'],
            name='Weekly Volume',
            marker_color='lightblue'
        ))
        
        # Add trend line
        if len(weekly) > 4:
            z = np.polyfit(range(len(weekly)), weekly['distance_km'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=weekly['week'],
                y=p(range(len(weekly))),
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title='Weekly Training Volume',
            xaxis_title='Week',
            yaxis_title='Distance (km)',
            hovermode='x unified',
            height=350
        )
        
        return fig
    
    def plot_hrv_trend(self, df=None):
        """Plot HRV trend with baseline"""
        if df is None or 'hrv_rmssd' not in df.columns:
            return None
        
        df = df.copy()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['hrv_rmssd'],
            name='Daily HRV',
            mode='lines+markers',
            line=dict(color='purple', width=1),
            marker=dict(size=3)
        ))
        
        if 'hrv_baseline_28d' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['hrv_baseline_28d'],
                name='28-day Baseline',
                line=dict(color='orange', dash='dash')
            ))
        
        fig.update_layout(
            title='HRV Trend',
            xaxis_title='Date',
            yaxis_title='HRV (ms)',
            hovermode='x unified',
            height=300
        )
        
        return fig
    
    def plot_intensity_distribution(self, zones):
        """Plot HR zone distribution pie chart"""
        fig = go.Figure(data=[go.Pie(
            labels=list(zones.keys()),
            values=list(zones.values()),
            hole=0.3,
            marker=dict(colors=['#90EE90', '#87CEEB', '#FFD700', '#FF8C00', '#FF4500'])
        )])
        
        fig.update_layout(
            title='Training Intensity Distribution (Time in Zones)',
            height=350
        )
        
        return fig
    
    def plot_combined_health_metrics(self, df=None):
        """Plot sleep, stress, RHR together"""
        if df is None:
            df = self.df
        
        fig = go.Figure()
        
        if 'sleep_hours' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['sleep_hours'],
                name='Sleep (hours)',
                yaxis='y1',
                line=dict(color='blue')
            ))
        
        if 'stress_avg' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['stress_avg'],
                name='Stress',
                yaxis='y2',
                line=dict(color='red')
            ))
        
        if 'resting_hr' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['resting_hr'],
                name='Resting HR',
                yaxis='y3',
                line=dict(color='green')
            ))
        
        fig.update_layout(
            title='Health Metrics Overview',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Sleep (h)', side='left'),
            yaxis2=dict(title='Stress', overlaying='y', side='right', anchor='free', position=0.85),
            yaxis3=dict(title='RHR', overlaying='y', side='right'),
            hovermode='x unified',
            height=350
        )
        
        return fig
