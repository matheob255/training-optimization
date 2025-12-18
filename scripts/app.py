"""
Training Plan Simulator - Complete Mobile-Optimized Application
Version 2.0 - Full Featured with Race Day Optimization
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
sys.path.append('scripts')
from simulator import TrainingSimulator
from garmin_analyzer import GarminAnalyzer
from race_optimizer import RaceOptimizer

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Training Simulator Pro",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AI-powered training optimization with race day planning"
    }
)

# Mobile-responsive CSS
st.markdown("""
<style>
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        .stButton button {
            width: 100%;
            margin-top: 0.5rem;
        }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.1rem !important; }
        .stMetric {
            background-color: #f0f2f6;
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
    }
    .stButton button {
        min-height: 44px;
        font-size: 1rem;
    }
    .element-container {
        margin-bottom: 0.5rem;
    }
    .wizard-step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_garmin_data():
    """Load processed Garmin data"""
    try:
        df = pd.read_csv('data/processed/training_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        return None

def get_current_state(df):
    """Extract current fitness state from latest data"""
    if df is None or len(df) == 0:
        return None
    
    latest = df.iloc[-1]
    last_7_days = df.tail(7)
    
    state = {
        'date': latest['date'].strftime('%Y-%m-%d'),
        'hrv_rmssd': float(latest['hrv_rmssd']) if pd.notna(latest['hrv_rmssd']) else 80,
        'resting_hr': float(latest['resting_hr']) if pd.notna(latest['resting_hr']) else 45,
        'acute_load_km': float(latest['acute_load_km']) if pd.notna(latest['acute_load_km']) else 30,
        'chronic_load_km': float(latest['chronic_load_km']) if pd.notna(latest['chronic_load_km']) else 150,
        'acwr': float(latest['acwr']) if pd.notna(latest['acwr']) else 1.0,
        'recovery_score': float(latest['recovery_score']) if pd.notna(latest['recovery_score']) else 0.7,
        'consecutive_hard_days': int(latest.get('consecutive_hard_days', 0)) if pd.notna(latest.get('consecutive_hard_days', 0)) else 0,
        'sleep_hours': float(latest['sleep_hours']) if pd.notna(latest['sleep_hours']) else 7.0,
        'stress_avg': float(latest['stress_avg']) if pd.notna(latest['stress_avg']) else 30,
        'sleep_debt_7d': float(latest['sleep_debt_7d']) if pd.notna(latest['sleep_debt_7d']) else 1.0,
        'weekly_km_avg': float(last_7_days['distance_km'].sum()) if len(last_7_days) > 0 else 40
    }
    
    return state

def format_time(minutes):
    """Convert minutes to MM:SS format"""
    mins = int(minutes)
    secs = int((minutes - mins) * 60)
    return f"{mins}:{secs:02d}"

def parse_time(time_str):
    """Parse MM:SS to minutes"""
    try:
        parts = time_str.split(':')
        return int(parts[0]) + int(parts[1]) / 60
    except:
        return 40.0

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'simulator' not in st.session_state:
    st.session_state.simulator = TrainingSimulator()

if 'race_optimizer' not in st.session_state:
    st.session_state.race_optimizer = RaceOptimizer()

if 'wizard_completed' not in st.session_state:
    st.session_state.wizard_completed = False

if 'current_wizard_step' not in st.session_state:
    st.session_state.current_wizard_step = 1

if 'data_filters' not in st.session_state:
    st.session_state.data_filters = {
        'start_date': datetime.now() - timedelta(days=90),
        'end_date': datetime.now(),
        'metrics': {
            'hrv': True,
            'sleep': True,
            'rhr': True,
            'stress': True,
            'training': True
        }
    }

# Load data
garmin_df = load_garmin_data()
current_state_auto = get_current_state(garmin_df)

# ============================================================================
# HEADER
# ============================================================================

st.title("🏃 Training Simulator Pro")
st.caption("AI-powered training optimization with race day planning")

# ============================================================================
# WIZARD WORKFLOW (First-time users)
# ============================================================================

if not st.session_state.wizard_completed:
    st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
    st.markdown("### 🧙‍♂️ Welcome! Let's set up your training plan")
    st.markdown(f"**Step {st.session_state.current_wizard_step} of 4**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.current_wizard_step == 1:
        st.subheader("Step 1: Import Your Garmin Data")
        
        if garmin_df is not None:
            st.success(f"✅ Found {len(garmin_df)} days of Garmin data!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Distance", f"{garmin_df['distance_km'].sum():.0f} km")
            with col2:
                st.metric("Training Days", f"{(garmin_df['distance_km'] > 0).sum()}")
            
            if st.button("✅ Use This Data", type="primary", use_container_width=True):
                st.session_state.current_wizard_step = 2
                st.rerun()
        else:
            st.warning("⚠️ No Garmin data found")
            st.info("""
            **To import your data:**
            1. Run `python scripts/data_collection.py`
            2. Then `python scripts/preprocessing.py`
            3. Or upload a CSV file below
            """)
            
            uploaded_file = st.file_uploader("Upload training data (CSV)", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df
                st.success("✅ Data uploaded!")
                if st.button("Continue →", type="primary"):
                    st.session_state.current_wizard_step = 2
                    st.rerun()
        
        if st.button("Skip wizard (use defaults)", type="secondary"):
            st.session_state.wizard_completed = True
            st.rerun()
    
    elif st.session_state.current_wizard_step == 2:
        st.subheader("Step 2: Set Your Race Goal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            race_distance = st.number_input("Race Distance (km)", value=10.0, min_value=1.0, max_value=100.0, step=0.1)
            elevation_gain = st.number_input("Elevation Gain (m)", value=0, min_value=0, max_value=5000, step=50)
        
        with col2:
            race_date = st.date_input("Race Date", value=datetime.now() + timedelta(weeks=12))
            target_time = st.text_input("Target Time (MM:SS)", value="39:30")
        
        terrain_type = st.selectbox("Terrain Type", ["Road", "Trail - Easy", "Trail - Technical", "Mountain/Ultra", "Track"])
        
        st.session_state.wizard_race = {
            'distance_km': race_distance,
            'elevation_gain_m': elevation_gain,
            'race_date': race_date,
            'target_time': target_time,
            'terrain_type': terrain_type
        }
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.current_wizard_step = 1
                st.rerun()
        with col2:
            if st.button("Continue →", type="primary"):
                st.session_state.current_wizard_step = 3
                st.rerun()
    
    elif st.session_state.current_wizard_step == 3:
        st.subheader("Step 3: Training Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            runs_per_week = st.slider("Runs per Week", min_value=3, max_value=7, value=4)
            strength_sessions = st.slider("Strength Sessions per Week", min_value=0, max_value=4, value=1)
        
        with col2:
            if current_state_auto:
                current_weekly = current_state_auto['weekly_km_avg']
            else:
                current_weekly = st.number_input("Current Weekly Volume (km)", value=40.0, min_value=10.0, max_value=150.0)
        
        st.session_state.wizard_prefs = {
            'runs_per_week': runs_per_week,
            'strength_sessions': strength_sessions,
            'current_weekly_km': current_weekly
        }
        
        # Guidance based on race goal
        weeks_to_race = max(4, (st.session_state.wizard_race['race_date'] - datetime.now().date()).days // 7)
        race_km = st.session_state.wizard_race['distance_km']
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**🤖 AI Guidance:**")
        
        if runs_per_week < 4 and race_km >= 21:
            st.warning(f"⚠️ For a {race_km}km race, 4-5 runs/week is recommended for adequate preparation.")
        elif runs_per_week >= 5:
            st.success(f"✅ {runs_per_week} runs/week is excellent for your {race_km}km goal!")
        else:
            st.info(f"✓ {runs_per_week} runs/week is adequate for {race_km}km if quality is high.")
        
        if strength_sessions == 0 and weeks_to_race > 8:
            st.info("💡 Consider adding 1-2 strength sessions to reduce injury risk.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.current_wizard_step = 2
                st.rerun()
        with col2:
            if st.button("Continue →", type="primary"):
                st.session_state.current_wizard_step = 4
                st.rerun()
    
    elif st.session_state.current_wizard_step == 4:
        st.subheader("Step 4: Review & Generate Plan")
        
        st.markdown("**Your Configuration:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Race Goal**")
            st.write(f"📍 {st.session_state.wizard_race['distance_km']} km")
            st.write(f"📅 {st.session_state.wizard_race['race_date']}")
            st.write(f"⏱️ Target: {st.session_state.wizard_race['target_time']}")
        
        with col2:
            st.markdown("**Training**")
            st.write(f"🏃 {st.session_state.wizard_prefs['runs_per_week']} runs/week")
            st.write(f"💪 {st.session_state.wizard_prefs['strength_sessions']} strength/week")
            st.write(f"📊 Current: {st.session_state.wizard_prefs['current_weekly_km']:.0f} km/week")
        
        with col3:
            st.markdown("**Data Source**")
            if garmin_df is not None:
                st.write(f"✅ Garmin: {len(garmin_df)} days")
            else:
                st.write("⚠️ Manual inputs")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.current_wizard_step = 3
                st.rerun()
        with col2:
            if st.button("🚀 Complete Setup", type="primary", use_container_width=True):
                st.session_state.wizard_completed = True
                st.balloons()
                st.rerun()

else:
    # ============================================================================
    # MAIN APPLICATION (After wizard)
    # ============================================================================
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Garmin Dashboard",
        "🎯 Plan Setup",
        "📈 Simulation Results",
        "🏁 Race Day Optimizer",
        "⚙️ Settings"
    ])
    
    # ========================================================================
    # TAB 1: GARMIN DASHBOARD
    # ========================================================================
    
    with tab1:
        st.header("📊 Garmin Data Analysis")
        
        if garmin_df is None:
            st.warning("No Garmin data loaded. Upload data in Settings tab.")
        else:
            analyzer = GarminAnalyzer(garmin_df)
            
            # Interactive filters
            st.subheader("📅 Data Filters")
            
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                filter_preset = st.selectbox(
                    "Quick Filter",
                    ["Last 30 days", "Last 3 months", "Last 6 months", "Last year", "All time", "Custom"]
                )
                
                if filter_preset == "Last 30 days":
                    start_date = datetime.now() - timedelta(days=30)
                    end_date = datetime.now()
                elif filter_preset == "Last 3 months":
                    start_date = datetime.now() - timedelta(days=90)
                    end_date = datetime.now()
                elif filter_preset == "Last 6 months":
                    start_date = datetime.now() - timedelta(days=180)
                    end_date = datetime.now()
                elif filter_preset == "Last year":
                    start_date = datetime.now() - timedelta(days=365)
                    end_date = datetime.now()
                elif filter_preset == "All time":
                    start_date = garmin_df['date'].min()
                    end_date = garmin_df['date'].max()
                else:  # Custom
                    col_a, col_b = st.columns(2)
                    with col_a:
                        start_date = st.date_input("From", value=datetime.now() - timedelta(days=90))
                    with col_b:
                        end_date = st.date_input("To", value=datetime.now())
            
            with col2:
                st.markdown("**Include Metrics:**")
                metric_hrv = st.checkbox("HRV", value=True, key="metric_hrv")
                metric_sleep = st.checkbox("Sleep", value=True, key="metric_sleep")
            
            with col3:
                metric_rhr = st.checkbox("Resting HR", value=True, key="metric_rhr")
                metric_stress = st.checkbox("Stress", value=True, key="metric_stress")
                metric_training = st.checkbox("Training", value=True, key="metric_training")
            
            # Apply filters
            metrics_dict = {
                'hrv': metric_hrv,
                'sleep': metric_sleep,
                'rhr': metric_rhr,
                'stress': metric_stress,
                'training': metric_training
            }
            
            filtered_df = analyzer.filter_data(start_date, end_date, metrics_dict)
            
            st.caption(f"📊 Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({len(filtered_df)} days)")
            
            # ----------------------------------------------------------------
            # Volume Statistics
            # ----------------------------------------------------------------
            
            st.subheader("🏃 Training Volume")
            
            volume_stats = analyzer.get_volume_stats(filtered_df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Distance", f"{volume_stats['total_distance_km']:.0f} km")
                st.metric("Total Runs", f"{volume_stats['total_runs']}")
            
            with col2:
                st.metric("Avg Weekly (4w)", f"{volume_stats['avg_weekly_volume_4w']:.0f} km")
                st.metric("Avg Weekly (12w)", f"{volume_stats['avg_weekly_volume_12w']:.0f} km")
            
            with col3:
                st.metric("Longest Run", f"{volume_stats['longest_run_km']:.1f} km")
                st.metric("Avg Run Distance", f"{volume_stats['avg_run_distance_km']:.1f} km")
            
            with col4:
                st.metric("Runs per Week", f"{volume_stats['runs_per_week']:.1f}")
                st.metric("Total Hours", f"{volume_stats['total_duration_hours']:.0f} h")
            
            # Volume trend chart
            fig_volume = analyzer.plot_volume_trend(filtered_df)
            if fig_volume:
                st.plotly_chart(fig_volume, use_container_width=True)
            
            # ----------------------------------------------------------------
            # Intensity Distribution
            # ----------------------------------------------------------------
            
            st.subheader("⚡ Intensity Distribution")
            
            zones = analyzer.get_intensity_distribution(filtered_df)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Time in Zones:**")
                for zone, pct in zones.items():
                    st.write(f"{zone}: {pct:.1f}%")
            
            with col2:
                fig_zones = analyzer.plot_intensity_distribution(zones)
                st.plotly_chart(fig_zones, use_container_width=True)
            
            # ----------------------------------------------------------------
            # Physiological Trends
            # ----------------------------------------------------------------
            
            st.subheader("💪 Health & Recovery Metrics")
            
            trends = analyzer.get_physiological_trends(filtered_df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if trends['hrv']['mean']:
                    st.metric(
                        "HRV (avg)", 
                        f"{trends['hrv']['mean']:.0f} ms",
                        delta=f"{trends['hrv']['current'] - trends['hrv']['mean']:.0f} ms" if trends['hrv']['current'] else None
                    )
            
            with col2:
                if trends['rhr']['mean']:
                    st.metric("Resting HR (avg)", f"{trends['rhr']['mean']:.0f} bpm")
            
            with col3:
                if trends['sleep']['mean']:
                    st.metric("Sleep (avg)", f"{trends['sleep']['mean']:.1f} h")
            
            with col4:
                if trends['stress']['mean']:
                    st.metric("Stress (avg)", f"{trends['stress']['mean']:.0f}")
            
            # HRV trend
            if 'hrv_rmssd' in filtered_df.columns:
                fig_hrv = analyzer.plot_hrv_trend(filtered_df)
                if fig_hrv:
                    st.plotly_chart(fig_hrv, use_container_width=True)
            
            # Combined health metrics
            fig_health = analyzer.plot_combined_health_metrics(filtered_df)
            if fig_health:
                st.plotly_chart(fig_health, use_container_width=True)
            
            # ----------------------------------------------------------------
            # Raw Data Table
            # ----------------------------------------------------------------
            
            with st.expander("📋 View Raw Data"):
                st.dataframe(
                    filtered_df.sort_values('date', ascending=False).head(50),
                    use_container_width=True,
                    height=400
                )
                
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Filtered Data (CSV)",
                    data=csv,
                    file_name=f"garmin_data_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
    
    # ========================================================================
    # TAB 2: PLAN SETUP
    # ========================================================================
    
    with tab2:
        st.header("🎯 Training Plan Configuration")
        
        # Race Configuration
        st.subheader("🏁 Race Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            race_distance_km = st.number_input("Distance (km)", value=10.0, min_value=1.0, max_value=100.0, step=0.5)
            elevation_gain_m = st.number_input("Elevation Gain (m)", value=0, min_value=0, max_value=5000, step=50)
        
        with col2:
            race_date = st.date_input("Race Date", value=datetime.now() + timedelta(weeks=12))
            target_time_str = st.text_input("Target Time (MM:SS)", value="39:30")
        
        with col3:
            terrain_type = st.selectbox(
                "Terrain Type",
                ["Road", "Trail - Easy", "Trail - Technical", "Mountain/Ultra", "Track"]
            )
            
            weeks_to_race = max(4, min(24, (race_date - datetime.now().date()).days // 7))
            st.metric("Weeks to Race", weeks_to_race)
        
        # Current Fitness
        st.subheader("💪 Current Fitness")
        
        use_garmin = st.checkbox("Auto-fill from Garmin data", value=current_state_auto is not None)
        
        col1, col2, col3 = st.columns(3)
        
        if use_garmin and current_state_auto:
            with col1:
                current_weekly_km = st.number_input(
                    "Current Weekly km",
                    value=float(current_state_auto['weekly_km_avg']),
                    min_value=10.0,
                    max_value=150.0,
                    step=5.0
                )
            with col2:
                current_hrv = st.number_input(
                    "Recent HRV (ms)",
                    value=float(current_state_auto['hrv_rmssd']),
                    min_value=30.0,
                    max_value=120.0,
                    step=1.0
                )
            with col3:
                current_rhr = st.number_input(
                    "Resting HR (bpm)",
                    value=float(current_state_auto['resting_hr']),
                    min_value=35.0,
                    max_value=80.0,
                    step=1.0
                )
            
            start_state = current_state_auto.copy()
            start_state['hrv_rmssd'] = current_hrv
            start_state['resting_hr'] = current_rhr
            start_state['acute_load_km'] = current_weekly_km
        
        else:
            with col1:
                current_weekly_km = st.number_input("Current Weekly km", value=45.0, min_value=10.0, max_value=150.0, step=5.0)
            with col2:
                current_hrv = st.number_input("Recent HRV (ms)", value=79.0, min_value=30.0, max_value=120.0, step=1.0)
            with col3:
                current_rhr = st.number_input("Resting HR (bpm)", value=43.0, min_value=35.0, max_value=80.0, step=1.0)
            
            start_state = {
                'hrv_rmssd': current_hrv,
                'resting_hr': current_rhr,
                'acute_load_km': current_weekly_km,
                'chronic_load_km': current_weekly_km * 4,
                'acwr': 1.0,
                'recovery_score': 0.70,
                'consecutive_hard_days': 0,
                'sleep_hours': 7.0,
                'stress_avg': 30,
                'sleep_debt_7d': 1.0
            }
        
        # Training Preferences
        st.subheader("📋 Training Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            runs_per_week = st.slider("Runs per Week", min_value=3, max_value=7, value=4)
            strength_sessions_per_week = st.slider("Strength Sessions per Week", min_value=0, max_value=4, value=1)
        
        with col2:
            st.info(f"""
            **Intensity distribution will be auto-generated based on:**
            - Race distance: {race_distance_km} km
            - Time to race: {weeks_to_race} weeks
            - Current volume: {current_weekly_km:.0f} km/week
            
            Typical split:
            - Easy (Z2): 70-80%
            - Tempo (Z3): 10-15%
            - Intervals (Z4-5): 5-10%
            """)
        
        # AI Guidance
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**🤖 AI Training Guidance:**")
        
        # Check if runs per week is adequate
        if runs_per_week < 4 and race_distance_km >= 21:
            st.warning(f"⚠️ For a {race_distance_km}km race, 4-5 runs/week is recommended. Consider increasing frequency.")
        elif runs_per_week >= 5:
            st.success(f"✅ {runs_per_week} runs/week is excellent preparation for {race_distance_km}km!")
        else:
            st.info(f"✓ {runs_per_week} runs/week is adequate if quality and consistency are high.")
        
        # Check strength training
        if strength_sessions_per_week == 0 and weeks_to_race > 8:
            st.info("💡 Adding 1-2 strength sessions can reduce injury risk by 30-50%.")
        elif strength_sessions_per_week >= 2:
            st.success("✅ Good strength training volume!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plan Generation
        st.subheader("📅 Plan Generation")
        
        plan_mode = st.radio("Plan Mode", ["Auto-generate (Recommended)", "Custom Weekly Volumes"], horizontal=True)
        
        if plan_mode == "Auto-generate (Recommended)":
            col1, col2 = st.columns(2)
            
            with col1:
                peak_volume_km = st.slider(
                    "Peak Weekly Volume (km)",
                    min_value=int(current_weekly_km),
                    max_value=min(150, int(current_weekly_km * 2)),
                    value=int(current_weekly_km * 1.3)
                )
            
            with col2:
                taper_weeks = st.slider("Taper Duration (weeks)", min_value=1, max_value=3, value=2)
            
            # Generate progressive plan
            build_weeks = weeks_to_race - taper_weeks
            weekly_volumes = []
            
            for i in range(build_weeks):
                # Smooth progression with some variation
                base_progression = current_weekly_km + (peak_volume_km - current_weekly_km) * (i / build_weeks)
                # Add slight wave (3 weeks up, 1 week down pattern)
                if (i + 1) % 4 == 0:
                    week_volume = base_progression * 0.85  # Recovery week
                else:
                    week_volume = base_progression
                weekly_volumes.append(int(week_volume))
            
            # Taper
            for i in range(taper_weeks):
                taper_factor = 1 - ((i + 1) / taper_weeks) * 0.6
                weekly_volumes.append(int(peak_volume_km * taper_factor))
            
            # Preview
            st.caption("📊 Generated Plan Preview:")
            preview_df = pd.DataFrame({
                'Week': range(1, len(weekly_volumes) + 1),
                'Volume (km)': weekly_volumes
            })
            
            # Show as compact table
            st.dataframe(preview_df.T, use_container_width=True, hide_index=False)
        
        else:
            # Custom plan
            st.caption("Enter weekly volumes manually:")
            weekly_volumes = []
            
            cols_per_row = 4
            for block_start in range(0, weeks_to_race, cols_per_row):
                cols = st.columns(cols_per_row)
                for i in range(cols_per_row):
                    week_idx = block_start + i
                    if week_idx < weeks_to_race:
                        with cols[i]:
                            default_val = int(current_weekly_km + (week_idx * 2) - ((weeks_to_race - week_idx) * 1.5))
                            vol = st.number_input(
                                f"W{week_idx + 1}",
                                value=max(0, default_val),
                                min_value=0,
                                max_value=150,
                                step=5,
                                key=f"custom_week_{week_idx}"
                            )
                            weekly_volumes.append(vol)
        
        # Run Simulation Button
        st.markdown("---")
        
        if st.button("🚀 Run Training Simulation", type="primary", use_container_width=True):
            with st.spinner("Simulating your training plan..."):
                results = st.session_state.simulator.simulate_plan(
                    start_state.copy(),
                    weekly_volumes,
                    race_date
                )
                
                st.session_state.simulation_results = results
                st.session_state.simulation_config = {
                    'race_distance_km': race_distance_km,
                    'elevation_gain_m': elevation_gain_m,
                    'race_date': race_date,
                    'target_time_str': target_time_str,
                    'terrain_type': terrain_type,
                    'weekly_volumes': weekly_volumes,
                    'runs_per_week': runs_per_week,
                    'strength_sessions': strength_sessions_per_week,
                    'start_state': start_state
                }
            
            st.success("✅ Simulation complete! Check the **Simulation Results** tab")
            st.balloons()
    
    # ========================================================================
    # TAB 3: SIMULATION RESULTS
    # ========================================================================
    
    with tab3:
        st.header("📈 Training Plan Simulation Results")
        
        if 'simulation_results' not in st.session_state:
            st.info("👈 Configure and run a simulation in the **Plan Setup** tab first")
        
        else:
            results = st.session_state.simulation_results
            config = st.session_state.simulation_config
            weekly_volumes = config['weekly_volumes']
            
            # Summary Metrics
            st.subheader("📊 Risk Assessment")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                injury_color = "🟢" if results['max_injury_risk'] < 0.2 else "🟡" if results['max_injury_risk'] < 0.4 else "🔴"
                st.metric("Max Injury Risk", f"{injury_color} {results['max_injury_risk']*100:.0f}%")
            
            with col2:
                acwr_color = "🟢" if results['max_acwr'] < 1.3 else "🟡" if results['max_acwr'] < 1.5 else "🔴"
                st.metric("Max ACWR", f"{acwr_color} {results['max_acwr']:.2f}")
            
            with col3:
                st.metric("Total Warnings", len(results['warnings']))
            
            with col4:
                st.metric("Peak Fitness Week", f"Week {results['peak_fitness_week']}")
            
            # Main Visualization
            st.subheader("📈 Training Volume & Load")
            
            weeks = list(range(1, len(weekly_volumes) + 1))
            
            fig = go.Figure()
            
            # Volume bars
            fig.add_trace(go.Bar(
                x=weeks,
                y=weekly_volumes,
                name='Weekly Volume (km)',
                marker_color='lightblue',
                yaxis='y1'
            ))
            
            # ACWR line
            acwr_values = [s.get('acwr', 1.0) for s in results['weekly_states']]
            fig.add_trace(go.Scatter(
                x=weeks,
                y=acwr_values,
                name='ACWR',
                yaxis='y2',
                line=dict(color='orange', width=2)
            ))
            
            # Safe ACWR threshold
            fig.add_trace(go.Scatter(
                x=weeks,
                y=[1.3] * len(weeks),
                name='Safe ACWR Limit (1.3)',
                yaxis='y2',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title='Training Volume & ACWR Progression',
                xaxis=dict(title='Week'),
                yaxis=dict(title='Volume (km)', side='left'),
                yaxis2=dict(title='ACWR', overlaying='y', side='right', range=[0, 2]),
                hovermode='x unified',
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Intensity Breakdown (Summary + Expandable Detail)
            st.subheader("⚡ Intensity Distribution")
            
            # Auto-calculate intensity split
            total_weeks = len(weekly_volumes)
            
            st.markdown("**Weekly Intensity Breakdown:**")
            
            intensity_summary = []
            
            for week_num, week_km in enumerate(weekly_volumes, 1):
                # Determine phase
                if week_num <= total_weeks * 0.6:  # Base building
                    easy_pct = 80
                    tempo_pct = 15
                    intervals_pct = 5
                elif week_num <= total_weeks - 2:  # Build/specific
                    easy_pct = 70
                    tempo_pct = 15
                    intervals_pct = 15
                else:  # Taper
                    easy_pct = 85
                    tempo_pct = 10
                    intervals_pct = 5
                
                easy_km = week_km * (easy_pct / 100)
                tempo_km = week_km * (tempo_pct / 100)
                intervals_km = week_km * (intervals_pct / 100)
                
                intensity_summary.append({
                    'Week': week_num,
                    'Total km': week_km,
                    'Easy (Z2)': f"{easy_km:.0f} km ({easy_pct}%)",
                    'Tempo (Z3)': f"{tempo_km:.0f} km ({tempo_pct}%)",
                    'Intervals (Z4-5)': f"{intervals_km:.0f} km ({intervals_pct}%)"
                })
            
            intensity_df = pd.DataFrame(intensity_summary)
            
            # Show first 4 weeks + last 2 weeks
            st.dataframe(
                intensity_df.head(4),
                use_container_width=True,
                hide_index=True
            )
            
            with st.expander("📋 View Full Intensity Breakdown (All Weeks)"):
                st.dataframe(intensity_df, use_container_width=True, hide_index=True)
            
            # Daily Workout Detail (without specific days)
            with st.expander("🗓️ Daily Workout Structure (Example Week)"):
                example_week = min(4, len(weekly_volumes))
                example_km = weekly_volumes[example_week - 1]
                
                st.markdown(f"**Example: Week {example_week} ({example_km} km total)**")
                
                # Generate sample structure
                runs = config['runs_per_week']
                strength = config['strength_sessions']
                
                long_run_km = example_km * 0.30
                quality_session_km = example_km * 0.20
                remaining_km = example_km - long_run_km - quality_session_km
                easy_km_per_run = remaining_km / max(1, runs - 2)
                
                st.markdown(f"""
                **Workout Structure ({runs} runs + {strength} strength):**
                
                - **Long Run:** {long_run_km:.0f} km @ Z2 (easy pace)
                - **Quality Session:** {quality_session_km:.0f} km total
                  - Example: 2 km warm-up + 5×1000m @ Z4 (2min rest) + 2 km cool-down
                - **Easy Runs (×{max(0, runs-2)}):** {easy_km_per_run:.0f} km each @ Z2
                - **Strength Sessions (×{strength}):** 30-45 min (core, stability, resistance)
                - **Rest Days:** {7 - runs - strength} day(s)
                
                *Note: Actual workout timing and structure will adapt to your recovery.*
                """)
            
            # Warnings
            if results['warnings']:
                st.subheader("⚠️ Risk Warnings")
                
                warnings_df = pd.DataFrame(results['warnings'])
                warnings_df['severity_sort'] = warnings_df['severity'].map({'critical': 0, 'high': 1, 'medium': 2})
                warnings_df = warnings_df.sort_values(['severity_sort', 'week'])
                
                severity_counts = warnings_df['severity'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🚨 Critical", severity_counts.get('critical', 0))
                with col2:
                    st.metric("⚠️ High", severity_counts.get('high', 0))
                with col3:
                    st.metric("⚡ Medium", severity_counts.get('medium', 0))
                
                # Display warnings
                for _, w in warnings_df.head(10).iterrows():
                    severity_emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '⚡'}.get(w['severity'], '•')
                    st.caption(f"{severity_emoji} **Week {w['week']}:** {w['message']}")
                
                if len(warnings_df) > 10:
                    with st.expander(f"Show all {len(warnings_df)} warnings"):
                        for _, w in warnings_df.iterrows():
                            severity_emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '⚡'}.get(w['severity'], '•')
                            st.caption(f"{severity_emoji} Week {w['week']}: {w['message']}")
            
            else:
                st.success("✅ No warnings detected - This plan looks safe!")
            
            # Recovery & HRV Trends
            st.subheader("💪 Recovery Trajectory")
            
            recovery_scores = [s.get('recovery_score', 0.7) * 100 for s in results['weekly_states']]
            hrv_values = [s.get('hrv_rmssd', 80) for s in results['weekly_states']]
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=weeks,
                y=recovery_scores,
                name='Recovery Score (%)',
                line=dict(color='green', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,255,0,0.1)'
            ))
            
            fig2.add_trace(go.Scatter(
                x=weeks,
                y=hrv_values,
                name='Predicted HRV (ms)',
                yaxis='y2',
                line=dict(color='purple', width=2)
            ))
            
            fig2.update_layout(
                xaxis=dict(title='Week'),
                yaxis=dict(title='Recovery Score (%)', range=[0, 100]),
                yaxis2=dict(title='HRV (ms)', overlaying='y', side='right'),
                hovermode='x unified',
                height=350
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Export Options
            st.subheader("📥 Export Plan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                export_df = pd.DataFrame({
                    'Week': weeks,
                    'Volume_km': weekly_volumes,
                    'ACWR': acwr_values,
                    'Recovery_%': recovery_scores,
                    'Predicted_HRV': hrv_values
                })
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    "📊 Download CSV",
                    data=csv,
                    file_name=f"training_plan_{config['race_date']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Text summary
                summary = f"""TRAINING PLAN SUMMARY
{'='*50}

Race: {config['race_distance_km']} km on {config['race_date']}
Target Time: {config['target_time_str']}
Terrain: {config['terrain_type']}

Training Structure:
- {config['runs_per_week']} runs per week
- {config['strength_sessions']} strength sessions per week
- {weeks_to_race} weeks to race

Risk Assessment:
- Max Injury Risk: {results['max_injury_risk']*100:.0f}%
- Max ACWR: {results['max_acwr']:.2f}
- Total Warnings: {len(results['warnings'])}
- Peak Fitness: Week {results['peak_fitness_week']}

Weekly Volumes (km):
{''.join([f'Week {i+1}: {vol} km\n' for i, vol in enumerate(weekly_volumes)])}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
                
                st.download_button(
                    "📄 Download Summary",
                    data=summary,
                    file_name=f"plan_summary_{config['race_date']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # ========================================================================
    # TAB 4: RACE DAY OPTIMIZER
    # ========================================================================
    
    with tab4:
        st.header("🏁 Race Day Optimization")
        
        if 'simulation_config' not in st.session_state:
            st.info("👈 Run a training simulation first to enable race day optimization")
        
        else:
            config = st.session_state.simulation_config
            optimizer = st.session_state.race_optimizer
            
            st.subheader("🏁 Race Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Race Distance", f"{config['race_distance_km']} km")
                st.metric("Elevation Gain", f"{config['elevation_gain_m']} m")
            
            with col2:
                st.metric("Target Time", config['target_time_str'])
                st.metric("Terrain", config['terrain_type'])
            
            # Race Start Time
            st.subheader("⏰ Race Logistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                race_start_time = st.time_input("Race Start Time", value=datetime.strptime("08:00", "%H:%M").time())
                race_start_str = race_start_time.strftime("%H:%M")
            
            with col2:
                pre_race_routine_min = st.slider("Pre-race Routine Duration (min)", min_value=60, max_value=180, value=90, step=15)
            
            # Calculate wake time
            routine = optimizer.calculate_wake_time(race_start_str, pre_race_routine_min)
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("**📅 Race Day Timeline:**")
            st.markdown(f"""
            - **{routine['wake']}** - Wake up
            - **{routine['breakfast']}** - Breakfast ({routine['digestion_buffer']} before race)
            - **{routine['arrive_venue']}** - Arrive at venue
            - **{routine['warmup_start']}** - Begin warm-up
            - **{routine['race_start']}** - 🏁 **RACE START**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Nutrition Planning
            st.subheader("🍌 Nutrition & Fueling Strategy")
            
            target_time_min = parse_time(config['target_time_str'])
            
            energy = optimizer.estimate_energy_needs(
                config['race_distance_km'],
                target_time_min,
                body_weight_kg=67  # Could make this configurable
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Pre-Race Nutrition:**")
                st.write(f"⏰ Breakfast timing: {routine['breakfast']} (2.5-3h before)")
                st.write("🍽️ Recommended meal:")
                st.write("  - 80-100g oats or 2 slices toast")
                st.write("  - 1 banana")
                st.write("  - 400-500 mL water")
                st.write("  - Familiar foods only!")
                
                st.write(f"\n💧 Pre-race hydration: 400-500 mL water/electrolytes")
            
            with col2:
                st.markdown("**During-Race Fueling:**")
                st.write(f"🔥 Estimated calorie burn: {energy['calories_burned']} kcal")
                
                if energy['needs_fueling']:
                    st.write(f"⚡ Carbs needed: {energy['carbs_per_hour_g']} g/hour")
                    st.write(f"🍬 Total gels: {energy['total_gels']}")
                    
                    if energy['total_gels'] > 0:
                        st.write("\n📍 Gel timing:")
                        for i, timing in enumerate(energy['gel_timing_min'], 1):
                            st.write(f"  - Gel #{i}: ~{timing} min (km {timing * config['race_distance_km'] / target_time_min:.1f})")
                    
                    st.write(f"\n💧 Hydration: Sip water at aid stations if > 15°C")
                else:
                    st.success(f"✅ No mid-race fueling needed for {config['race_distance_km']}km!")
                    st.write("💧 Light sips of water if available/thirsty")
            
            # Pacing Strategy
            st.subheader("🏃 Pacing Strategy")
            
            pacing = optimizer.calculate_pacing_strategy(
                config['race_distance_km'],
                target_time_min,
                config['elevation_gain_m']
            )
            
            st.info(f"**Recommended Strategy:** {pacing['strategy'].replace('_', ' ').title()}")
            
            pacing_df = pd.DataFrame(pacing['quarters'])
            
            st.dataframe(pacing_df, use_container_width=True, hide_index=True)
            
            st.caption(f"💡 Average pace: {pacing['avg_pace_min_km']:.2f} min/km")
            
            # Warm-up Protocol
            st.subheader("🔥 Warm-up Routine")
            
            warmup = optimizer.warmup_protocol(config['race_distance_km'])
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Warm-up Duration", f"{warmup['duration_min']} min")
                st.caption(f"Start at: {routine['warmup_start']}")
            
            with col2:
                st.markdown("**Warm-up Structure:**")
                for step in warmup['structure']:
                    st.write(f"• {step}")
            
            # Monte Carlo Simulation
            st.subheader("🎲 Monte Carlo Race Outcome Simulation")
            
            st.markdown("""
            **What this simulates:**
            Run 1000 virtual races with realistic variability in:
            - Weather (temperature 8-20°C)
            - Digestion timing (2.0-3.5 hours individual variation)
            - Pacing execution (±3% from plan)
            - Glycogen availability (90-110%)
            - Hydration status (0-3% performance impact)
            """)
            
            # Interactive controls
            col1, col2 = st.columns(2)
            
            with col1:
                breakfast_hours_before = st.slider(
                    "Breakfast Timing (hours before race)",
                    min_value=2.0,
                    max_value=4.0,
                    value=2.75,
                    step=0.25
                )
            
            with col2:
                gel_strategy = st.selectbox(
                    "Gel Strategy",
                    ["No gels", "Conservative (1 gel)", "Moderate (2 gels)", "Aggressive (3+ gels)"]
                )
                
                gel_count_map = {"No gels": 0, "Conservative (1 gel)": 1, "Moderate (2 gels)": 2, "Aggressive (3+ gels)": 3}
                gel_count = gel_count_map[gel_strategy]
            
            if st.button("🎲 Run Monte Carlo Simulation", type="primary", use_container_width=True):
                with st.spinner("Running 1000 simulations..."):
                    mc_results = optimizer.monte_carlo_simulation(
                        config['race_distance_km'],
                        target_time_min,
                        config['elevation_gain_m'],
                        breakfast_hours_before,
                        gel_count,
                        n_simulations=1000
                    )
                    
                    st.session_state.mc_results = mc_results
                
                st.success("✅ Simulation complete!")
            
            # Display Monte Carlo Results
            if 'mc_results' in st.session_state:
                mc = st.session_state.mc_results
                
                st.markdown("---")
                st.markdown("### 📊 Simulation Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    success_color = "🟢" if mc['success_rate_pct'] > 70 else "🟡" if mc['success_rate_pct'] > 50 else "🔴"
                    st.metric("Success Rate", f"{success_color} {mc['success_rate_pct']:.0f}%")
                    st.caption("Within ±2% of target")
                
                with col2:
                    bonk_color = "🟢" if mc['bonk_risk_pct'] < 10 else "🟡" if mc['bonk_risk_pct'] < 25 else "🔴"
                    st.metric("Bonk Risk", f"{bonk_color} {mc['bonk_risk_pct']:.0f}%")
                
                with col3:
                    gi_color = "🟢" if mc['gi_distress_pct'] < 15 else "🟡" if mc['gi_distress_pct'] < 30 else "🔴"
                    st.metric("GI Distress Risk", f"{gi_color} {mc['gi_distress_pct']:.0f}%")
                
                with col4:
                    dehydration_color = "🟢" if mc['dehydration_pct'] < 10 else "🟡" if mc['dehydration_pct'] < 20 else "🔴"
                    st.metric("Dehydration Risk", f"{dehydration_color} {mc['dehydration_pct']:.0f}%")
                
                # Predicted finish time distribution
                st.markdown("**Predicted Finish Time Range:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("10th Percentile (Fast)", format_time(mc['p10_time']))
                
                with col2:
                    st.metric("50th Percentile (Median)", format_time(mc['p50_time']))
                
                with col3:
                    st.metric("90th Percentile (Slow)", format_time(mc['p90_time']))
                
                # Distribution chart
                fig_mc = optimizer.plot_monte_carlo_results(mc)
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Recommendations
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("**🎯 Recommendations Based on Simulation:**")
                
                if mc['success_rate_pct'] > 70:
                    st.success(f"✅ Your current strategy has a {mc['success_rate_pct']:.0f}% chance of hitting your target!")
                elif mc['success_rate_pct'] > 50:
                    st.warning(f"⚠️ Moderate success probability ({mc['success_rate_pct']:.0f}%). Consider adjusting pacing or nutrition.")
                else:
                    st.error(f"🔴 Low success rate ({mc['success_rate_pct']:.0f}%). Review your target time or strategy.")
                
                if mc['bonk_risk_pct'] > 20:
                    st.warning(f"⚠️ Elevated bonk risk ({mc['bonk_risk_pct']:.0f}%). Consider adding a gel or slowing early pace.")
                
                if mc['gi_distress_pct'] > 25:
                    st.warning(f"⚠️ High GI distress risk ({mc['gi_distress_pct']:.0f}%). Move breakfast earlier or use more familiar foods.")
                
                if mc['dehydration_pct'] > 15:
                    st.info(f"💧 Dehydration risk detected ({mc['dehydration_pct']:.0f}%). Plan to hydrate at aid stations.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Strategy comparison
                st.markdown("---")
                st.markdown("### 🔄 Try Different Strategies")
                st.caption("Adjust breakfast timing and gel count above, then re-run simulation to compare outcomes.")
    
    # ========================================================================
    # TAB 5: SETTINGS
    # ========================================================================
    
    with tab5:
        st.header("⚙️ Settings & Data Management")
        
        # Data Source
        st.subheader("📊 Garmin Data")
        
        if garmin_df is not None:
            st.success(f"✅ {len(garmin_df)} days of Garmin data loaded")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Date Range", f"{garmin_df['date'].min().strftime('%Y-%m-%d')} to {garmin_df['date'].max().strftime('%Y-%m-%d')}")
            
            with col2:
                st.metric("Total Distance", f"{garmin_df['distance_km'].sum():.0f} km")
            
            if st.button("🔄 Refresh Garmin Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        else:
            st.warning("⚠️ No Garmin data found")
            
            st.info("""
            **To sync Garmin data:**
            
            1. Open terminal
            2. Run: `python scripts/data_collection.py`
            3. Then: `python scripts/preprocessing.py`
            4. Refresh this page
            """)
            
            st.markdown("---")
            st.subheader("📤 Upload Data Manually")
            
            uploaded_file = st.file_uploader("Upload training_dataset.csv", type=['csv'])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                df.to_csv('data/processed/training_dataset.csv', index=False)
                st.success("✅ Data uploaded! Refresh the page.")
                
                if st.button("Refresh Now"):
                    st.rerun()
        
        # Wizard Reset
        st.markdown("---")
        st.subheader("🧙‍♂️ Setup Wizard")
        
        if st.button("🔄 Restart Setup Wizard", use_container_width=True):
            st.session_state.wizard_completed = False
            st.session_state.current_wizard_step = 1
            st.rerun()
        
        # App Info
        st.markdown("---")
        st.subheader("ℹ️ About")
        
        st.markdown("""
        **Training Simulator Pro v2.0**
        
        AI-powered training optimization with:
        - Comprehensive Garmin data analysis
        - Monte Carlo race day simulation
        - Personalized training plans
        - Injury risk assessment
        - Race nutrition & pacing optimization
        
        Built with Python, Streamlit, scikit-learn
        
        © 2025
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Keep ACWR < 1.3 | Increase volume gradually | Trust your data")
