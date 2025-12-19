"""
Training Plan Simulator - Complete Mobile-Optimized Application
Version 2.1 - Streamlined with Import-First Workflow
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import time
from scripts.simulator import TrainingSimulator
from scripts.garmin_analyzer import GarminAnalyzer
from scripts.race_optimizer import RaceOptimizer



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
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
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

# Quick status indicator
if garmin_df is not None:
    st.success(f"✅ Data loaded: {len(garmin_df)} days ({garmin_df['date'].min().strftime('%Y-%m-%d')} to {garmin_df['date'].max().strftime('%Y-%m-%d')})")
else:
    st.info("👉 Start by importing your data in the **Import Data** tab")

# ============================================================================
# TAB NAVIGATION (Import Data is now FIRST)
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 Import Data",
    "📊 Garmin Dashboard",
    "🎯 Plan Setup",
    "📈 Simulation Results",
    "🏁 Race Day Optimizer",
    "⚙️ Settings"
])

# ============================================================================
# TAB 1: IMPORT DATA (NEW - MOVED TO FIRST POSITION)
# ============================================================================

with tab1:
    st.header("📥 Import Training Data")
    
    # Welcome message for new users
    if garmin_df is None:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 👋 Welcome to Training Simulator Pro!
        
        To get started, import your training data using one of the methods below:
        - **Upload CSV files** - Best if you already exported from Garmin
        - **Connect Garmin directly** - Automatic sync (requires login)
        - **Manual entry** - For quick testing with a few activities
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Import method selection
    import_method = st.radio(
        "Choose Import Method",
        ["📂 Upload CSV Files", "🔗 Connect Garmin Account", "✍️ Manual Entry"],
        horizontal=True
    )
    
    # ====================================================================
    # METHOD 1: CSV UPLOAD
    # ====================================================================
    
    if import_method == "📂 Upload CSV Files":
        st.subheader("📂 Upload Your Training Data")
        
        with st.expander("ℹ️ How to export data from Garmin Connect", expanded=False):
            st.markdown("""
            **Method 1: Garmin Connect Website**
            1. Go to [connect.garmin.com](https://connect.garmin.com)
            2. Click Activities → All Activities
            3. Use filters to select date range
            4. Click Export → CSV
            
            **Method 2: Using this app's data collection script**
            1. Open terminal
            2. Run: `python scripts/data_collection.py`
            3. Upload the generated CSV files below
            
            **Expected columns:**
            - Activities: `date, distance, duration, averageHR, elevationGain`
            - Health: `date, resting_hr, hrv_rmssd, sleep_hours, stress_avg`
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            activities_file = st.file_uploader(
                "Upload Activities CSV",
                type=['csv'],
                key="activities_upload",
                help="Your running/training activities"
            )
        
        with col2:
            health_file = st.file_uploader(
                "Upload Health Metrics CSV (Optional)",
                type=['csv'],
                key="health_upload",
                help="HRV, sleep, resting HR data"
            )
        
        if activities_file is not None:
            # Read and preview data
            activities_df = pd.read_csv(activities_file)
            
            st.success(f"✅ Loaded {len(activities_df)} activities!")
            
            # Data preview
            with st.expander("👀 Preview Activities Data", expanded=True):
                st.dataframe(activities_df.head(10), width="stretch")
                
                # Column mapping helper
                st.markdown("**🔧 Column Mapping:**")
                st.caption("Map your CSV columns to required fields (auto-detected, adjust if needed):")
                
                available_cols = list(activities_df.columns)
                
                # Auto-detect columns
                def find_column(keywords):
                    for keyword in keywords:
                        for col in available_cols:
                            if keyword.lower() in col.lower():
                                return col
                    return available_cols[0] if available_cols else None
                
                date_col = find_column(['date', 'time', 'start', 'timestamp'])
                distance_col = find_column(['distance', 'km', 'miles'])
                duration_col = find_column(['duration', 'time', 'minutes', 'hours'])
                hr_col = find_column(['hr', 'heart', 'averageHR', 'avg_hr'])
                elevation_col = find_column(['elevation', 'gain', 'ascent'])
                
                col_map = {}
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    col_map['date'] = st.selectbox(
                        "Date/Time column",
                        available_cols,
                        index=available_cols.index(date_col) if date_col in available_cols else 0
                    )
                    col_map['distance'] = st.selectbox(
                        "Distance column",
                        available_cols,
                        index=available_cols.index(distance_col) if distance_col in available_cols else 0
                    )
                
                with col2:
                    col_map['duration'] = st.selectbox(
                        "Duration column",
                        available_cols,
                        index=available_cols.index(duration_col) if duration_col in available_cols else 0
                    )
                    col_map['hr'] = st.selectbox(
                        "Heart Rate column",
                        available_cols,
                        index=available_cols.index(hr_col) if hr_col in available_cols else 0
                    )
                
                with col3:
                    elevation_options = ['None'] + available_cols
                    col_map['elevation'] = st.selectbox(
                        "Elevation column (optional)",
                        elevation_options,
                        index=elevation_options.index(elevation_col) if elevation_col in elevation_options else 0
                    )
            
            # Process uploaded data
            if st.button("✅ Process & Import Activities", type="primary", width="stretch"):
                with st.spinner("Processing uploaded data..."):
                    try:
                        # Rename columns
                        processed_df = activities_df.copy()
                        
                        rename_map = {
                            col_map['date']: 'startTimeLocal',
                            col_map['distance']: 'distance',
                            col_map['duration']: 'duration',
                            col_map['hr']: 'averageHR'
                        }
                        
                        if col_map['elevation'] != 'None':
                            rename_map[col_map['elevation']] = 'elevationGain'
                        
                        processed_df = processed_df.rename(columns=rename_map)
                        
                        # Convert date
                        processed_df['date'] = pd.to_datetime(processed_df['startTimeLocal'])
                        
                        # Ensure data directory exists
                        import os
                        os.makedirs('data/raw', exist_ok=True)
                        
                        # Save to file
                        processed_df.to_csv('data/raw/activities.csv', index=False)
                        st.session_state.imported_activities = processed_df
                        
                        st.success(f"✅ Successfully imported {len(processed_df)} activities!")
                        st.balloons()
                        
                        # Show next steps
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown("""
                        **✅ Data imported successfully!**
                        
                        **Next steps:**
                        1. Go to **Settings** tab
                        2. Click "Run Preprocessing" to generate training features
                        3. Then use **Plan Setup** to create your training plan
                        
                        *Or continue to import health metrics below (optional)*
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error processing data: {e}")
                        st.info("💡 Check your column mapping and data format. Common issues: date format, missing columns")
        
        # Health metrics upload (optional)
        if health_file is not None:
            health_df = pd.read_csv(health_file)
            
            st.success(f"✅ Loaded {len(health_df)} days of health data!")
            
            with st.expander("👀 Preview Health Data"):
                st.dataframe(health_df.head(10), width="stretch")
            
            if st.button("✅ Import Health Metrics", type="secondary", width="stretch"):
                import os
                os.makedirs('data/raw', exist_ok=True)
                health_df.to_csv('data/raw/health_metrics.csv', index=False)
                st.success("✅ Health metrics imported! Don't forget to run preprocessing.")
    
    # ====================================================================
    # METHOD 2: GARMIN DIRECT CONNECTION
    # ====================================================================
    
    elif import_method == "🔗 Connect Garmin Account":
        st.subheader("🔗 Connect Your Garmin Account")
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ⚠️ **Privacy Notice:**
        - Your credentials are used **only** to fetch data from Garmin servers
        - Credentials are **NOT stored** anywhere
        - All data processing happens locally on your device
        - You can disconnect anytime
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            garmin_email = st.text_input("Garmin Email", type="default", placeholder="your.email@example.com")
            garmin_password = st.text_input("Garmin Password", type="password")
        
        with col2:
            st.markdown("**📅 Data Collection Settings:**")
            
            collect_activities = st.number_input(
                "Number of Activities to Fetch",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Recent activities (runs, bikes, etc.)"
            )
            
            collect_days = st.number_input(
                "Days of Health Data to Fetch",
                min_value=7,
                max_value=365,
                value=90,
                step=7,
                help="HRV, sleep, resting HR, stress"
            )
            
            st.caption(f"⏱️ Estimated time: ~{collect_days * 0.5 / 60:.0f} minutes")
        
        if st.button("🔄 Connect & Download Data", type="primary", width="stretch"):
            if not garmin_email or not garmin_password:
                st.error("❌ Please enter your Garmin credentials")
            else:
                with st.spinner(f"Connecting to Garmin and downloading data..."):
                    try:
                        from garminconnect import Garmin
                        
                        # Connect
                        client = Garmin(garmin_email, garmin_password)
                        client.login()
                        st.success("✅ Connected to Garmin!")
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Fetch activities
                        status_text.text("📥 Downloading activities...")
                        progress_bar.progress(10)
                        
                        activities = client.get_activities(0, collect_activities)
                        activities_df = pd.DataFrame(activities)
                        
                        progress_bar.progress(40)
                        status_text.text(f"✅ Downloaded {len(activities_df)} activities!")
                        
                        # Save activities
                        import os
                        os.makedirs('data/raw', exist_ok=True)
                        activities_df.to_csv('data/raw/activities.csv', index=False)
                        st.session_state.imported_activities = activities_df
                        
                        # Fetch health data
                        status_text.text("📥 Downloading health metrics (HRV, sleep, etc.)...")
                        progress_bar.progress(50)
                        
                        health_data = []
                        end_date = datetime.now().date()
                        start_date = end_date - timedelta(days=collect_days)
                        current_date = start_date
                        
                        days_processed = 0
                        total_days = collect_days
                        
                        while current_date <= end_date:
                            date_str = current_date.strftime('%Y-%m-%d')
                            
                            daily_stats = {
                                'date': date_str,
                                'resting_hr': None,
                                'hrv_rmssd': None,
                                'sleep_hours': None,
                                'stress_avg': None
                            }
                            
                            try:
                                hrv = client.get_hrv_data(date_str)
                                if hrv and 'hrvSummary' in hrv:
                                    daily_stats['hrv_rmssd'] = hrv['hrvSummary'].get('lastNightAvg')
                            except:
                                pass
                            
                            try:
                                sleep = client.get_sleep_data(date_str)
                                if sleep and 'dailySleepDTO' in sleep:
                                    daily_stats['sleep_hours'] = sleep['dailySleepDTO'].get('sleepTimeSeconds', 0) / 3600
                            except:
                                pass
                            
                            try:
                                stats = client.get_stats(date_str)
                                if stats:
                                    daily_stats['resting_hr'] = stats.get('restingHeartRate')
                            except:
                                pass
                            
                            try:
                                stress = client.get_stress_data(date_str)
                                if stress and 'avgStressLevel' in stress:
                                    daily_stats['stress_avg'] = stress['avgStressLevel']
                            except:
                                pass
                            
                            health_data.append(daily_stats)
                            
                            days_processed += 1
                            progress = 50 + int((days_processed / total_days) * 45)
                            progress_bar.progress(min(progress, 95))
                            
                            if days_processed % 10 == 0:
                                status_text.text(f"📥 Processing health data... {days_processed}/{total_days} days")
                            
                            current_date += timedelta(days=1)
                            time.sleep(0.5)  # Rate limiting to be nice to Garmin
                        
                        health_df = pd.DataFrame(health_data)
                        health_df.to_csv('data/raw/health_metrics.csv', index=False)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Download complete!")
                        
                        st.balloons()
                        
                        # Success summary
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"""
                        **✅ Successfully imported from Garmin:**
                        - 🏃 **{len(activities_df)} activities**
                        - 📊 **{len(health_df)} days of health metrics**
                        - 📅 Date range: **{start_date}** to **{end_date}**
                        
                        **Next steps:**
                        1. Go to **Settings** tab
                        2. Click "Run Preprocessing" to generate features
                        3. Explore your data in **Garmin Dashboard**
                        4. Create your plan in **Plan Setup**
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")
                        st.info("""
                        **Troubleshooting:**
                        - Check your email and password
                        - Make sure you can log in at connect.garmin.com
                        - Try reducing the number of days to fetch
                        - Some Garmin accounts require 2FA - in that case, use CSV upload instead
                        """)
    
    # ====================================================================
    # METHOD 3: MANUAL ENTRY
    # ====================================================================
    
    elif import_method == "✍️ Manual Entry":
        st.subheader("✍️ Manual Data Entry")
        
        st.info("💡 Perfect for quick testing or if you only have a few recent activities to add")
        
        # Initialize session state for manual entries
        if 'manual_entries' not in st.session_state:
            st.session_state.manual_entries = []
        
        st.markdown("**➕ Add New Activity:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            entry_date = st.date_input("Date", value=datetime.now())
        
        with col2:
            entry_distance = st.number_input("Distance (km)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
        
        with col3:
            entry_duration = st.number_input("Duration (min)", min_value=1, max_value=600, value=50, step=1)
        
        with col4:
            entry_hr = st.number_input("Avg HR (bpm)", min_value=60, max_value=220, value=150, step=1)
        
        if st.button("➕ Add Activity", width="stretch"):
            st.session_state.manual_entries.append({
                'date': entry_date.strftime('%Y-%m-%d'),
                'distance': entry_distance,
                'duration': entry_duration * 60,  # Convert to seconds
                'averageHR': entry_hr,
                'startTimeLocal': entry_date.strftime('%Y-%m-%d')
            })
            st.success("✅ Activity added!")
            st.rerun()
        
        # Show current entries
        if len(st.session_state.manual_entries) > 0:
            st.markdown(f"**📋 Current Entries ({len(st.session_state.manual_entries)}):**")
            
            entries_df = pd.DataFrame(st.session_state.manual_entries)
            
            # Display with better formatting
            display_df = entries_df.copy()
            display_df['duration_min'] = (display_df['duration'] / 60).round(0).astype(int)
            display_df = display_df[['date', 'distance', 'duration_min', 'averageHR']]
            display_df.columns = ['Date', 'Distance (km)', 'Duration (min)', 'Avg HR']
            
            st.dataframe(display_df, width="stretch", hide_index=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save All Entries", type="primary", width="stretch"):
                    import os
                    os.makedirs('data/raw', exist_ok=True)
                    entries_df.to_csv('data/raw/activities.csv', index=False)
                    st.session_state.imported_activities = entries_df
                    st.success("✅ Saved to activities.csv! Go to Settings tab to run preprocessing.")
            
            with col2:
                if st.button("🗑️ Clear All", width="stretch"):
                    st.session_state.manual_entries = []
                    st.rerun()
        else:
            st.info("👆 Add activities using the form above")
    
    # ====================================================================
    # DATA VISUALIZATION (shown after any import)
    # ====================================================================
    
    st.markdown("---")
    st.subheader("📊 Imported Data Visualization")
    
    # Check if data exists
    viz_df = None
    if 'imported_activities' in st.session_state:
        viz_df = st.session_state.imported_activities
        data_source = "Current session"
    elif garmin_df is not None:
        viz_df = garmin_df
        data_source = "Processed dataset"
    
    if viz_df is not None:
        # Data summary
        st.markdown(f"**📈 Data Summary** *(Source: {data_source})*")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Activities", len(viz_df))
        
        with col2:
            if 'distance_km' in viz_df.columns:
                st.metric("Total Distance", f"{viz_df['distance_km'].sum():.0f} km")
            elif 'distance' in viz_df.columns:
                # Convert meters to km if needed
                distance_sum = viz_df['distance'].sum()
                if distance_sum > 10000:  # Probably in meters
                    distance_sum = distance_sum / 1000
                st.metric("Total Distance", f"{distance_sum:.0f} km")
        
        with col3:
            date_col = 'date' if 'date' in viz_df.columns else 'startTimeLocal'
            if date_col in viz_df.columns:
                viz_df[date_col] = pd.to_datetime(viz_df[date_col])
                date_span = (viz_df[date_col].max() - viz_df[date_col].min()).days
                st.metric("Date Span", f"{date_span} days")
        
        with col4:
            if 'averageHR' in viz_df.columns:
                avg_hr = viz_df['averageHR'].mean()
                if not pd.isna(avg_hr):
                    st.metric("Avg HR", f"{avg_hr:.0f} bpm")
        
        # Activity distribution chart
        try:
            if date_col in viz_df.columns:
                st.markdown("**📅 Activity Timeline:**")
                
                viz_df_sorted = viz_df.sort_values(date_col)
                
                # Determine distance column
                dist_col = 'distance_km' if 'distance_km' in viz_df.columns else 'distance'
                distances = viz_df_sorted[dist_col].copy()
                
                # Convert to km if in meters
                if distances.mean() > 100:
                    distances = distances / 1000
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=viz_df_sorted[date_col],
                    y=distances,
                    mode='markers+lines',
                    name='Distance',
                    marker=dict(size=8, color='blue'),
                    line=dict(width=1, color='lightblue')
                ))
                
                fig.update_layout(
                    title='Training Activity Timeline',
                    xaxis_title='Date',
                    yaxis_title='Distance (km)',
                    hovermode='x unified',
                    height=350
                )
                
                st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Could not create timeline chart: {e}")
        
        # Data quality indicators
        st.markdown("**✅ Data Quality Check:**")
        
        quality_checks = []
        
        if date_col in viz_df.columns:
            complete = viz_df[date_col].notna().sum()
            quality_checks.append(("✅" if complete == len(viz_df) else "⚠️", "Date/time data", f"{complete}/{len(viz_df)} complete"))
        
        dist_col = 'distance_km' if 'distance_km' in viz_df.columns else 'distance'
        if dist_col in viz_df.columns:
            complete = viz_df[dist_col].notna().sum()
            quality_checks.append(("✅" if complete == len(viz_df) else "⚠️", "Distance data", f"{complete}/{len(viz_df)} complete"))
        
        if 'averageHR' in viz_df.columns:
            hr_complete = viz_df['averageHR'].notna().sum()
            pct = hr_complete/len(viz_df)
            quality_checks.append((("✅" if pct > 0.8 else "⚠️" if pct > 0.5 else "❌"), "Heart rate data", f"{hr_complete}/{len(viz_df)} complete ({pct*100:.0f}%)"))
        
        dur_col = 'duration_hours' if 'duration_hours' in viz_df.columns else 'duration'
        if dur_col in viz_df.columns:
            complete = viz_df[dur_col].notna().sum()
            quality_checks.append(("✅" if complete == len(viz_df) else "⚠️", "Duration data", f"{complete}/{len(viz_df)} complete"))
        
        for emoji, check, status in quality_checks:
            st.caption(f"{emoji} **{check}:** {status}")
        
        # Next steps
        if 'imported_activities' in st.session_state and garmin_df is None:
            st.markdown("---")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            **🎯 Next Steps:**
            1. Go to **Settings** tab
            2. Click **"Run Preprocessing"** to generate training features
            3. Wait for processing to complete
            4. Return to explore **Garmin Dashboard** with your data
            5. Create your training plan in **Plan Setup**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("👆 Import data using one of the methods above to see visualization and quality checks")
    

# ========================================================================
# TAB 2: GARMIN DASHBOARD
# ========================================================================

with tab2:
    st.header("📊 Garmin Data Analysis")
    
    if garmin_df is None:
        st.warning("⚠️ No processed data available")
        st.info("""
        **To view your dashboard:**
        1. Import data in the **Import Data** tab
        2. Go to **Settings** and run preprocessing
        3. Return here to see comprehensive analytics
        """)
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
        
        st.caption(f"📊 Showing data from {start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date} to {end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date} ({len(filtered_df)} days)")
        
        # Volume Statistics
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
            st.plotly_chart(fig_volume, width="stretch")
        
        # Intensity Distribution
        st.subheader("⚡ Intensity Distribution")
        
        zones = analyzer.get_intensity_distribution(filtered_df)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Time in Zones:**")
            for zone, pct in zones.items():
                st.write(f"{zone}: {pct:.1f}%")
        
        with col2:
            fig_zones = analyzer.plot_intensity_distribution(zones)
            st.plotly_chart(fig_zones, width="stretch")
        
        # Physiological Trends
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
                st.plotly_chart(fig_hrv, width="stretch")
        
        # Combined health metrics
        fig_health = analyzer.plot_combined_health_metrics(filtered_df)
        if fig_health:
            st.plotly_chart(fig_health, width="stretch")
        
        # Raw Data Table
        with st.expander("📋 View Raw Data"):
            st.dataframe(
                filtered_df.sort_values('date', ascending=False).head(50),
                width="stretch",
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
# TAB 3: PLAN SETUP (COMPLETELY REBUILT)
# ========================================================================

with tab3:
    st.header("🎯 Intelligent Training Plan Generator")
    
    st.markdown("""
    Create a personalized training plan that adapts to **your data, your goals, and your life**.
    No rigid schedules - just intelligent, adaptive coaching.
    """)
    
    # ====================================================================
    # STEP 1: TRAINING OBJECTIVE
    # ====================================================================
    
    st.subheader("1️⃣ What's Your Goal?")
    
    objective_type = st.radio(
        "Choose your training objective:",
        ["🏁 Specific Race Goal", "💪 General Fitness Improvement", "🔄 Return from Injury/Break"],
        horizontal=False,
        key="plan_setup_objective_type"
    )
    
    objective_details = {}  # Initialize empty dict
    
    # --------------------------------------------------------------------
    # OBJECTIVE A: RACE GOAL
    # --------------------------------------------------------------------
    
    if objective_type == "🏁 Specific Race Goal":
        st.markdown("**Race Configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            race_distance_km = st.number_input(
                "Race Distance (km)", 
                value=10.0, 
                min_value=1.0, 
                max_value=100.0, 
                step=0.5,
                key="plan_setup_race_distance"
            )
            
            elevation_gain_m = st.number_input(
                "Elevation Gain (m)", 
                value=0, 
                min_value=0, 
                max_value=5000, 
                step=50,
                key="plan_setup_elevation"
            )
            
            race_date = st.date_input(
                "Race Date",
                value=(datetime.now() + timedelta(weeks=12)).date(),
                key="plan_setup_race_date"
            )
        
        with col2:
            terrain_type = st.selectbox(
                "Terrain",
                ["Road", "Trail - Easy", "Trail - Technical", "Mountain/Ultra", "Track"],
                index=0,
                key="plan_setup_terrain"
            )
            
            target_time_str = st.text_input(
                "Target Time (MM:SS)", 
                value="39:30", 
                help="Leave empty for 'finish comfortably' goal",
                key="plan_setup_target_time"
            )
            
            weeks_to_race = max(4, min(24, (race_date - datetime.now().date()).days // 7))
            st.metric("Weeks to Race", weeks_to_race)
        
        objective_details = {
            'type': 'race',
            'distance_km': race_distance_km,
            'elevation_m': elevation_gain_m,
            'terrain': terrain_type,
            'target_time': target_time_str if target_time_str else None,
            'race_date': race_date,
            'duration_weeks': weeks_to_race
        }
    
    # --------------------------------------------------------------------
    # OBJECTIVE B: GENERAL FITNESS
    # --------------------------------------------------------------------
    
    elif objective_type == "💪 General Fitness Improvement":
        st.markdown("**Select Your Primary Focus:**")
        
        fitness_goal = st.selectbox(
            "What do you want to improve?",
            [
                "🏃 Build Endurance (run longer distances)",
                "⚡ Increase Speed (run faster)",
                "📈 Increase Weekly Volume",
                "❤️ Improve VO2max / Aerobic Capacity",
                "💓 Lower Resting Heart Rate",
                "📊 Develop Training Consistency",
                "🛡️ Injury Prevention & Resilience",
                "⚖️ Weight Loss / Body Composition",
                "🏋️ Combine Running + Strength Training"
            ],
            key="plan_setup_fitness_goal"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            duration_weeks = st.slider(
                "Training Block Duration (weeks)", 
                min_value=4, 
                max_value=24, 
                value=12,
                key="plan_setup_duration_fitness"
            )
        
        with col2:
            intensity_preference = st.selectbox(
                "Training Intensity Preference",
                ["Conservative (focus on consistency)", "Moderate (balanced)", "Aggressive (maximize gains)"],
                key="plan_setup_intensity_pref"
            )
        
        objective_details = {
            'type': 'fitness',
            'goal': fitness_goal,
            'duration_weeks': duration_weeks,
            'intensity': intensity_preference
        }
    
    # --------------------------------------------------------------------
    # OBJECTIVE C: RETURN FROM INJURY/BREAK
    # --------------------------------------------------------------------
    
    elif objective_type == "🔄 Return from Injury/Break":
        st.markdown("**Gradual Return to Running:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            break_duration = st.selectbox(
                "How long have you been away?",
                ["1-2 weeks", "3-4 weeks", "1-2 months", "3-6 months", "6+ months"],
                key="plan_setup_break_duration"
            )
            
            previous_volume = st.number_input(
                "Previous Weekly Volume (km)",
                value=40.0,
                min_value=10.0,
                max_value=150.0,
                step=5.0,
                help="Your typical weekly km before the break",
                key="plan_setup_previous_volume"
            )
        
        with col2:
            return_reason = st.selectbox(
                "Reason for break",
                ["Injury recovery", "Illness", "Life circumstances", "Burnout/overtraining"],
                key="plan_setup_return_reason"
            )
            
            duration_weeks = st.slider(
                "Rebuild Duration (weeks)", 
                min_value=4, 
                max_value=16, 
                value=8,
                key="plan_setup_duration_return"
            )
        
        objective_details = {
            'type': 'return',
            'break_duration': break_duration,
            'previous_volume': previous_volume,
            'reason': return_reason,
            'duration_weeks': duration_weeks
        }
    
    # ====================================================================
    # STEP 2: PLAN STYLE
    # ====================================================================
    
    st.markdown("---")
    st.subheader("2️⃣ Choose Your Plan Style")
    
    plan_style = st.radio(
        "How do you want your training structured?",
        [
            "🟢 Structured Classic - Fixed weekly progression with recovery cycles",
            "🟣 Adaptive AI Coach - Daily workouts adjust to your recovery & data",
            "🔵 Minimalist - 3 sessions/week, maximum efficiency",
            "🔴 High Performance - 5-6 sessions/week, serious training",
            "🟡 Flexible Guidance - Weekly targets + daily suggestions (not mandatory)"
        ],
        key="plan_setup_plan_style"
    )
    
    # Additional options per style
    use_recovery_weeks = False
    progression_rate = 10
    adaptation_sensitivity = 3
    minimalist_focus = None
    include_doubles = False
    
    if "Structured Classic" in plan_style:
        use_recovery_weeks = st.checkbox("Include recovery weeks (3 weeks build + 1 easy)", value=True, key="plan_setup_recovery_weeks")
        progression_rate = st.slider("Volume increase per week (%)", min_value=5, max_value=15, value=10, key="plan_setup_progression_rate")
    
    elif "Adaptive AI" in plan_style:
        st.info("""
        **How Adaptive AI works:**
        - Each morning, the app analyzes your HRV, resting HR, sleep, and fatigue
        - Today's workout is calculated in real-time based on your recovery state
        - If you're fatigued → easier session or rest day
        - If you're fresh → opportunity for quality work
        """)
        
        adaptation_sensitivity = st.slider(
            "Adaptation Sensitivity",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = strict (follows data closely), 5 = flexible (prioritizes consistency)",
            key="plan_setup_adaptation_sensitivity"
        )
    
    elif "Minimalist" in plan_style:
        minimalist_focus = st.selectbox(
            "Primary session type",
            ["2 easy + 1 long run", "1 easy + 1 tempo + 1 long", "2 easy + 1 intervals"],
            key="plan_setup_minimalist_focus"
        )
    
    elif "High Performance" in plan_style:
        include_doubles = st.checkbox("Include double sessions (AM/PM runs)", value=False, key="plan_setup_include_doubles")
        st.warning("⚠️ High volume training requires excellent recovery habits")
    
    # ====================================================================
    # STEP 3: CURRENT FITNESS & DATA
    # ====================================================================
    
    st.markdown("---")
    st.subheader("3️⃣ Current Fitness State")
    
    use_garmin_auto = st.checkbox(
        "Auto-fill from Garmin data",
        value=current_state_auto is not None,
        help="Use your recent training data to set baseline",
        key="plan_setup_use_garmin_auto"
    )
    
    col1, col2, col3 = st.columns(3)
    
    if use_garmin_auto and current_state_auto:
        with col1:
            current_weekly_km = st.number_input(
                "Current Weekly Volume (km)",
                value=float(current_state_auto['weekly_km_avg']),
                min_value=10.0,
                max_value=150.0,
                step=5.0,
                key="plan_setup_weekly_km_auto"
            )
        
        with col2:
            current_hrv = st.number_input(
                "Recent HRV (ms)",
                value=float(current_state_auto['hrv_rmssd']),
                min_value=30.0,
                max_value=120.0,
                step=1.0,
                key="plan_setup_hrv_auto"
            )
        
        with col3:
            current_rhr = st.number_input(
                "Resting HR (bpm)",
                value=float(current_state_auto['resting_hr']),
                min_value=35.0,
                max_value=80.0,
                step=1.0,
                key="plan_setup_rhr_auto"
            )
        
        start_state = current_state_auto.copy()
        start_state['hrv_rmssd'] = current_hrv
        start_state['resting_hr'] = current_rhr
        start_state['acute_load_km'] = current_weekly_km
    
    else:
        with col1:
            current_weekly_km = st.number_input(
                "Current Weekly Volume (km)", 
                value=40.0, 
                min_value=0.0, 
                max_value=150.0, 
                step=5.0,
                key="plan_setup_weekly_km_manual"
            )
        
        with col2:
            current_hrv = st.number_input(
                "Recent HRV (ms)", 
                value=79.0, 
                min_value=30.0, 
                max_value=120.0, 
                step=1.0,
                key="plan_setup_hrv_manual"
            )
        
        with col3:
            current_rhr = st.number_input(
                "Resting HR (bpm)", 
                value=45.0, 
                min_value=35.0, 
                max_value=80.0, 
                step=1.0,
                key="plan_setup_rhr_manual"
            )
        
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
    
    # ====================================================================
    # STEP 4: TRAINING PREFERENCES & CONSTRAINTS
    # ====================================================================
    
    st.markdown("---")
    st.subheader("4️⃣ Training Preferences & Constraints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Weekly Structure:**")
        
        runs_per_week = st.slider("Runs per Week", min_value=2, max_value=7, value=4, key="plan_setup_runs_per_week")
        strength_per_week = st.slider("Strength/Cross-training Sessions", min_value=0, max_value=4, value=1, key="plan_setup_strength_per_week")
        
        max_session_duration = st.number_input(
            "Max Single Session Duration (minutes)",
            value=120,
            min_value=30,
            max_value=240,
            step=15,
            help="No individual run longer than this",
            key="plan_setup_max_session_duration"
        )
    
    with col2:
        st.markdown("**Training Focus:**")
        
        priority = st.selectbox(
            "Primary Focus",
            ["Balanced", "Speed Priority", "Endurance Priority", "Volume Priority"],
            key="plan_setup_priority"
        )
        
        include_technical_work = st.checkbox("Include Technical Drills (form, cadence)", value=False, key="plan_setup_technical_work")
        
        environment = st.selectbox("Primary Training Environment", ["Outdoor", "Indoor/Treadmill", "Mixed"], key="plan_setup_environment")
    
    # Advanced preferences
    with st.expander("🔧 Advanced Customization"):
        col1, col2 = st.columns(2)
        
        with col1:
            long_run_day = st.selectbox("Preferred Long Run Day", ["Weekend", "Friday", "Any day", "No preference"], key="plan_setup_long_run_day")
            avoid_consecutive_hard = st.checkbox("Avoid back-to-back hard sessions", value=True, key="plan_setup_avoid_consecutive")
        
        with col2:
            prefer_morning = st.checkbox("Prefer morning runs", value=False, key="plan_setup_prefer_morning")
            include_hills = st.checkbox("Include hill work", value=False, key="plan_setup_include_hills")
    
    # ====================================================================
    # STEP 5: AI COACHING INTELLIGENCE
    # ====================================================================
    
    st.markdown("---")
    st.subheader("5️⃣ Adaptive Coaching Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_adjust = st.checkbox(
            "Enable Auto-Adjustment",
            value=True,
            help="Plan adapts weekly based on your performance and recovery",
            key="plan_setup_auto_adjust"
        )
        
        adjust_hrv = False
        adjust_rhr = False
        adjust_fatigue = False
        adjust_volume = False
        
        if auto_adjust:
            st.markdown("**Auto-adjust when:**")
            adjust_hrv = st.checkbox("HRV drops >10% below baseline", value=True, key="plan_setup_adjust_hrv")
            adjust_rhr = st.checkbox("Resting HR elevated >5 bpm", value=True, key="plan_setup_adjust_rhr")
            adjust_fatigue = st.checkbox("Consecutive hard days >3", value=True, key="plan_setup_adjust_fatigue")
            adjust_volume = st.checkbox("ACWR >1.3 (injury risk)", value=True, key="plan_setup_adjust_volume")
    
    with col2:
        progression_tracking = st.selectbox(
            "Track Progress By",
            [
                "Heart Rate Efficiency (faster at same HR)",
                "Resting HR Trend",
                "HRV Improvement",
                "Weekly Volume Increase",
                "Pace at Aerobic Threshold",
                "Running Economy (cadence, form)"
            ],
            key="plan_setup_progression_tracking"
        )
        
        weekly_feedback = st.checkbox("Receive weekly progress summaries", value=True, key="plan_setup_weekly_feedback")
    
    # ====================================================================
    # STEP 6: AI GUIDANCE & VALIDATION (MOVED AFTER ALL INPUTS)
    # ====================================================================
    
    st.markdown("---")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Plan Validation")
    
    # Validate based on objective and current state
    warnings = []
    recommendations = []
    
    # Validation logic
    duration = objective_details.get('duration_weeks', 12)
    
    if objective_details.get('type') == 'race':
        target_km = objective_details.get('distance_km', 10)
        
        # Check if enough time
        weeks_needed = max(8, int(target_km / 2))  # Rough heuristic
        if duration < weeks_needed:
            warnings.append(f"⚠️ {duration} weeks might be tight for {target_km}km. Consider {weeks_needed}+ weeks.")
        
        # Check volume jump
        weekly_needed = target_km * 4  # Rough weekly volume for race distance
        if current_weekly_km < weekly_needed * 0.6:
            recommendations.append(f"💡 Build base to ~{weekly_needed * 0.6:.0f} km/week before adding speed work")
    
    elif objective_details.get('type') == 'fitness':
        goal = objective_details.get('goal', '')
        
        if "Speed" in goal and runs_per_week < 4:
            recommendations.append("💡 Speed development works best with 4+ runs/week (recovery + quality)")
        
        if "VO2max" in goal:
            recommendations.append("💡 VO2max improvement requires 1-2 high-intensity sessions/week")
        
        if "Volume" in goal:
            max_safe_increase = current_weekly_km * 1.5
            recommendations.append(f"💡 Safe peak volume target: ~{max_safe_increase:.0f} km/week over {duration} weeks")
        
        if "Consistency" in goal and runs_per_week > 5:
            warnings.append("⚠️ For consistency, fewer runs (3-4/week) with better adherence beats 6 runs with missed sessions")
    
    elif objective_details.get('type') == 'return':
        prev_vol = objective_details.get('previous_volume', 40)
        
        recommendations.append(f"💡 Start at 50% of previous volume (~{prev_vol * 0.5:.0f} km/week)")
        recommendations.append("💡 Focus on easy pace and consistency for first 4 weeks")
        
        if "Injury" in objective_details.get('reason', ''):
            warnings.append("⚠️ Return from injury: Prioritize ZERO pain over progression speed")
    
    # Check runs per week adequacy
    if runs_per_week < 3:
        warnings.append("⚠️ <3 runs/week limits aerobic development. Consider 3-4 for better results.")
    
    # Display warnings and recommendations
    if warnings:
        for w in warnings:
            st.warning(w)
    
    if recommendations:
        for r in recommendations:
            st.info(r)
    
    if not warnings and not recommendations:
        st.success("✅ Your configuration looks solid! Plan should be safe and effective.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ====================================================================
    # PLAN GENERATION
    # ====================================================================
    
    st.markdown("---")
    
    if st.button("🚀 Generate Intelligent Training Plan", type="primary", width="stretch"):
        with st.spinner("🧠 AI is creating your personalized plan..."):
            
            # Generate plan based on all inputs
            duration = objective_details.get('duration_weeks', 12)
            
            # ----------------------------------------------------------------
            # PLAN GENERATION ALGORITHM
            # ----------------------------------------------------------------
            
            weekly_volumes = []
            
            # Determine target peak volume based on objective
            if objective_details.get('type') == 'race':
                peak_volume = objective_details.get('distance_km', 10) * 4
            elif objective_details.get('type') == 'fitness':
                if "Volume" in objective_details.get('goal', ''):
                    peak_volume = min(current_weekly_km * 1.5, 100)
                elif "Speed" in objective_details.get('goal', ''):
                    peak_volume = current_weekly_km * 1.2  # Less volume, more quality
                else:
                    peak_volume = current_weekly_km * 1.3
            elif objective_details.get('type') == 'return':
                peak_volume = min(objective_details.get('previous_volume', 40), current_weekly_km * 1.8)
            else:
                peak_volume = current_weekly_km * 1.3
            
            # Adjust based on plan style
            if "Minimalist" in plan_style:
                peak_volume = peak_volume * 0.85
            elif "High Performance" in plan_style:
                peak_volume = peak_volume * 1.15
            
            # Generate weekly progression
            if "Structured Classic" in plan_style:
                # Classic 3/1 progression
                build_weeks = duration - 2  # Reserve 2 weeks for taper
                
                for i in range(build_weeks):
                    base_vol = current_weekly_km + (peak_volume - current_weekly_km) * (i / build_weeks)
                    
                    # 3 weeks up, 1 week down
                    if use_recovery_weeks and (i + 1) % 4 == 0:
                        week_vol = base_vol * 0.75  # Recovery week
                    else:
                        week_vol = base_vol
                    
                    weekly_volumes.append(int(week_vol))
                
                # Taper
                weekly_volumes.append(int(peak_volume * 0.7))
                weekly_volumes.append(int(peak_volume * 0.5))
            
            elif "Adaptive AI" in plan_style:
                # Start conservative, will adjust live
                for i in range(duration):
                    base_vol = current_weekly_km + (peak_volume - current_weekly_km) * (i / duration) * 0.9
                    weekly_volumes.append(int(base_vol))
                
                st.info("📱 Adaptive AI: Daily workouts will adjust based on your live data. This is your baseline plan.")
            
            elif "Minimalist" in plan_style:
                # Slower progression, 3 runs/week
                for i in range(duration):
                    week_vol = current_weekly_km + (peak_volume - current_weekly_km) * (i / duration) * 0.85
                    weekly_volumes.append(int(week_vol))
            
            elif "High Performance" in plan_style:
                # Aggressive progression
                build_weeks = duration - 1
                for i in range(build_weeks):
                    week_vol = current_weekly_km + (peak_volume - current_weekly_km) * (i / build_weeks)
                    weekly_volumes.append(int(week_vol))
                weekly_volumes.append(int(peak_volume * 0.6))  # Short taper
            
            elif "Flexible Guidance" in plan_style:
                # Target ranges instead of fixed values
                for i in range(duration):
                    base_vol = current_weekly_km + (peak_volume - current_weekly_km) * (i / duration)
                    weekly_volumes.append(int(base_vol))
                
                st.info("🎯 Flexible mode: These are target volumes. Daily suggestions will help you hit them.")
            
            # ----------------------------------------------------------------
            # RUN SIMULATION
            # ----------------------------------------------------------------
            
            results = st.session_state.simulator.simulate_plan(
                start_state.copy(),
                weekly_volumes,
                objective_details.get('race_date', datetime.now().date() + timedelta(weeks=duration))
            )
            
            # ----------------------------------------------------------------
            # GENERATE INTENSITY DISTRIBUTION
            # ----------------------------------------------------------------
            
            intensity_plans = []
            
            for week_num, week_km in enumerate(weekly_volumes, 1):
                # Determine training phase
                phase_pct = week_num / len(weekly_volumes)
                
                if objective_details.get('type') == 'race':
                    if phase_pct < 0.5:  # Base phase
                        easy, tempo, intervals = 80, 15, 5
                    elif phase_pct < 0.85:  # Build phase
                        easy, tempo, intervals = 70, 15, 15
                    else:  # Taper
                        easy, tempo, intervals = 85, 10, 5
                
                elif objective_details.get('type') == 'fitness':
                    if "Speed" in objective_details.get('goal', ''):
                        easy, tempo, intervals = 65, 20, 15
                    elif "Endurance" in objective_details.get('goal', ''):
                        easy, tempo, intervals = 85, 12, 3
                    elif "VO2max" in objective_details.get('goal', ''):
                        easy, tempo, intervals = 70, 10, 20
                    else:
                        easy, tempo, intervals = 75, 15, 10
                
                elif objective_details.get('type') == 'return':
                    if phase_pct < 0.6:
                        easy, tempo, intervals = 95, 5, 0  # Easy rebuilding
                    else:
                        easy, tempo, intervals = 80, 15, 5  # Add some quality
                else:
                    easy, tempo, intervals = 75, 15, 10
                
                intensity_plans.append({
                    'week': week_num,
                    'total_km': week_km,
                    'easy_km': int(week_km * easy / 100),
                    'tempo_km': int(week_km * tempo / 100),
                    'intervals_km': int(week_km * intervals / 100),
                    'easy_pct': easy,
                    'tempo_pct': tempo,
                    'intervals_pct': intervals
                })
            
            # ----------------------------------------------------------------
            # SAVE TO SESSION STATE
            # ----------------------------------------------------------------
            
            st.session_state.simulation_results = results
            st.session_state.simulation_config = {
                **objective_details,
                'plan_style': plan_style,
                'weekly_volumes': weekly_volumes,
                'runs_per_week': runs_per_week,
                'strength_sessions': strength_per_week,
                'start_state': start_state,
                'intensity_plans': intensity_plans,
                'priority': priority,
                'max_duration': max_session_duration,
                'auto_adjust': auto_adjust,
                'progression_metric': progression_tracking
            }
            
            st.success("✅ Plan generated! Check the **Simulation Results** tab")
            st.balloons()


    
# ========================================================================
# TAB 4: SIMULATION RESULTS (INTELLIGENT COACHING VIEW)
# ========================================================================

with tab4:
    st.header("📈 Your Intelligent Training Plan")
    
    if 'simulation_results' not in st.session_state:
        st.info("👈 Configure and generate a plan in the **Plan Setup** tab first")
        
        # Show example preview
        st.markdown("### What you'll see here:")
        st.markdown("""
        - 📊 **Weekly progression overview** with adaptive insights
        - ⚡ **Intensity distribution** (easy/tempo/intervals)
        - 💪 **Progress tracking** based on your chosen metrics
        - 🤖 **AI coaching recommendations** per week
        - ✅ **Success indicators** beyond race times
        - ⚠️ **Risk warnings** and adaptation suggestions
        """)
    
    else:
        results = st.session_state.simulation_results
        config = st.session_state.simulation_config
        weekly_volumes = config['weekly_volumes']
        intensity_plans = config.get('intensity_plans', [])
        
        # ================================================================
        # HEADER SUMMARY
        # ================================================================
        
        st.subheader("📋 Plan Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if config.get('type') == 'race':
                st.metric("Goal", f"{config['distance_km']} km race")
                st.caption(f"📅 {config['race_date']}")
            elif config.get('type') == 'fitness':
                goal_short = config['goal'].split('-')[0].strip()
                st.metric("Goal", goal_short)
                st.caption(f"⏱️ {config['duration_weeks']} weeks")
            else:
                st.metric("Goal", "Return to Running")
                st.caption(f"⏱️ {config['duration_weeks']} weeks")
        
        with col2:
            st.metric("Plan Style", config.get('plan_style', 'Custom').split('-')[0].strip())
            st.caption(f"🏃 {config['runs_per_week']} runs/week")
        
        with col3:
            st.metric("Total Volume", f"{sum(weekly_volumes)} km")
            st.caption(f"📈 {weekly_volumes[0]} → {max(weekly_volumes)} km/week")
        
        # ================================================================
        # RISK & SAFETY ASSESSMENT
        # ================================================================
        
        st.markdown("---")
        st.subheader("🛡️ Safety & Risk Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            injury_risk = results['max_injury_risk']
            risk_color = "🟢" if injury_risk < 0.2 else "🟡" if injury_risk < 0.4 else "🔴"
            st.metric("Max Injury Risk", f"{risk_color} {injury_risk*100:.0f}%")
            
            if injury_risk < 0.2:
                st.caption("Low risk - plan is conservative")
            elif injury_risk < 0.4:
                st.caption("Moderate - monitor recovery")
            else:
                st.caption("⚠️ High - consider adjustments")
        
        with col2:
            acwr = results['max_acwr']
            acwr_color = "🟢" if acwr < 1.3 else "🟡" if acwr < 1.5 else "🔴"
            st.metric("Max ACWR", f"{acwr_color} {acwr:.2f}")
            
            if acwr < 1.3:
                st.caption("✅ Within safe limits")
            elif acwr < 1.5:
                st.caption("⚠️ Elevated - watch closely")
            else:
                st.caption("🔴 Too high - reduce volume")
        
        with col3:
            st.metric("Total Warnings", len(results['warnings']))
            
            critical = sum(1 for w in results['warnings'] if w.get('severity') == 'critical')
            if critical > 0:
                st.caption(f"🚨 {critical} critical warnings")
            else:
                st.caption("No critical issues")
        
        with col4:
            st.metric("Peak Fitness Week", f"Week {results['peak_fitness_week']}")
            st.caption("Optimal performance window")
        
        # ================================================================
        # WEEKLY PROGRESSION VISUALIZATION
        # ================================================================
        
        st.markdown("---")
        st.subheader("📊 Training Volume & Load Progression")
        
        weeks = list(range(1, len(weekly_volumes) + 1))
        acwr_values = [s.get('acwr', 1.0) for s in results['weekly_states']]
        
        # Create figure with volume and ACWR
        fig = go.Figure()
        
        # Volume bars with color coding
        colors = []
        for i, (vol, acwr) in enumerate(zip(weekly_volumes, acwr_values)):
            if acwr > 1.5 or vol > max(weekly_volumes) * 0.95:
                colors.append('rgba(255, 100, 100, 0.7)')  # Red for high risk
            elif acwr > 1.3 or (i > 0 and vol > weekly_volumes[i-1] * 1.15):
                colors.append('rgba(255, 200, 100, 0.7)')  # Orange for moderate
            else:
                colors.append('rgba(100, 150, 255, 0.7)')  # Blue for safe
        
        fig.add_trace(go.Bar(
            x=weeks,
            y=weekly_volumes,
            name='Weekly Volume (km)',
            marker_color=colors,
            yaxis='y1',
            hovertemplate='Week %{x}<br>Volume: %{y} km<extra></extra>'
        ))
        
        # ACWR line
        fig.add_trace(go.Scatter(
            x=weeks,
            y=acwr_values,
            name='ACWR (Acute:Chronic)',
            yaxis='y2',
            line=dict(color='orange', width=3),
            hovertemplate='Week %{x}<br>ACWR: %{y:.2f}<extra></extra>'
        ))
        
        # Safe ACWR threshold
        fig.add_trace(go.Scatter(
            x=weeks,
            y=[1.3] * len(weeks),
            name='Safe ACWR Limit',
            yaxis='y2',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Safe limit: 1.3<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis=dict(title='Week', dtick=1),
            yaxis=dict(title='Volume (km)', side='left'),
            yaxis2=dict(title='ACWR', overlaying='y', side='right', range=[0, max(2, max(acwr_values) * 1.1)]),
            hovermode='x unified',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # ================================================================
        # INTENSITY DISTRIBUTION OVERVIEW
        # ================================================================
        
        st.markdown("---")
        st.subheader("⚡ Training Intensity Distribution")
        
        if intensity_plans:
            # Summary stats
            total_easy = sum(w['easy_km'] for w in intensity_plans)
            total_tempo = sum(w['tempo_km'] for w in intensity_plans)
            total_intervals = sum(w['intervals_km'] for w in intensity_plans)
            total_km = total_easy + total_tempo + total_intervals
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Easy (Z2)", f"{total_easy} km")
                st.caption(f"{total_easy/total_km*100:.0f}% of volume")
            
            with col2:
                st.metric("Total Tempo (Z3)", f"{total_tempo} km")
                st.caption(f"{total_tempo/total_km*100:.0f}% of volume")
            
            with col3:
                st.metric("Total Intervals (Z4-5)", f"{total_intervals} km")
                st.caption(f"{total_intervals/total_km*100:.0f}% of volume")
            
            with col4:
                # Polarization index (should be ~80/20)
                easy_pct = total_easy / total_km * 100
                hard_pct = (total_tempo + total_intervals) / total_km * 100
                
                if 75 <= easy_pct <= 85:
                    st.success(f"✅ Well polarized")
                    st.caption(f"{easy_pct:.0f}% easy / {hard_pct:.0f}% quality")
                else:
                    st.warning(f"⚠️ Check balance")
                    st.caption(f"{easy_pct:.0f}% easy / {hard_pct:.0f}% quality")
            
            # Intensity pie chart
            fig_intensity = go.Figure(data=[go.Pie(
                labels=['Easy (Z2)', 'Tempo (Z3)', 'Intervals (Z4-5)'],
                values=[total_easy, total_tempo, total_intervals],
                hole=0.4,
                marker=dict(colors=['#87CEEB', '#FFD700', '#FF6B6B']),
                textposition='inside',
                textinfo='label+percent'
            )])
            
            fig_intensity.update_layout(
                title='Overall Intensity Distribution',
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_intensity, width="stretch")
        
        # ================================================================
        # WEEKLY BREAKDOWN WITH AI COACHING
        # ================================================================
        
        st.markdown("---")
        st.subheader("📅 Week-by-Week Coaching Plan")
        
        st.markdown("**Expand any week to see detailed breakdown and AI recommendations:**")
        
        for week_num, week_km in enumerate(weekly_volumes, 1):
            week_state = results['weekly_states'][week_num - 1]
            week_intensity = intensity_plans[week_num - 1] if week_num <= len(intensity_plans) else None
            
            # Week warnings
            week_warnings = [w for w in results['warnings'] if w.get('week') == week_num]
            
            # Determine week type
            if week_num == 1:
                week_type = "🚀 Build Base"
            elif week_num == len(weekly_volumes):
                week_type = "🎯 Race Week" if config.get('type') == 'race' else "💪 Peak Week"
            elif week_num == len(weekly_volumes) - 1:
                week_type = "📉 Taper"
            elif week_num > 0 and week_km < weekly_volumes[week_num - 2] * 0.9:
                week_type = "🔋 Recovery Week"
            else:
                week_type = "📈 Build Week"
            
            # Warning indicator
            if len(week_warnings) > 0:
                critical = any(w.get('severity') == 'critical' for w in week_warnings)
                if critical:
                    status_icon = "🚨"
                elif any(w.get('severity') == 'high' for w in week_warnings):
                    status_icon = "⚠️"
                else:
                    status_icon = "⚡"
            else:
                status_icon = "✅"
            
            # Week header
            with st.expander(f"{status_icon} **Week {week_num}** - {week_km} km - {week_type}", expanded=(week_num <= 2)):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Intensity breakdown
                    if week_intensity:
                        st.markdown("**📊 Intensity Breakdown:**")
                        
                        intensity_df = pd.DataFrame([{
                            'Zone': 'Easy (Z2)',
                            'Volume': f"{week_intensity['easy_km']} km",
                            'Percentage': f"{week_intensity['easy_pct']}%",
                            'Purpose': 'Aerobic base, recovery'
                        }, {
                            'Zone': 'Tempo (Z3)',
                            'Volume': f"{week_intensity['tempo_km']} km",
                            'Percentage': f"{week_intensity['tempo_pct']}%",
                            'Purpose': 'Lactate threshold'
                        }, {
                            'Zone': 'Intervals (Z4-5)',
                            'Volume': f"{week_intensity['intervals_km']} km",
                            'Percentage': f"{week_intensity['intervals_pct']}%",
                            'Purpose': 'VO2max, speed'
                        }])
                        
                        st.dataframe(intensity_df, width="stretch", hide_index=True)
                    
                    # Session structure example
                    st.markdown("**🗓️ Example Weekly Structure:**")
                    
                    runs_this_week = config.get('runs_per_week', 4)
                    strength_this_week = config.get('strength_sessions', 1)
                    
                    if week_intensity:
                        long_run_km = int(week_km * 0.30)
                        quality_km = week_intensity['intervals_km'] + week_intensity['tempo_km']
                        easy_km_total = week_km - long_run_km - quality_km
                        easy_per_run = int(easy_km_total / max(1, runs_this_week - 2))
                        
                        sample_structure = []
                        
                        # Long run
                        sample_structure.append(f"• **Long Run:** {long_run_km} km @ Z2 (conversational pace)")
                        
                        # Quality session
                        if week_intensity['intervals_km'] > 0:
                            sample_structure.append(f"• **Interval Session:** {week_intensity['intervals_km']} km intervals (e.g., 6×1k @ Z4 with 2min rest)")
                        
                        if week_intensity['tempo_km'] > 0 and week_intensity['tempo_km'] != week_intensity['intervals_km']:
                            sample_structure.append(f"• **Tempo Run:** {week_intensity['tempo_km']} km @ Z3 (comfortably hard)")
                        
                        # Easy runs
                        easy_runs = max(0, runs_this_week - 2)
                        if easy_runs > 0:
                            sample_structure.append(f"• **Easy Runs (×{easy_runs}):** ~{easy_per_run} km each @ Z2")
                        
                        # Strength
                        if strength_this_week > 0:
                            sample_structure.append(f"• **Strength/Cross-training (×{strength_this_week}):** 30-45 min")
                        
                        # Rest
                        rest_days = 7 - runs_this_week - strength_this_week
                        if rest_days > 0:
                            sample_structure.append(f"• **Rest Days:** {rest_days} day(s)")
                        
                        for item in sample_structure:
                            st.markdown(item)
                
                with col2:
                    # Predicted state
                    st.markdown("**📈 Predicted State:**")
                    
                    recovery_pct = week_state.get('recovery_score', 0.7) * 100
                    hrv = week_state.get('hrv_rmssd', 80)
                    acwr = week_state.get('acwr', 1.0)
                    
                    st.metric("Recovery", f"{recovery_pct:.0f}%")
                    st.metric("HRV", f"{hrv:.0f} ms")
                    st.metric("ACWR", f"{acwr:.2f}")
                    
                    # Fatigue indicator
                    if recovery_pct < 50:
                        st.error("🔴 High fatigue")
                    elif recovery_pct < 70:
                        st.warning("🟡 Moderate fatigue")
                    else:
                        st.success("🟢 Good recovery")
                
                # AI Coaching Recommendations
                st.markdown("---")
                st.markdown("**🤖 AI Coach Recommendations:**")
                
                recommendations = []
                
                # Generate contextual recommendations
                if week_num == 1:
                    recommendations.append("💡 **Week 1 Focus:** Establish baseline pace zones. All runs should feel easy - you should be able to hold a conversation.")
                
                if week_type == "🔋 Recovery Week":
                    recommendations.append("🔋 **Recovery Week:** Reduced volume to allow adaptation. Focus on sleep, nutrition, and easy running.")
                
                if week_state.get('acwr', 1.0) > 1.3:
                    recommendations.append("⚠️ **ACWR Elevated:** Load is building quickly. Prioritize sleep (8+ hours) and monitor morning HR/HRV.")
                
                if week_intensity and week_intensity['intervals_pct'] > 15:
                    recommendations.append("⚡ **High Intensity Week:** Quality over quantity. Ensure 48h between hard sessions. Easy runs should stay VERY easy.")
                
                if week_km > (weekly_volumes[week_num - 2] if week_num > 1 else 0) * 1.1:
                    increase_pct = ((week_km / weekly_volumes[week_num - 2]) - 1) * 100 if week_num > 1 else 0
                    recommendations.append(f"📈 **Volume Jump:** +{increase_pct:.0f}% from last week. Watch for unusual soreness or fatigue.")
                
                if week_num == len(weekly_volumes) - 2:
                    recommendations.append("📉 **Taper Begins:** Volume drops but intensity stays sharp. Trust the process - legs will feel fresh soon.")
                
                if week_num == len(weekly_volumes) and config.get('type') == 'race':
                    recommendations.append("🎯 **Race Week:** Minimal running, maximum rest. Visualize success. Trust your training.")
                
                # Progress tracking insight
                if week_num % 4 == 0:
                    total_so_far = sum(weekly_volumes[:week_num])
                    avg_weekly = total_so_far / week_num
                    recommendations.append(f"📊 **Progress Check:** You've completed {total_so_far} km total (avg {avg_weekly:.0f} km/week). Great consistency!")
                
                # Display recommendations
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.info("💪 Solid week - stick to the plan and listen to your body.")
                
                # Warnings
                if week_warnings:
                    st.markdown("**⚠️ Risk Warnings:**")
                    for warning in week_warnings:
                        severity = warning.get('severity', 'medium')
                        message = warning.get('message', '')
                        
                        if severity == 'critical':
                            st.error(f"🚨 {message}")
                        elif severity == 'high':
                            st.warning(f"⚠️ {message}")
                        else:
                            st.info(f"⚡ {message}")
                
                # Adaptive adjustments (if enabled)
                if config.get('auto_adjust'):
                    st.markdown("---")
                    st.markdown("**🔄 Adaptive Adjustments:**")
                    
                    # Simulate potential adjustments based on state
                    if week_state.get('hrv_rmssd', 80) < 70:
                        st.caption("📉 If HRV drops below baseline → reduce volume by 10-15%")
                    
                    if week_state.get('resting_hr', 45) > 50:
                        st.caption("💓 If resting HR elevated → convert hard session to easy run")
                    
                    if week_state.get('consecutive_hard_days', 0) >= 3:
                        st.caption("🛑 If 3+ hard days detected → mandatory rest day inserted")
                    
                    st.caption("✅ Plan will auto-adjust based on your live data each week")
        
        # ================================================================
        # PROGRESS TRACKING METRICS
        # ================================================================
        
        st.markdown("---")
        st.subheader("📈 Progress Tracking & Success Indicators")
        
        st.markdown(f"**Primary Metric:** {config.get('progression_metric', 'Overall fitness')}")
        
        # Recovery & physiological trends
        recovery_scores = [s.get('recovery_score', 0.7) * 100 for s in results['weekly_states']]
        hrv_values = [s.get('hrv_rmssd', 80) for s in results['weekly_states']]
        
        fig_progress = go.Figure()
        
        # Recovery score
        fig_progress.add_trace(go.Scatter(
            x=weeks,
            y=recovery_scores,
            name='Recovery Score (%)',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)',
            yaxis='y1'
        ))
        
        # HRV trend
        fig_progress.add_trace(go.Scatter(
            x=weeks,
            y=hrv_values,
            name='HRV (ms)',
            yaxis='y2',
            line=dict(color='purple', width=2)
        ))
        
        # HRV baseline
        hrv_baseline = hrv_values[0]
        fig_progress.add_trace(go.Scatter(
            x=weeks,
            y=[hrv_baseline] * len(weeks),
            name='HRV Baseline',
            yaxis='y2',
            line=dict(color='purple', width=1, dash='dash'),
            opacity=0.5
        ))
        
        fig_progress.update_layout(
            title='Recovery & HRV Progression',
            xaxis=dict(title='Week', dtick=1),
            yaxis=dict(title='Recovery Score (%)', range=[0, 100]),
            yaxis2=dict(title='HRV (ms)', overlaying='y', side='right'),
            hovermode='x unified',
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_progress, width="stretch")
        
        # ================================================================
        # SUCCESS INDICATORS (BEYOND RACE TIME)
        # ================================================================
        
        st.markdown("---")
        st.subheader("✅ Success Indicators")
        
        st.markdown("**Your progress will be measured by:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Physiological improvements
            st.markdown("**📊 Physiological Markers:**")
            
            hrv_change = hrv_values[-1] - hrv_values[0]
            recovery_change = recovery_scores[-1] - recovery_scores[0]
            
            success_items = []
            
            if hrv_change > 5:
                success_items.append(f"✅ HRV improved by {hrv_change:.0f} ms (better recovery capacity)")
            elif hrv_change > -5:
                success_items.append(f"✓ HRV stable (maintained recovery capacity)")
            else:
                success_items.append(f"⚠️ HRV decreased - may indicate overtraining")
            
            if recovery_change > 0:
                success_items.append(f"✅ Recovery score improved by {recovery_change:.0f}%")
            else:
                success_items.append(f"✓ Recovery managed despite increased load")
            
            # Volume completion
            total_planned = sum(weekly_volumes)
            success_items.append(f"🎯 Complete {total_planned} km total volume")
            
            # Consistency
            success_items.append(f"📅 Maintain {config.get('runs_per_week', 4)} runs/week consistency")
            
            for item in success_items:
                st.markdown(f"- {item}")
        
        with col2:
            # Performance indicators
            st.markdown("**🏃 Performance Indicators:**")
            
            perf_items = []
            
            if config.get('type') == 'race':
                if config.get('target_time'):
                    perf_items.append(f"🎯 Target finish time: {config['target_time']}")
                else:
                    perf_items.append(f"🎯 Complete {config['distance_km']} km comfortably")
                
                perf_items.append(f"📈 Peak week volume: {max(weekly_volumes)} km")
            
            elif config.get('type') == 'fitness':
                goal = config['goal']
                
                if "Endurance" in goal:
                    perf_items.append("✅ Increase longest run by 30-50%")
                    perf_items.append("📈 Heart rate at same pace decreases 5-10 bpm")
                
                elif "Speed" in goal:
                    perf_items.append("⚡ Improve 5K pace by 10-20 sec/km")
                    perf_items.append("📊 Complete interval sessions at target pace")
                
                elif "Volume" in goal:
                    target_vol = max(weekly_volumes)
                    perf_items.append(f"📈 Reach {target_vol} km/week sustainably")
                    perf_items.append("✅ No injury/overtraining symptoms")
                
                elif "VO2max" in goal:
                    perf_items.append("⚡ Complete 8-12 VO2max intervals sessions")
                    perf_items.append("📈 Garmin VO2max estimate increases")
                
                elif "Consistency" in goal:
                    perf_items.append(f"✅ Complete 90%+ of planned runs")
                    perf_items.append("📅 No missed weeks")
                
                else:
                    perf_items.append("✅ Complete training block without injury")
                    perf_items.append("📈 Increase fitness level sustainably")
            
            elif config.get('type') == 'return':
                perf_items.append("✅ Zero pain during/after runs")
                perf_items.append(f"📈 Return to {config.get('previous_volume', 40) * 0.8:.0f} km/week safely")
                perf_items.append("🛡️ Rebuild confidence and consistency")
            
            for item in perf_items:
                st.markdown(f"- {item}")
        
        # Overall plan health score
        st.markdown("---")
        st.markdown("**🎯 Overall Plan Health Score:**")
        
        # Calculate score
        score_components = []
        
        # Safety score (50 points)
        if results['max_injury_risk'] < 0.2:
            safety_score = 50
        elif results['max_injury_risk'] < 0.4:
            safety_score = 35
        else:
            safety_score = 20
        score_components.append(("Safety", safety_score, 50))
        
        # Progression appropriateness (30 points)
        max_jump = 0
        for i in range(1, len(weekly_volumes)):
            jump = (weekly_volumes[i] - weekly_volumes[i-1]) / weekly_volumes[i-1]
            max_jump = max(max_jump, jump)
        
        if max_jump < 0.10:
            progression_score = 30
        elif max_jump < 0.15:
            progression_score = 22
        else:
            progression_score = 15
        score_components.append(("Progression", progression_score, 30))
        
        # Recovery management (20 points)
        avg_recovery = sum(recovery_scores) / len(recovery_scores)
        if avg_recovery > 70:
            recovery_mgmt_score = 20
        elif avg_recovery > 50:
            recovery_mgmt_score = 14
        else:
            recovery_mgmt_score = 8
        score_components.append(("Recovery", recovery_mgmt_score, 20))
        
        total_score = sum(s[1] for s in score_components)
        max_score = sum(s[2] for s in score_components)
        
        # Display score
        score_pct = (total_score / max_score) * 100
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Progress bar
            st.progress(score_pct / 100)
            
            if score_pct >= 85:
                st.success(f"✅ **Excellent Plan** ({score_pct:.0f}/100)")
                st.caption("This plan is well-balanced, safe, and effective")
            elif score_pct >= 70:
                st.info(f"✓ **Good Plan** ({score_pct:.0f}/100)")
                st.caption("Solid plan with minor adjustments possible")
            else:
                st.warning(f"⚠️ **Needs Improvement** ({score_pct:.0f}/100)")
                st.caption("Consider reducing volume jumps or adding recovery")
        
        with col2:
            for name, score, max_s in score_components:
                st.caption(f"{name}: {score}/{max_s}")
        
        with col3:
            st.caption(f"Safety: {safety_score}/50")
            st.caption(f"Progression: {progression_score}/30")
            st.caption(f"Recovery: {recovery_mgmt_score}/20")
        
        # ================================================================
        # EXPORT OPTIONS
        # ================================================================
        
        st.markdown("---")
        st.subheader("📥 Export Your Plan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export with intensity
            export_data = []
            for i, week in enumerate(weeks):
                week_data = {
                    'Week': week,
                    'Volume_km': weekly_volumes[i],
                    'Easy_km': intensity_plans[i]['easy_km'] if i < len(intensity_plans) else 0,
                    'Tempo_km': intensity_plans[i]['tempo_km'] if i < len(intensity_plans) else 0,
                    'Intervals_km': intensity_plans[i]['intervals_km'] if i < len(intensity_plans) else 0,
                    'ACWR': acwr_values[i],
                    'Recovery_%': recovery_scores[i],
                    'HRV_ms': hrv_values[i]
                }
                export_data.append(week_data)
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                "📊 Download Full Plan (CSV)",
                data=csv,
                file_name=f"training_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch"
            )
        
        with col2:
            # Weekly summary text
            summary_text = f"""INTELLIGENT TRAINING PLAN
{'='*60}

GOAL: {config.get('goal', 'Custom training')}
DURATION: {len(weekly_volumes)} weeks
STYLE: {config.get('plan_style', 'Custom')}

VOLUME PROGRESSION:
Start: {weekly_volumes[0]} km
Peak: {max(weekly_volumes)} km
Total: {sum(weekly_volumes)} km

INTENSITY DISTRIBUTION:
Easy: {total_easy} km ({total_easy/total_km*100:.0f}%)
Tempo: {total_tempo} km ({total_tempo/total_km*100:.0f}%)
Intervals: {total_intervals} km ({total_intervals/total_km*100:.0f}%)

WEEKLY PLAN:
{chr(10).join([f'Week {i+1}: {vol} km - {intensity_plans[i]["easy_km"]}Z2 + {intensity_plans[i]["tempo_km"]}Z3 + {intensity_plans[i]["intervals_km"]}Z4-5' for i, vol in enumerate(weekly_volumes) if i < len(intensity_plans)])}

SAFETY ASSESSMENT:
Max Injury Risk: {results['max_injury_risk']*100:.0f}%
Max ACWR: {results['max_acwr']:.2f}
Total Warnings: {len(results['warnings'])}

SUCCESS INDICATORS:
- Complete {sum(weekly_volumes)} km total
- Maintain {config.get('runs_per_week', 4)} runs/week
- HRV improvement target: +{hrv_values[0]*0.1:.0f} ms
- Recovery stability: >65% average

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            
            st.download_button(
                "📄 Download Summary (TXT)",
                data=summary_text,
                file_name=f"plan_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                width="stretch"
            )
        
        with col3:
            # Calendar export (ICS format - future enhancement)
            st.button(
                "📅 Export to Calendar",
                width="stretch",
                disabled=True,
                help="Coming soon: Add workouts to Google Calendar/Apple Calendar"
            )

# ========================================================================
# TAB 5: RACE DAY OPTIMIZER & PREDICTION ENGINE
# ========================================================================

with tab5:
    st.header("🏁 Race Strategy & Performance Predictor")
    
    st.markdown("""
    Predict your race performance, optimize your strategy, and explore "what if" scenarios based on your training plan.
    """)
    
    # ====================================================================
    # SECTION 1: RACE SELECTION & TIMING
    # ====================================================================
    
    st.subheader("1️⃣ Select Your Race")
    
    # Check if user has a plan
    has_plan = 'simulation_config' in st.session_state
    
    if has_plan:
        config = st.session_state.simulation_config
        
        # Show current plan summary
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **Current Training Plan:**
        - Goal: {config.get('goal', 'Custom training') if config.get('type') != 'race' else f"{config['distance_km']} km race"}
        - Duration: {config.get('duration_weeks', 12)} weeks
        - Peak volume: {max(config['weekly_volumes'])} km/week
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Race configuration
    race_mode = st.radio(
        "What do you want to predict?",
        [
            "🎯 Specific Upcoming Race",
            "📊 Performance at Different Weeks",
            "🔮 Multiple Race Scenarios"
        ],
        horizontal=False
    )
    
    # ====================================================================
    # MODE A: SPECIFIC RACE PREDICTION
    # ====================================================================

    if race_mode == "🎯 Specific Upcoming Race":
        st.markdown("---")
        st.subheader("Race Details")
        
        # Define defaults FIRST (before any widgets use them)
        if has_plan and config.get('type') == 'race':
            default_distance = float(config['distance_km'])
            
            # Handle race_date conversion (might be string, datetime, or date)
            race_date_raw = config.get('race_date')
            if race_date_raw:
                if isinstance(race_date_raw, str):
                    try:
                        default_date = datetime.strptime(race_date_raw, '%Y-%m-%d').date()
                    except:
                        default_date = datetime.now().date() + timedelta(weeks=12)
                elif isinstance(race_date_raw, datetime):
                    default_date = race_date_raw.date()
                else:
                    default_date = race_date_raw
            else:
                default_date = datetime.now().date() + timedelta(weeks=12)
            
            default_elevation = int(config.get('elevation_m', 0))
            default_terrain = config.get('terrain', 'Road')
        else:
            default_distance = 10.0
            default_date = datetime.now().date() + timedelta(weeks=12)
            default_elevation = 0
            default_terrain = 'Road'
        
        # Now create the widgets
        col1, col2, col3 = st.columns(3)
        
        with col1:
            race_distance_km = st.number_input(
                "Race Distance (km)",
                value=default_distance,
                min_value=1.0,
                max_value=100.0,
                step=0.5,
                key="race_optimizer_distance"
            )
            
            elevation_gain_m = st.number_input(
                "Elevation Gain (m)",
                value=default_elevation,
                min_value=0,
                max_value=5000,
                step=50,
                key="race_optimizer_elevation"
            )
        
        with col2:
            race_date = st.date_input(
                "Race Date",
                value=default_date,
                key="race_optimizer_date"
            )
            
            terrain_options = ["Road", "Trail - Easy", "Trail - Technical", "Mountain/Ultra", "Track"]
            terrain_index = terrain_options.index(default_terrain) if default_terrain in terrain_options else 0
            
            terrain_type = st.selectbox(
                "Terrain",
                terrain_options,
                index=terrain_index,
                key="race_optimizer_terrain_type"
            )
        
        with col3:
            weeks_until = (race_date - datetime.now().date()).days // 7
            st.metric("Weeks Until Race", weeks_until)
            
            # Expected conditions
            expected_temp = st.slider(
                "Expected Temperature (°C)",
                min_value=0,
                max_value=35,
                value=15,
                key="race_optimizer_temp"
            )
        
        # Calculate fitness at race date
        st.markdown("---")
        st.subheader("📈 Predicted Fitness at Race Date")
        
        if has_plan and 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results
            weekly_volumes = config['weekly_volumes']
            
            # Find fitness at race week
            plan_duration = len(weekly_volumes)
            
            if weeks_until <= plan_duration:
                race_week_idx = weeks_until - 1
                race_week_state = results['weekly_states'][race_week_idx]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    fitness_pct = race_week_state.get('recovery_score', 0.7) * 100
                    st.metric("Fitness Level", f"{fitness_pct:.0f}%")
                
                with col2:
                    hrv = race_week_state.get('hrv_rmssd', 80)
                    st.metric("Predicted HRV", f"{hrv:.0f} ms")
                
                with col3:
                    volume_at_race = weekly_volumes[race_week_idx]
                    st.metric("Weekly Volume", f"{volume_at_race} km")
                
                with col4:
                    acwr = race_week_state.get('acwr', 1.0)
                    st.metric("ACWR", f"{acwr:.2f}")
                
                # Readiness assessment
                st.markdown("**Race Readiness:**")
                
                readiness_score = 0
                readiness_factors = []
                
                # Fitness
                if fitness_pct > 80:
                    readiness_score += 30
                    readiness_factors.append("✅ High fitness level")
                elif fitness_pct > 65:
                    readiness_score += 20
                    readiness_factors.append("✓ Good fitness level")
                else:
                    readiness_score += 10
                    readiness_factors.append("⚠️ Moderate fitness - may need more time")
                
                # ACWR (taper check)
                if 0.7 <= acwr <= 1.0:
                    readiness_score += 25
                    readiness_factors.append("✅ Perfect taper (fresh but fit)")
                elif acwr < 0.7:
                    readiness_score += 15
                    readiness_factors.append("⚠️ Possibly over-tapered")
                else:
                    readiness_score += 10
                    readiness_factors.append("⚠️ Still carrying fatigue")
                
                # Volume trend
                if race_week_idx > 0:
                    prev_week_vol = weekly_volumes[race_week_idx - 1]
                    taper_pct = (volume_at_race / prev_week_vol) if prev_week_vol > 0 else 1
                    
                    if 0.4 <= taper_pct <= 0.7:
                        readiness_score += 25
                        readiness_factors.append("✅ Good taper (40-70% volume)")
                    elif taper_pct < 0.4:
                        readiness_score += 15
                        readiness_factors.append("⚠️ Aggressive taper")
                    else:
                        readiness_score += 10
                        readiness_factors.append("⚠️ Insufficient taper")
                
                # HRV
                if hrv > 75:
                    readiness_score += 20
                    readiness_factors.append("✅ Excellent recovery (HRV elevated)")
                elif hrv > 65:
                    readiness_score += 15
                    readiness_factors.append("✓ Good recovery")
                else:
                    readiness_score += 8
                    readiness_factors.append("⚠️ Monitor recovery closely")
                
                # Display readiness
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Readiness Score", f"{readiness_score}/100")
                    
                    if readiness_score >= 80:
                        st.success("🟢 Excellent - Ready to race!")
                    elif readiness_score >= 65:
                        st.info("🟡 Good - Should perform well")
                    else:
                        st.warning("🟠 Fair - Consider adjusting plan")
                
                with col2:
                    for factor in readiness_factors:
                        st.caption(factor)
            
            else:
                st.warning(f"⚠️ Race is {weeks_until} weeks away, but your current plan is only {plan_duration} weeks. Extend your plan or choose a closer race.")
        
        else:
            st.info("💡 Create a training plan first to see predicted fitness at race date")
        
        # ================================================================
        # PERFORMANCE PREDICTION
        # ================================================================
        
        st.markdown("---")
        st.subheader("⏱️ Predicted Race Time")
        
        if has_plan and 'simulation_results' in st.session_state and weeks_until <= plan_duration:
            
            # Base prediction on current fitness and training volume
            current_volume_avg = sum(weekly_volumes[:race_week_idx+1]) / (race_week_idx + 1)
            
            # Estimate VDOT / running fitness
            # Simple heuristic: weekly volume correlates with endurance capacity
            base_pace_min_km = 5.5 - (current_volume_avg / 100) * 1.5  # Faster with more volume
            
            # Adjust for fitness level
            fitness_multiplier = (fitness_pct / 100)
            adjusted_pace = base_pace_min_km / fitness_multiplier
            
            # Adjust for terrain
            terrain_factors = {
                "Road": 1.0,
                "Trail - Easy": 1.08,
                "Trail - Technical": 1.15,
                "Mountain/Ultra": 1.25,
                "Track": 0.98
            }
            terrain_adjustment = terrain_factors.get(terrain_type, 1.0)
            
            # Adjust for elevation
            elevation_penalty_per_100m = 0.015  # 1.5% slower per 100m gain
            elevation_factor = 1 + (elevation_gain_m / 100) * elevation_penalty_per_100m
            
            # Adjust for weather
            if expected_temp > 20:
                temp_penalty = 1 + ((expected_temp - 20) * 0.01)  # 1% slower per degree over 20°C
            elif expected_temp < 10:
                temp_penalty = 1 + ((10 - expected_temp) * 0.005)  # 0.5% slower per degree under 10°C
            else:
                temp_penalty = 1.0
            
            # Final predicted pace
            predicted_pace = adjusted_pace * terrain_adjustment * elevation_factor * temp_penalty
            predicted_time_min = predicted_pace * race_distance_km
            
            # Confidence interval (±5%)
            best_case_time = predicted_time_min * 0.95
            worst_case_time = predicted_time_min * 1.05
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Finish Time",
                    format_time(predicted_time_min),
                    help="Based on your training and conditions"
                )
                st.caption(f"~{predicted_pace:.2f} min/km average pace")
            
            with col2:
                st.metric(
                    "Best Case Scenario",
                    format_time(best_case_time),
                    help="Perfect execution + conditions"
                )
                st.caption("Everything goes right")
            
            with col3:
                st.metric(
                    "Worst Case Scenario",
                    format_time(worst_case_time),
                    help="Conservative/tough conditions"
                )
                st.caption("Rough day / heat / hills")
            
            # Confidence breakdown
            st.markdown("**Prediction Factors:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption(f"📊 Base pace (volume-adjusted): {base_pace_min_km:.2f} min/km")
                st.caption(f"💪 Fitness multiplier: {fitness_multiplier:.2f}x")
                st.caption(f"🏔️ Terrain factor: {terrain_adjustment:.2f}x")
            
            with col2:
                st.caption(f"⛰️ Elevation penalty: {elevation_factor:.2f}x (+{elevation_gain_m}m)")
                st.caption(f"🌡️ Temperature factor: {temp_penalty:.2f}x ({expected_temp}°C)")
                st.caption(f"📈 Training weeks: {race_week_idx + 1}")
            
            st.info(f"💡 Confidence: Moderate - based on training simulation. Actual performance depends on race-day execution, nutrition, pacing, and conditions.")
        
        else:
            st.warning("⚠️ Create a training plan to see performance predictions")
        
        # ================================================================
        # RACE STRATEGY
        # ================================================================
        
        st.markdown("---")
        st.subheader("🎯 Optimal Race Strategy")
        
        if has_plan and 'simulation_results' in st.session_state and weeks_until <= plan_duration:
            
            optimizer = st.session_state.race_optimizer
            
            # Pacing strategy
            pacing = optimizer.calculate_pacing_strategy(
                race_distance_km,
                predicted_time_min,
                elevation_gain_m
            )
            
            st.markdown(f"**Recommended: {pacing['strategy'].replace('_', ' ').title()}**")
            
            # Pacing by quarters
            pacing_df = pd.DataFrame(pacing['quarters'])
            
            st.dataframe(pacing_df, width="stretch", hide_index=True)
            
            # Nutrition strategy
            st.markdown("---")
            st.markdown("**🍌 Nutrition & Fueling:**")
            
            energy = optimizer.estimate_energy_needs(
                race_distance_km,
                predicted_time_min,
                body_weight_kg=70
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Pre-Race (3h before):**")
                st.write("• 80-100g carbs (oats, toast, banana)")
                st.write("• 400-500 mL water")
                st.write("• Familiar foods only")
                
                st.markdown("**Pre-Start (30min):**")
                st.write("• 100-200 mL water/electrolytes")
                st.write("• Optional: 1 gel if nervous stomach")
            
            with col2:
                st.markdown("**During Race:**")
                
                if energy['needs_fueling']:
                    st.write(f"• Total gels needed: {energy['total_gels']}")
                    st.write(f"• Carbs per hour: {energy['carbs_per_hour_g']}g")
                    
                    if energy['total_gels'] > 0:
                        st.write("\n**Gel timing:**")
                        for i, timing in enumerate(energy['gel_timing_min'], 1):
                            km_mark = timing * race_distance_km / predicted_time_min
                            st.write(f"  - Gel #{i}: ~{timing:.0f} min (km {km_mark:.1f})")
                else:
                    st.success(f"✅ No fueling needed for {race_distance_km} km")
                    st.write("• Sip water if available and thirsty")
            
            # Warm-up
            st.markdown("---")
            st.markdown("**🔥 Pre-Race Warm-up:**")
            
            warmup = optimizer.warmup_protocol(race_distance_km)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Duration", f"{warmup['duration_min']} min")
                
                race_start_time = st.time_input(
                    "Race Start Time",
                    value=datetime.strptime("08:00", "%H:%M").time(),
                    key="race_optimizer_start_time"  # ← ADD THIS
                )

                warmup_start = (datetime.combine(datetime.today(), race_start_time) - timedelta(minutes=warmup['duration_min'])).time()
                st.caption(f"Start warm-up at: {warmup_start.strftime('%H:%M')}")
            
            with col2:
                st.markdown("**Warm-up Structure:**")
                for step in warmup['structure']:
                    st.write(f"• {step}")
            
            # ================================================================
            # MONTE CARLO SIMULATION
            # ================================================================
            
            st.markdown("---")
            st.subheader("🎲 Race Outcome Simulation (1000 scenarios)")
            
            st.markdown("""
            Run Monte Carlo simulation to understand the range of possible outcomes based on:
            - Weather variability
            - Pacing execution accuracy
            - Nutrition timing
            - Hydration status
            - Energy availability
            """)
            
            # Configuration
            col1, col2 = st.columns(2)
            
            with col1:
                breakfast_hours = st.slider(
                    "Breakfast timing (hours before)",
                    min_value=2.0,
                    max_value=4.0,
                    value=2.75,
                    step=0.25,
                    key="race_optimizer_breakfast"  # ← ADD THIS
                )

            
            with col2:
                gel_strategy = st.selectbox(
                    "Fueling Strategy",
                    ["No gels", "Conservative (1 gel)", "Moderate (2 gels)", "Aggressive (3+ gels)"],
                    key="race_optimizer_gel_strategy"  # ← ADD THIS
                )

                
                gel_count_map = {"No gels": 0, "Conservative (1 gel)": 1, "Moderate (2 gels)": 2, "Aggressive (3+ gels)": 3}
                gel_count = gel_count_map[gel_strategy]
            
            if st.button("🎲 Run Simulation (1000 scenarios)", type="primary"):
                with st.spinner("Running 1000 virtual races..."):
                    mc_results = optimizer.monte_carlo_simulation(
                        race_distance_km,
                        predicted_time_min,
                        elevation_gain_m,
                        breakfast_hours,
                        gel_count,
                        n_simulations=1000
                    )
                    
                    st.session_state.mc_results = mc_results
                
                st.success("✅ Simulation complete!")
            
            # Display results
            if 'mc_results' in st.session_state:
                mc = st.session_state.mc_results
                
                st.markdown("---")
                st.markdown("### 📊 Simulation Results")
                
                # Key risks
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    success_rate = mc['success_rate_pct']
                    success_color = "🟢" if success_rate > 70 else "🟡" if success_rate > 50 else "🔴"
                    st.metric("Success Rate", f"{success_color} {success_rate:.0f}%")
                    st.caption("Within ±2% of target")
                
                with col2:
                    bonk_risk = mc['bonk_risk_pct']
                    bonk_color = "🟢" if bonk_risk < 10 else "🟡" if bonk_risk < 25 else "🔴"
                    st.metric("Bonk Risk", f"{bonk_color} {bonk_risk:.0f}%")
                
                with col3:
                    gi_risk = mc['gi_distress_pct']
                    gi_color = "🟢" if gi_risk < 15 else "🟡" if gi_risk < 30 else "🔴"
                    st.metric("GI Distress Risk", f"{gi_color} {gi_risk:.0f}%")
                
                with col4:
                    dehydration_risk = mc['dehydration_pct']
                    dehydration_color = "🟢" if dehydration_risk < 10 else "🟡" if dehydration_risk < 20 else "🔴"
                    st.metric("Dehydration Risk", f"{dehydration_color} {dehydration_risk:.0f}%")
                
                # Time distribution
                st.markdown("**Predicted Finish Time Range:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Best 10%", format_time(mc['p10_time']))
                    st.caption("Optimistic scenario")
                
                with col2:
                    st.metric("Most Likely", format_time(mc['p50_time']))
                    st.caption("Median outcome")
                
                with col3:
                    st.metric("Worst 10%", format_time(mc['p90_time']))
                    st.caption("Conservative scenario")
                
                # Distribution chart
                fig_mc = optimizer.plot_monte_carlo_results(mc)
                st.plotly_chart(fig_mc, width="stretch")
                
                # Recommendations
                st.markdown("---")
                st.markdown("**💡 Race Day Recommendations:**")
                
                if mc['bonk_risk_pct'] > 20:
                    st.warning(f"⚠️ Elevated bonk risk ({mc['bonk_risk_pct']:.0f}%). Add a gel or slow early pace slightly.")
                
                if mc['gi_distress_pct'] > 25:
                    st.warning(f"⚠️ High GI risk ({mc['gi_distress_pct']:.0f}%). Move breakfast earlier (3.5h before) or use simpler foods.")
                
                if mc['success_rate_pct'] > 75:
                    st.success(f"✅ Strong probability of hitting your goal ({mc['success_rate_pct']:.0f}%). Stick to the plan!")
                else:
                    st.info(f"💡 Moderate success rate ({mc['success_rate_pct']:.0f}%). Consider adjusting target time or nutrition strategy.")
    
    # ====================================================================
    # MODE B: PERFORMANCE OVER TIME
    # ====================================================================
    
    elif race_mode == "📊 Performance at Different Weeks":
        st.markdown("---")
        st.subheader("How Your Performance Changes Week-by-Week")
        
        if not has_plan or 'simulation_results' not in st.session_state:
            st.warning("⚠️ Create a training plan first to see performance progression")
        else:
            results = st.session_state.simulation_results
            config = st.session_state.simulation_config
            weekly_volumes = config['weekly_volumes']
            
            # Race configuration
            col1, col2 = st.columns(2)
            
            with col1:
                test_distance = st.selectbox(
                    "Test Distance",
                    [5.0, 10.0, 21.1, 42.2],
                    format_func=lambda x: f"{x} km" + (" (5K)" if x==5 else " (10K)" if x==10 else " (Half)" if x==21.1 else " (Marathon)"),
                    key="race_perf_test_distance"  # ← ADD THIS
                )

            
            with col2:
                test_terrain = st.selectbox(
                    "Terrain", 
                    ["Road", "Trail - Easy", "Trail - Technical"],
                    key="race_perf_terrain"  # ← ADD THIS
                )

            
            # Calculate performance for each week
            performance_data = []
            
            for week_num in range(1, len(weekly_volumes) + 1):
                week_state = results['weekly_states'][week_num - 1]
                
                # Volume-based fitness
                volume_avg = sum(weekly_volumes[:week_num]) / week_num
                base_pace = 5.5 - (volume_avg / 100) * 1.5
                
                # Fitness adjustment
                fitness_pct = week_state.get('recovery_score', 0.7) * 100
                fitness_mult = (fitness_pct / 100)
                
                # Terrain
                terrain_mult = 1.0 if test_terrain == "Road" else 1.08 if "Easy" in test_terrain else 1.15
                
                predicted_pace = (base_pace / fitness_mult) * terrain_mult
                predicted_time = predicted_pace * test_distance
                
                performance_data.append({
                    'week': week_num,
                    'time_min': predicted_time,
                    'pace_min_km': predicted_pace,
                    'volume': weekly_volumes[week_num - 1],
                    'fitness': fitness_pct
                })
            
            # Create performance progression chart
            fig_perf = go.Figure()
            
            weeks = [p['week'] for p in performance_data]
            times = [p['time_min'] for p in performance_data]
            
            fig_perf.add_trace(go.Scatter(
                x=weeks,
                y=times,
                mode='lines+markers',
                name='Predicted Race Time',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                hovertemplate='Week %{x}<br>Time: ' + '<br>'.join([format_time(t) for t in times]) + '<extra></extra>'
            ))
            
            fig_perf.update_layout(
                title=f'Predicted {test_distance} km Performance Over Time',
                xaxis=dict(title='Week', dtick=1),
                yaxis=dict(title='Finish Time (minutes)'),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_perf, width="stretch")
            
            # Best race windows
            st.markdown("---")
            st.markdown("**🏆 Optimal Race Windows:**")
            
            # Find peaks (local minima in time)
            best_weeks = []
            for i in range(1, len(times) - 1):
                if times[i] < times[i-1] and times[i] < times[i+1]:
                    best_weeks.append((i+1, times[i]))
            
            # Also add overall best
            best_overall_idx = times.index(min(times))
            best_weeks.append((best_overall_idx + 1, times[best_overall_idx]))
            best_weeks = sorted(set(best_weeks), key=lambda x: x[1])[:3]
            
            if best_weeks:
                for rank, (week, time) in enumerate(best_weeks, 1):
                    col1, col2, col3 = st.columns([1, 2, 2])
                    
                    with col1:
                        st.metric(f"Peak #{rank}", f"Week {week}")
                    
                    with col2:
                        st.metric("Predicted Time", format_time(time))
                    
                    with col3:
                        pace = time / test_distance
                        st.metric("Pace", f"{pace:.2f} min/km")
            
            # Performance comparison table
            st.markdown("---")
            st.markdown("**📋 Week-by-Week Performance Table:**")
            
            perf_df = pd.DataFrame(performance_data)
            perf_df['Time'] = perf_df['time_min'].apply(format_time)
            perf_df['Pace (min/km)'] = perf_df['pace_min_km'].round(2)
            perf_df['Volume (km)'] = perf_df['volume']
            perf_df['Fitness (%)'] = perf_df['fitness'].round(0).astype(int)
            
            display_df = perf_df[['week', 'Time', 'Pace (min/km)', 'Volume (km)', 'Fitness (%)']].rename(columns={'week': 'Week'})
            
            st.dataframe(display_df, width="stretch", hide_index=True, height=400)
    
    # ====================================================================
    # MODE C: MULTIPLE RACE SCENARIOS
    # ====================================================================
    
    elif race_mode == "🔮 Multiple Race Scenarios":
        st.markdown("---")
        st.subheader("Compare Different Race Options")
        
        if not has_plan or 'simulation_results' not in st.session_state:
            st.warning("⚠️ Create a training plan first")
        else:
            results = st.session_state.simulation_results
            config = st.session_state.simulation_config
            weekly_volumes = config['weekly_volumes']
            
            st.markdown("**Add races you're considering:**")
            
            # Initialize race scenarios
            if 'race_scenarios' not in st.session_state:
                st.session_state.race_scenarios = []
            
            # Add new race
            with st.expander("➕ Add New Race Scenario", expanded=len(st.session_state.race_scenarios) == 0):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    scenario_name = st.text_input("Race Name", value=f"Race {len(st.session_state.race_scenarios) + 1}")
                
                with col2:
                    scenario_distance = st.number_input("Distance (km)", value=10.0, min_value=1.0, max_value=100.0, step=0.5, key="scenario_dist")
                
                with col3:
                    scenario_week = st.number_input("Week", value=min(12, len(weekly_volumes)), min_value=1, max_value=len(weekly_volumes), step=1, key="scenario_week")
                
                with col4:
                    scenario_terrain = st.selectbox("Terrain", ["Road", "Trail - Easy", "Trail - Technical"], key="scenario_terrain")
                
                if st.button("Add Scenario"):
                    st.session_state.race_scenarios.append({
                        'name': scenario_name,
                        'distance': scenario_distance,
                        'week': scenario_week,
                        'terrain': scenario_terrain
                    })
                    st.rerun()
            
            # Display and compare scenarios
            if st.session_state.race_scenarios:
                st.markdown("---")
                st.markdown("**📊 Scenario Comparison:**")
                
                comparison_data = []
                
                for scenario in st.session_state.race_scenarios:
                    week_idx = scenario['week'] - 1
                    week_state = results['weekly_states'][week_idx]
                    
                    # Calculate performance
                    volume_avg = sum(weekly_volumes[:scenario['week']]) / scenario['week']
                    base_pace = 5.5 - (volume_avg / 100) * 1.5
                    
                    fitness_pct = week_state.get('recovery_score', 0.7) * 100
                    fitness_mult = (fitness_pct / 100)
                    
                    terrain_mult = 1.0 if scenario['terrain'] == "Road" else 1.08 if "Easy" in scenario['terrain'] else 1.15
                    
                    predicted_pace = (base_pace / fitness_mult) * terrain_mult
                    predicted_time = predicted_pace * scenario['distance']
                    
                    # Readiness
                    acwr = week_state.get('acwr', 1.0)
                    
                    if fitness_pct > 80 and 0.7 <= acwr <= 1.0:
                        readiness = "🟢 Excellent"
                    elif fitness_pct > 65 and acwr <= 1.2:
                        readiness = "🟡 Good"
                    else:
                        readiness = "🟠 Fair"
                    
                    comparison_data.append({
                        'Race': scenario['name'],
                        'Distance': f"{scenario['distance']} km",
                        'Week': scenario['week'],
                        'Terrain': scenario['terrain'],
                        'Predicted Time': format_time(predicted_time),
                        'Pace (min/km)': f"{predicted_pace:.2f}",
                        'Fitness': f"{fitness_pct:.0f}%",
                        'Readiness': readiness
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width="stretch", hide_index=True)
                
                # Clear scenarios
                if st.button("🗑️ Clear All Scenarios"):
                    st.session_state.race_scenarios = []
                    st.rerun()
                
                # Visual comparison
                st.markdown("---")
                st.markdown("**📈 Visual Comparison:**")
                
                fig_compare = go.Figure()
                
                for i, row in comparison_df.iterrows():
                    time_val = parse_time(row['Predicted Time'])
                    fig_compare.add_trace(go.Bar(
                        name=row['Race'],
                        x=[row['Race']],
                        y=[time_val],
                        text=[row['Predicted Time']],
                        textposition='auto',
                        hovertemplate=f"<b>{row['Race']}</b><br>Time: {row['Predicted Time']}<br>Week: {row['Week']}<extra></extra>"
                    ))
                
                fig_compare.update_layout(
                    title='Predicted Race Times Comparison',
                    xaxis_title='Race',
                    yaxis_title='Time (minutes)',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_compare, width="stretch")
            
            else:
                st.info("👆 Add race scenarios above to compare them")

    

    # ========================================================================
    # TAB 5: SETTINGS
    # ========================================================================
    
    with tab6:
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
            
            if st.button("🔄 Refresh Garmin Data", width="stretch"):
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
