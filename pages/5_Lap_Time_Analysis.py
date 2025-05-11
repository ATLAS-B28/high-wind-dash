import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from utils import load_data, SessionStateManager

# Initialize session state
SessionStateManager.initialize_session_state()

st.set_page_config(
    page_title="F1 Lap Time Analysis",
    page_icon="⏱️",
    layout="wide"
)

st.title("Formula 1 Lap Time Analysis")
st.write("Analyze and compare lap times from Formula 1 races")

# Function to load lap times data
def load_lap_times_data():
    base_path = os.path.join(os.getcwd(), "formulaarchive")
    
    # Dictionary to store all dataframes
    dfs = {}
    
    # Load lap times data
    lap_times_df = pd.read_csv(os.path.join(base_path, "lap_times.csv"))
    dfs["lap_times"] = lap_times_df
    
    # Load drivers data
    drivers_df = pd.read_csv(os.path.join(base_path, "drivers.csv"))
    dfs["drivers"] = drivers_df
    
    # Load races data
    races_df = pd.read_csv(os.path.join(base_path, "races.csv"))
    dfs["races"] = races_df
    
    # Merge data to get comprehensive lap time information
    merged_df = lap_times_df.merge(drivers_df, on='driverId')
    merged_df = merged_df.merge(races_df, on='raceId')
    
    # Create full driver name
    merged_df['driverName'] = merged_df['forename'] + ' ' + merged_df['surname']
    dfs["lap_times_complete"] = merged_df
    
    return dfs

# Data source selection
data_source = st.radio(
    "Choose data source",
    ["Formula Archive data", "Use existing dataset", "Upload new file"],
    horizontal=True,
    index=0  # Default to Formula Archive data
)

lap_times_df = None
drivers_df = None
races_df = None
merged_df = None
data_loaded = False

if data_source == "Formula Archive data":
    try:
        # Load Formula Archive data
        f1_dfs = load_lap_times_data()
        
        if f1_dfs:
            lap_times_df = f1_dfs["lap_times"]
            drivers_df = f1_dfs["drivers"]
            races_df = f1_dfs["races"]
            merged_df = f1_dfs["lap_times_complete"]
            data_loaded = True
            
            st.success("Loaded Formula Archive lap times data")
            
            # Option to save to session state
            save_to_session = st.checkbox("Save this dataset for use across all pages", value=False)
            
            if save_to_session:
                if st.button("Save Formula Archive Lap Times Data"):
                    # Save individual datasets
                    SessionStateManager.save_dataset("F1_Lap_Times", lap_times_df, {'source': 'formulaarchive/lap_times.csv'})
                    SessionStateManager.save_dataset("F1_Drivers", drivers_df, {'source': 'formulaarchive/drivers.csv'})
                    SessionStateManager.save_dataset("F1_Races", races_df, {'source': 'formulaarchive/races.csv'})
                    SessionStateManager.save_dataset("F1_Lap_Times_Complete", merged_df, {'source': 'formulaarchive merged lap times data'})
                    st.success("Formula 1 lap times datasets saved successfully!")
                    st.rerun()
    except Exception as e:
        st.error(f"Error loading Formula Archive data: {e}")
        st.info("Falling back to other data sources.")
        data_source = "Use existing dataset" if st.session_state.datasets else "Upload new file"

if data_source == "Use existing dataset" and not data_loaded:
    # Dataset selector
    selected_dataset = SessionStateManager.create_dataset_selector("Select a dataset with lap times")
    
    if selected_dataset:
        merged_df = SessionStateManager.get_dataset(selected_dataset)
        
        # Check if this dataset has the required columns for lap time analysis
        required_cols = ['lap', 'milliseconds', 'driverId', 'raceId']
        if all(col in merged_df.columns for col in required_cols):
            data_loaded = True
            st.success(f"Loaded dataset: {selected_dataset}")
            
            # Try to extract driver names if available
            if 'forename' in merged_df.columns and 'surname' in merged_df.columns and 'driverName' not in merged_df.columns:
                merged_df['driverName'] = merged_df['forename'] + ' ' + merged_df['surname']
            
            # If we don't have races_df but have race info in merged_df
            if 'name' in merged_df.columns and 'year' in merged_df.columns:
                races_df = merged_df[['raceId', 'name', 'year', 'date']].drop_duplicates()
        else:
            st.warning(f"Selected dataset doesn't have the required columns for lap time analysis. Please select another dataset.")

elif data_source == "Upload new file" and not data_loaded:
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file with F1 lap times data", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load data
        merged_df = load_data(uploaded_file)
        
        if merged_df is not None:
            # Check if this dataset has the required columns for lap time analysis
            required_cols = ['lap', 'milliseconds', 'driverId', 'raceId']
            if all(col in merged_df.columns for col in required_cols):
                data_loaded = True
                
                # Try to extract driver names if available
                if 'forename' in merged_df.columns and 'surname' in merged_df.columns and 'driverName' not in merged_df.columns:
                    merged_df['driverName'] = merged_df['forename'] + ' ' + merged_df['surname']
                
                # If we don't have races_df but have race info in merged_df
                if 'name' in merged_df.columns and 'year' in merged_df.columns:
                    races_df = merged_df[['raceId', 'name', 'year', 'date']].drop_duplicates()
                
                # Option to save to session state
                save_to_session = st.checkbox("Save this dataset for use across all pages", value=True)
                
                if save_to_session:
                    dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])
                    
                    if st.button("Save Dataset"):
                        SessionStateManager.save_dataset(
                            dataset_name, 
                            merged_df, 
                            {'source': uploaded_file.name, 'type': 'lap_times'}
                        )
                        st.success(f"Dataset '{dataset_name}' saved successfully!")
                        st.rerun()
            else:
                st.warning("Uploaded file doesn't have the required columns for lap time analysis.")
                data_loaded = False

if data_loaded:
    # Race selection
    st.subheader("Select Race")
    
    # Get unique races
    races = races_df[['raceId', 'name', 'year', 'date']].sort_values(['year', 'date'], ascending=[False, True])
    race_options = [f"{row['year']} {row['name']}" for _, row in races.iterrows()]
    race_ids = races['raceId'].tolist()
    
    selected_race_index = st.selectbox("Select Race", range(len(race_options)), format_func=lambda x: race_options[x])
    selected_race_id = race_ids[selected_race_index]
    
    # Filter data for selected race
    race_data = merged_df[merged_df['raceId'] == selected_race_id].copy()
    
    if not race_data.empty:
        # Get race info
        race_info = races[races['raceId'] == selected_race_id].iloc[0]
        st.write(f"**Race:** {race_info['name']} {race_info['year']} ({race_info['date']})")
        
        # Driver selection
        st.subheader("Select Drivers to Compare")
        
        # Get unique drivers in this race
        race_drivers = race_data[['driverId', 'driverName']].drop_duplicates().sort_values('driverName')
        
        # Multi-select for drivers
        selected_drivers = st.multiselect(
            "Select drivers to compare",
            race_drivers['driverId'].tolist(),
            default=race_drivers['driverId'].tolist()[:3],
            format_func=lambda x: race_drivers[race_drivers['driverId'] == x]['driverName'].iloc[0]
        )
        
        if selected_drivers:
            # Filter data for selected drivers
            driver_data = race_data[race_data['driverId'].isin(selected_drivers)].copy()
            
            # Create lap time chart
            st.subheader("Lap Time Comparison")
            
            # Convert milliseconds to seconds for better readability
            driver_data.loc[:, 'seconds'] = driver_data['milliseconds'] / 1000
            
            # Create chart
            fig = px.line(
                driver_data,
                x='lap',
                y='seconds',
                color='driverName',
                markers=True,
                title=f"Lap Time Comparison - {race_info['name']} {race_info['year']}",
                labels={'lap': 'Lap Number', 'seconds': 'Lap Time (seconds)'}
            )
            
            # Add hover data - using values() to avoid SettingWithCopyWarning
            # Check if 'times' column exists, otherwise try 'time'
            time_column = 'times' if 'times' in driver_data.columns else 'time'
            
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>Lap: %{x}<br>Time: %{customdata[1]}<br>Position: %{customdata[2]}<extra></extra>",
                customdata=np.column_stack((
                    driver_data['driverName'].values, 
                    driver_data[time_column].values if time_column in driver_data.columns else ["N/A"] * len(driver_data),
                    driver_data['position'].values
                ))
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Lap time statistics
            st.subheader("Lap Time Statistics")
            
            # Calculate statistics by driver
            stats = driver_data.groupby('driverName').agg(
                fastest_lap=('seconds', 'min'),
                slowest_lap=('seconds', 'max'),
                avg_lap=('seconds', 'mean'),
                median_lap=('seconds', 'median'),
                total_laps=('lap', 'count')
            ).reset_index()
            
            # Format times
            stats['fastest_lap'] = stats['fastest_lap'].round(3)
            stats['slowest_lap'] = stats['slowest_lap'].round(3)
            stats['avg_lap'] = stats['avg_lap'].round(3)
            stats['median_lap'] = stats['median_lap'].round(3)
            
            # Display statistics
            st.dataframe(stats, use_container_width=True)
            
            # Position by lap chart
            st.subheader("Position by Lap")
            
            # Create position chart
            fig_pos = px.line(
                driver_data,
                x='lap',
                y='position',
                color='driverName',
                markers=True,
                title=f"Position by Lap - {race_info['name']} {race_info['year']}",
                labels={'lap': 'Lap Number', 'position': 'Position'}
            )
            
            # Invert y-axis for positions (1 is best)
            fig_pos.update_yaxes(autorange="reversed")
            
            st.plotly_chart(fig_pos, use_container_width=True)
            
            # Detailed lap times table
            with st.expander("View Detailed Lap Times"):
                # Pivot table to show lap times by driver
                # Check if 'times' column exists, otherwise try 'time'
                time_column = 'times' if 'times' in driver_data.columns else 'time'
                if time_column in driver_data.columns:
                    pivot_df = driver_data.pivot(index='lap', columns='driverName', values=time_column)
                    st.dataframe(pivot_df, use_container_width=True)
                else:
                    st.warning("Time data not available for detailed view")
            
            # Save analysis to session state
            if st.button("Save this lap time analysis"):
                analysis_name = st.text_input("Analysis Name", value=f"Lap Time Analysis - {race_info['name']} {race_info['year']}")
                
                if analysis_name:
                    # Create a simplified version of the analysis data
                    analysis_data = {
                        "type": "lap_time_analysis",
                        "race_id": selected_race_id,
                        "race_name": f"{race_info['name']} {race_info['year']}",
                        "driver_ids": selected_drivers,
                        "driver_names": [race_drivers[race_drivers['driverId'] == x]['driverName'].iloc[0] for x in selected_drivers],
                        "stats": stats.to_dict('records')
                    }
                    
                    SessionStateManager.save_analysis_result(
                        analysis_name,
                        analysis_data,
                        {"type": "lap_time_analysis"}
                    )
                    st.success(f"Analysis '{analysis_name}' saved successfully!")
        else:
            st.info("Please select at least one driver to compare lap times.")
    else:
        st.warning("No lap time data found for the selected race.")
else:
    # Check if there are any saved lap time analyses
    saved_analyses = [name for name in SessionStateManager.get_analysis_names() 
                     if 'type' in SessionStateManager.analysis_results[name]['metadata'] 
                     and SessionStateManager.analysis_results[name]['metadata']['type'] == 'lap_time_analysis']
    
    if saved_analyses:
        st.subheader("Previously Saved Lap Time Analyses")
        selected_analysis = st.selectbox("Select a previous analysis", saved_analyses)
        
        if st.button("Load Analysis"):
            analysis_data = SessionStateManager.get_analysis_result(selected_analysis)
            
            st.success(f"Loaded analysis: {selected_analysis}")
            st.write(f"**Race:** {analysis_data['race_name']}")
            st.write(f"**Drivers:** {', '.join(analysis_data['driver_names'])}")
            
            # Display stats
            st.subheader("Lap Time Statistics")
            stats_df = pd.DataFrame(analysis_data['stats'])
            st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("Please select a data source to analyze lap times.")
        
        # Example data structure
        st.subheader("Example Data Format")
        example_df = pd.DataFrame({
            'raceId': [841, 841, 841, 841],
            'driverId': [20, 20, 1, 1],
            'lap': [1, 2, 1, 2],
            'position': [1, 1, 2, 2],
            'times': ["1:38.109", "1:33.006", "1:40.573", "1:33.774"],
            'milliseconds': [98109, 93006, 100573, 93774]
        })
        st.dataframe(example_df)