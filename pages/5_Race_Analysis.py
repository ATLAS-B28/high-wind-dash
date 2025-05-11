import streamlit as st
import pandas as pd
import plotly.express as px
import os
from utils import load_data, create_filtered_dataframe, get_column_types, SessionStateManager

# Initialize session state
SessionStateManager.initialize_session_state()

st.set_page_config(
    page_title="F1 Race Analysis",
    page_icon="🏁",
    layout="wide"
)

st.title("Formula 1 Race Analysis")
st.write("Analyze race results and performance data")

# Function to load race data from formulaarchive
def load_race_data():
    base_path = os.path.join(os.getcwd(), "formulaarchive")
    
    # Dictionary to store all dataframes
    dfs = {}
    
    # Load necessary files
    try:
        # Load results
        results_df = pd.read_csv(os.path.join(base_path, "results.csv"))
        dfs["results"] = results_df
        
        # Load drivers
        drivers_df = pd.read_csv(os.path.join(base_path, "drivers.csv"))
        dfs["drivers"] = drivers_df
        
        # Load constructors
        constructors_df = pd.read_csv(os.path.join(base_path, "constructors.csv"))
        dfs["constructors"] = constructors_df
        
        # Load races
        races_df = pd.read_csv(os.path.join(base_path, "races.csv"))
        dfs["races"] = races_df
        
        # Create merged dataframe for race analysis
        race_results = results_df.merge(
            drivers_df, on='driverId'
        ).merge(
            constructors_df, on='constructorId'
        ).merge(
            races_df, on='raceId'
        )
        
        # Create driver name and set team name
        race_results['DriverName'] = race_results['forename'] + ' ' + race_results['surname']
        race_results['Team'] = race_results['name_x']  # Constructor name
        race_results['Race'] = race_results['name_y']  # Race name
        
        dfs["race_results"] = race_results
        
        # Try to load lap times if available
        try:
            lap_times_df = pd.read_csv(os.path.join(base_path, "lap_times.csv"))
            dfs["lap_times"] = lap_times_df
            
            # Merge lap times with race results
            lap_times_with_info = lap_times_df.merge(
                race_results[['raceId', 'driverId', 'DriverName', 'Team', 'Race']], 
                on=['raceId', 'driverId']
            )
            dfs["lap_times_with_info"] = lap_times_with_info
        except:
            pass
        
        return dfs
    except Exception as e:
        st.error(f"Error loading race data: {e}")
        return {}

# Data source selection
data_source = st.radio(
    "Choose data source",
    ["Formula Archive data", "Use existing dataset", "Upload new file"],
    horizontal=True,
    index=0  # Default to Formula Archive data
)

df = None

if data_source == "Formula Archive data":
    try:
        # Load Formula Archive data
        f1_dfs = load_race_data()
        
        if f1_dfs and "race_results" in f1_dfs:
            df = f1_dfs["race_results"]
            st.success("Loaded Formula Archive race results data")
            
            # Option to save to session state
            save_to_session = st.checkbox("Save this dataset for use across all pages", value=False)
            
            if save_to_session:
                if st.button("Save Formula Archive Race Data"):
                    # Save individual datasets
                    for key, dataset in f1_dfs.items():
                        SessionStateManager.save_dataset(
                            f"F1_{key}", 
                            dataset, 
                            {'source': f'formulaarchive/{key}'}
                        )
                    st.success("Formula 1 race datasets saved successfully!")
                    st.rerun()
    except Exception as e:
        st.error(f"Error loading Formula Archive data: {e}")
        st.info("Falling back to other data sources.")
        data_source = "Use existing dataset" if st.session_state.datasets else "Upload new file"

if data_source == "Use existing dataset" and df is None:
    # Dataset selector
    selected_dataset = SessionStateManager.create_dataset_selector("Select a dataset with race data")
    
    if selected_dataset:
        df = SessionStateManager.get_dataset(selected_dataset)
        st.success(f"Loaded dataset: {selected_dataset}")

elif data_source == "Upload new file" and df is None:
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file with F1 race data", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Option to save to session state
            save_to_session = st.checkbox("Save this dataset for use across all pages", value=True)
            
            if save_to_session:
                dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])
                
                if st.button("Save Dataset"):
                    SessionStateManager.save_dataset(
                        dataset_name, 
                        df, 
                        {'source': uploaded_file.name, 'type': 'race_data'}
                    )
                    st.success(f"Dataset '{dataset_name}' saved successfully!")
                    st.rerun()

if df is not None:
        # Create filtered dataframe
        filtered_df = create_filtered_dataframe(df)
        
        # Get column types
        column_types = get_column_types(filtered_df)
        
        # Race selection
        race_col = None
        for col in ['Race', 'RaceName', 'Grand Prix', 'GrandPrix']:
            if col in filtered_df.columns:
                race_col = col
                break
        
        if race_col:
            races = sorted(filtered_df[race_col].unique())
            selected_race = st.selectbox("Select Race", races)
            
            # Filter for selected race
            race_df = filtered_df[filtered_df[race_col] == selected_race]
            
            # Display race results
            st.subheader(f"{selected_race} Results")
            st.dataframe(race_df, use_container_width=True)
            
            # Race analysis options
            st.subheader("Race Analysis")
            
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Position Chart", "Team Comparison", "Lap Time Analysis"]
            )
            
            if analysis_type == "Position Chart":
                # Find position column
                position_col = None
                for col in ['Position', 'FinalPosition']:
                    if col in race_df.columns:
                        position_col = col
                        break
                
                if position_col:
                    # Find driver column
                    driver_col = None
                    for col in ['DriverName', 'Driver', 'Name']:
                        if col in race_df.columns:
                            driver_col = col
                            break
                    
                    if driver_col:
                        # Create position chart
                        fig = px.bar(
                            race_df.sort_values(position_col), 
                            x=position_col, 
                            y=driver_col,
                            color='Team' if 'Team' in race_df.columns else None,
                            title=f"{selected_race} Final Positions",
                            labels={position_col: "Position", driver_col: "Driver"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not find driver column in the data")
                else:
                    st.warning("Could not find position column in the data")
            
            elif analysis_type == "Team Comparison":
                if 'Team' in race_df.columns:
                    # Group by team
                    team_cols = ['Team']
                    
                    # Find position column
                    position_col = None
                    for col in ['Position', 'FinalPosition']:
                        if col in race_df.columns:
                            position_col = col
                            break
                    
                    if position_col:
                        team_cols.append(position_col)
                    
                    # Find points column
                    if 'Points' in race_df.columns:
                        team_cols.append('Points')
                    
                    # Create team comparison
                    team_df = race_df[team_cols].groupby('Team').agg({
                        col: 'mean' if col == position_col else 'sum' 
                        for col in team_cols if col != 'Team'
                    }).reset_index()
                    
                    # Display team comparison
                    st.dataframe(team_df.sort_values('Points' if 'Points' in team_df.columns else position_col, 
                                                   ascending='Points' not in team_df.columns),
                               use_container_width=True)
                    
                    # Create team points chart
                    if 'Points' in team_df.columns:
                        fig = px.bar(
                            team_df.sort_values('Points', ascending=False),
                            x='Team',
                            y='Points',
                            title=f"Team Points in {selected_race}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No Team column found in the data")
            
            elif analysis_type == "Lap Time Analysis":
                # Check for lap time data
                lap_time_cols = [col for col in race_df.columns if 'Lap' in col and 'Time' in col]
                
                if lap_time_cols:
                    # Find driver column
                    driver_col = None
                    for col in ['DriverName', 'Driver', 'Name']:
                        if col in race_df.columns:
                            driver_col = col
                            break
                    
                    if driver_col:
                        # Select drivers to compare
                        selected_drivers = st.multiselect(
                            "Select Drivers to Compare",
                            sorted(race_df[driver_col].unique()),
                            default=sorted(race_df[driver_col].unique())[:3]
                        )
                        
                        if selected_drivers:
                            # Filter for selected drivers
                            drivers_df = race_df[race_df[driver_col].isin(selected_drivers)]
                            
                            # Reshape data for lap time comparison
                            lap_times = []
                            
                            for _, row in drivers_df.iterrows():
                                driver = row[driver_col]
                                for lap_col in lap_time_cols:
                                    if pd.notna(row[lap_col]):
                                        lap_num = int(lap_col.replace('Lap', '').replace('Time', ''))
                                        lap_times.append({
                                            'Driver': driver,
                                            'Lap': lap_num,
                                            'Time': row[lap_col]
                                        })
                            
                            if lap_times:
                                lap_df = pd.DataFrame(lap_times)
                                
                                # Convert lap times to seconds if they're strings
                                if lap_df['Time'].dtype == 'object':
                                    def convert_to_seconds(time_str):
                                        try:
                                            if ':' in time_str:
                                                parts = time_str.split(':')
                                                return float(parts[0]) * 60 + float(parts[1])
                                            else:
                                                return float(time_str)
                                        except:
                                            return None
                                    
                                    lap_df['TimeSeconds'] = lap_df['Time'].apply(convert_to_seconds)
                                else:
                                    lap_df['TimeSeconds'] = lap_df['Time']
                                
                                # Create lap time chart
                                fig = px.line(
                                    lap_df,
                                    x='Lap',
                                    y='TimeSeconds',
                                    color='Driver',
                                    markers=True,
                                    title=f"Lap Time Comparison - {selected_race}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Save lap time analysis to session state
                                if st.button("Save this lap time analysis"):
                                    analysis_name = st.text_input("Analysis Name", value=f"Lap Time Analysis - {selected_race}")
                                    
                                    if analysis_name:
                                        analysis_data = {
                                            "type": "race_lap_time_analysis",
                                            "race": selected_race,
                                            "drivers": selected_drivers,
                                            "lap_times": lap_df.to_dict('records')
                                        }
                                        
                                        SessionStateManager.save_analysis_result(
                                            analysis_name,
                                            analysis_data,
                                            {"type": "race_lap_time_analysis"}
                                        )
                                        st.success(f"Analysis '{analysis_name}' saved successfully!")
                            else:
                                st.warning("No valid lap time data found")
                        else:
                            st.info("Please select at least one driver")
                    else:
                        st.warning("Could not find driver column in the data")
                else:
                    st.warning("No lap time data found in the dataset")
        else:
            st.warning("Could not identify race column in the data")
            
        # Save race analysis to session state
        if st.button("Save this race analysis"):
            analysis_name = st.text_input("Analysis Name", value=f"Race Analysis - {selected_race}")
            
            if analysis_name:
                analysis_data = {
                    "type": "race_analysis",
                    "race": selected_race,
                    "results": race_df.to_dict('records')
                }
                
                SessionStateManager.save_analysis_result(
                    analysis_name,
                    analysis_data,
                    {"type": "race_analysis"}
                )
                st.success(f"Analysis '{analysis_name}' saved successfully!")
else:
    # Check if there are any saved race analyses
    saved_analyses = [name for name in SessionStateManager.get_analysis_names() 
                     if 'type' in SessionStateManager.analysis_results[name]['metadata'] 
                     and SessionStateManager.analysis_results[name]['metadata']['type'] in 
                     ['race_analysis', 'race_lap_time_analysis']]
    
    if saved_analyses:
        st.subheader("Previously Saved Race Analyses")
        selected_analysis = st.selectbox("Select a previous analysis", saved_analyses)
        
        if st.button("Load Analysis"):
            analysis_data = SessionStateManager.get_analysis_result(selected_analysis)
            analysis_type = SessionStateManager.analysis_results[selected_analysis]['metadata']['type']
            
            st.success(f"Loaded analysis: {selected_analysis}")
            
            if analysis_type == 'race_analysis':
                st.subheader(f"Race Analysis - {analysis_data['race']}")
                results_df = pd.DataFrame(analysis_data['results'])
                st.dataframe(results_df, use_container_width=True)
                
                # Try to create position chart if possible
                if 'Position' in results_df.columns and 'DriverName' in results_df.columns:
                    fig = px.bar(
                        results_df.sort_values('Position'), 
                        x='Position', 
                        y='DriverName',
                        color='Team' if 'Team' in results_df.columns else None,
                        title=f"{analysis_data['race']} Final Positions"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_type == 'race_lap_time_analysis':
                st.subheader(f"Lap Time Analysis - {analysis_data['race']}")
                st.write(f"Drivers: {', '.join(analysis_data['drivers'])}")
                
                lap_df = pd.DataFrame(analysis_data['lap_times'])
                
                # Create lap time chart
                fig = px.line(
                    lap_df,
                    x='Lap',
                    y='TimeSeconds',
                    color='Driver',
                    markers=True,
                    title=f"Lap Time Comparison - {analysis_data['race']}"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select a data source to analyze race data.")
        
        # Example data structure
        st.subheader("Example Data Format")
        example_df = pd.DataFrame({
            'Race': ['British GP', 'British GP', 'British GP'],
            'DriverName': ['Lewis Hamilton', 'Max Verstappen', 'Charles Leclerc'],
            'Team': ['Mercedes', 'Red Bull', 'Ferrari'],
            'Position': [1, 2, 3],
            'Points': [25, 18, 15],
            'Lap1Time': ['1:30.5', '1:30.8', '1:31.2'],
            'Lap2Time': ['1:29.8', '1:30.1', '1:30.5']
        })
        st.dataframe(example_df)