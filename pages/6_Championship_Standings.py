import streamlit as st
import pandas as pd
import plotly.express as px
import os
from utils import load_data, SessionStateManager

# Initialize session state
SessionStateManager.initialize_session_state()

st.set_page_config(
    page_title="F1 Championship Standings",
    page_icon="🏆",
    layout="wide"
)

st.title("Formula 1 Championship Standings")
st.write("View and analyze driver and constructor championship standings")

# Function to load championship data from formulaarchive
def load_championship_data():
    base_path = os.path.join(os.getcwd(), "formulaarchive")
    
    # Dictionary to store all dataframes
    dfs = {}
    
    # Load necessary files
    try:
        # Load driver standings
        driver_standings_df = pd.read_csv(os.path.join(base_path, "driver_standings.csv"))
        dfs["driver_standings"] = driver_standings_df
        
        # Load constructor standings
        constructor_standings_df = pd.read_csv(os.path.join(base_path, "constructor_standings.csv"))
        dfs["constructor_standings"] = constructor_standings_df
        
        # Load drivers
        drivers_df = pd.read_csv(os.path.join(base_path, "drivers.csv"))
        dfs["drivers"] = drivers_df
        
        # Load constructors
        constructors_df = pd.read_csv(os.path.join(base_path, "constructors.csv"))
        dfs["constructors"] = constructors_df
        
        # Load races
        races_df = pd.read_csv(os.path.join(base_path, "races.csv"))
        dfs["races"] = races_df
        
        # Create merged dataframes
        # Driver standings with driver names and race info
        driver_standings_complete = driver_standings_df.merge(
            drivers_df, on='driverId'
        ).merge(
            races_df, on='raceId'
        )
        driver_standings_complete['DriverName'] = driver_standings_complete['forename'] + ' ' + driver_standings_complete['surname']
        dfs["driver_standings_complete"] = driver_standings_complete
        
        # Constructor standings with constructor names and race info
        constructor_standings_complete = constructor_standings_df.merge(
            constructors_df, on='constructorId'
        ).merge(
            races_df, on='raceId'
        )
        constructor_standings_complete['Team'] = constructor_standings_complete['name_x']  # Rename for compatibility
        dfs["constructor_standings_complete"] = constructor_standings_complete
        
        return dfs
    except Exception as e:
        st.error(f"Error loading championship data: {e}")
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
        f1_dfs = load_championship_data()
        
        if f1_dfs:
            # Let user select which dataset to analyze
            championship_type = st.radio(
                "Select Championship Type",
                ["Drivers' Championship", "Constructors' Championship"]
            )
            
            if championship_type == "Drivers' Championship":
                df = f1_dfs["driver_standings_complete"]
                st.success("Loaded Formula Archive driver standings data")
            else:
                df = f1_dfs["constructor_standings_complete"]
                st.success("Loaded Formula Archive constructor standings data")
            
            # Option to save to session state
            save_to_session = st.checkbox("Save this dataset for use across all pages", value=False)
            
            if save_to_session:
                if st.button("Save Formula Archive Championship Data"):
                    # Save individual datasets
                    for key, dataset in f1_dfs.items():
                        SessionStateManager.save_dataset(
                            f"F1_{key}", 
                            dataset, 
                            {'source': f'formulaarchive/{key}'}
                        )
                    st.success("Formula 1 championship datasets saved successfully!")
                    st.rerun()
    except Exception as e:
        st.error(f"Error loading Formula Archive data: {e}")
        st.info("Falling back to other data sources.")
        data_source = "Use existing dataset" if st.session_state.datasets else "Upload new file"

if data_source == "Use existing dataset" and df is None:
    # Dataset selector
    selected_dataset = SessionStateManager.create_dataset_selector("Select a dataset with championship data")
    
    if selected_dataset:
        df = SessionStateManager.get_dataset(selected_dataset)
        st.success(f"Loaded dataset: {selected_dataset}")

elif data_source == "Upload new file" and df is None:
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file with F1 results data", type=["csv", "xlsx", "xls"])
    
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
                        {'source': uploaded_file.name, 'type': 'championship_data'}
                    )
                    st.success(f"Dataset '{dataset_name}' saved successfully!")
                    st.rerun()

if df is not None:
        # Check if we have the necessary columns
        required_cols = ['DriverName', 'Team', 'Points']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            # Championship type selection
            championship_type = st.radio(
                "Select Championship Type",
                ["Drivers' Championship", "Constructors' Championship"]
            )
            
            if championship_type == "Drivers' Championship":
                # Calculate drivers' championship standings
                drivers_standings = df.groupby('DriverName')['Points'].sum().reset_index()
                drivers_standings = drivers_standings.sort_values('Points', ascending=False)
                
                # Add position column
                drivers_standings['Position'] = range(1, len(drivers_standings) + 1)
                
                # Reorder columns
                drivers_standings = drivers_standings[['Position', 'DriverName', 'Points']]
                
                # Display standings
                st.subheader("Drivers' Championship Standings")
                st.dataframe(drivers_standings, use_container_width=True)
                
                # Create bar chart
                fig = px.bar(
                    drivers_standings.head(10),
                    x='DriverName',
                    y='Points',
                    title="Top 10 Drivers by Points",
                    labels={'DriverName': 'Driver', 'Points': 'Championship Points'},
                    color='Points',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Driver comparison
                st.subheader("Driver Comparison")
                
                # Select drivers to compare
                selected_drivers = st.multiselect(
                    "Select Drivers to Compare",
                    sorted(df['DriverName'].unique()),
                    default=sorted(df['DriverName'].unique())[:3]
                )
                
                if selected_drivers:
                    # Filter for selected drivers
                    comparison_df = df[df['DriverName'].isin(selected_drivers)]
                    
                    # Group by driver and race
                    race_col = None
                    for col in ['Race', 'RaceName', 'Grand Prix', 'GrandPrix']:
                        if col in comparison_df.columns:
                            race_col = col
                            break
                    
                    if race_col:
                        # Calculate cumulative points
                        comparison_df = comparison_df.sort_values([race_col])
                        cumulative_points = comparison_df.groupby(['DriverName', race_col])['Points'].sum().reset_index()
                        cumulative_points['CumulativePoints'] = cumulative_points.groupby('DriverName')['Points'].cumsum()
                        
                        # Create cumulative points chart
                        fig = px.line(
                            cumulative_points,
                            x=race_col,
                            y='CumulativePoints',
                            color='DriverName',
                            markers=True,
                            title="Cumulative Points by Race"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not identify race column in the data")
                else:
                    st.info("Please select at least one driver to compare")
            
            elif championship_type == "Constructors' Championship":
                # Calculate constructors' championship standings
                constructors_standings = df.groupby('Team')['Points'].sum().reset_index()
                constructors_standings = constructors_standings.sort_values('Points', ascending=False)
                
                # Add position column
                constructors_standings['Position'] = range(1, len(constructors_standings) + 1)
                
                # Reorder columns
                constructors_standings = constructors_standings[['Position', 'Team', 'Points']]
                
                # Display standings
                st.subheader("Constructors' Championship Standings")
                st.dataframe(constructors_standings, use_container_width=True)
                
                # Create bar chart
                fig = px.bar(
                    constructors_standings,
                    x='Team',
                    y='Points',
                    title="Constructors by Points",
                    labels={'Team': 'Constructor', 'Points': 'Championship Points'},
                    color='Points',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Team comparison
                st.subheader("Team Performance by Race")
                
                # Select teams to compare
                selected_teams = st.multiselect(
                    "Select Teams to Compare",
                    sorted(df['Team'].unique()),
                    default=sorted(df['Team'].unique())[:3]
                )
                
                if selected_teams:
                    # Filter for selected teams
                    team_df = df[df['Team'].isin(selected_teams)]
                    
                    # Group by team and race
                    race_col = None
                    for col in ['Race', 'RaceName', 'Grand Prix', 'GrandPrix']:
                        if col in team_df.columns:
                            race_col = col
                            break
                    
                    if race_col:
                        # Calculate points per race
                        team_points = team_df.groupby(['Team', race_col])['Points'].sum().reset_index()
                        
                        # Create points by race chart
                        fig = px.bar(
                            team_points,
                            x=race_col,
                            y='Points',
                            color='Team',
                            barmode='group',
                            title="Team Points by Race"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate cumulative points
                        team_points = team_points.sort_values([race_col])
                        team_points['CumulativePoints'] = team_points.groupby('Team')['Points'].cumsum()
                        
                        # Create cumulative points chart
                        fig = px.line(
                            team_points,
                            x=race_col,
                            y='CumulativePoints',
                            color='Team',
                            markers=True,
                            title="Cumulative Team Points by Race"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save team comparison to session state
                        if st.button("Save this team comparison"):
                            analysis_name = st.text_input("Analysis Name", value=f"Team Comparison - {', '.join(selected_teams)}")
                            
                            if analysis_name:
                                analysis_data = {
                                    "type": "team_comparison",
                                    "teams": selected_teams,
                                    "points_by_race": team_points.to_dict('records')
                                }
                                
                                SessionStateManager.save_analysis_result(
                                    analysis_name,
                                    analysis_data,
                                    {"type": "team_comparison"}
                                )
                                st.success(f"Analysis '{analysis_name}' saved successfully!")
                    else:
                        st.warning("Could not identify race column in the data")
                else:
                    st.info("Please select at least one team to compare")
            
            # Save championship standings to session state
            if st.button("Save championship standings"):
                if championship_type == "Drivers' Championship":
                    analysis_name = st.text_input("Analysis Name", value="Drivers Championship Standings")
                    standings_data = drivers_standings.to_dict('records')
                    standings_type = "drivers_championship"
                else:
                    analysis_name = st.text_input("Analysis Name", value="Constructors Championship Standings")
                    standings_data = constructors_standings.to_dict('records')
                    standings_type = "constructors_championship"
                
                if analysis_name:
                    analysis_data = {
                        "type": standings_type,
                        "standings": standings_data
                    }
                    
                    SessionStateManager.save_analysis_result(
                        analysis_name,
                        analysis_data,
                        {"type": standings_type}
                    )
                    st.success(f"Championship standings '{analysis_name}' saved successfully!")
else:
    # Check if there are any saved championship analyses
    saved_analyses = [name for name in SessionStateManager.get_analysis_names() 
                     if 'type' in SessionStateManager.analysis_results[name]['metadata'] 
                     and SessionStateManager.analysis_results[name]['metadata']['type'] in 
                     ['drivers_championship', 'constructors_championship', 'team_comparison']]
    
    if saved_analyses:
        st.subheader("Previously Saved Championship Analyses")
        selected_analysis = st.selectbox("Select a previous analysis", saved_analyses)
        
        if st.button("Load Analysis"):
            analysis_data = SessionStateManager.get_analysis_result(selected_analysis)
            analysis_type = SessionStateManager.analysis_results[selected_analysis]['metadata']['type']
            
            st.success(f"Loaded analysis: {selected_analysis}")
            
            if analysis_type == 'drivers_championship':
                st.subheader("Drivers' Championship Standings")
                standings_df = pd.DataFrame(analysis_data['standings'])
                st.dataframe(standings_df, use_container_width=True)
                
                # Create bar chart
                fig = px.bar(
                    standings_df.head(10),
                    x='DriverName',
                    y='Points',
                    title="Top 10 Drivers by Points",
                    labels={'DriverName': 'Driver', 'Points': 'Championship Points'},
                    color='Points',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_type == 'constructors_championship':
                st.subheader("Constructors' Championship Standings")
                standings_df = pd.DataFrame(analysis_data['standings'])
                st.dataframe(standings_df, use_container_width=True)
                
                # Create bar chart
                fig = px.bar(
                    standings_df,
                    x='Team',
                    y='Points',
                    title="Constructors by Points",
                    labels={'Team': 'Constructor', 'Points': 'Championship Points'},
                    color='Points',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_type == 'team_comparison':
                st.subheader("Team Comparison")
                st.write(f"Teams: {', '.join(analysis_data['teams'])}")
                
                points_df = pd.DataFrame(analysis_data['points_by_race'])
                
                # Create cumulative points chart
                fig = px.line(
                    points_df,
                    x='Race' if 'Race' in points_df.columns else 'RaceName' if 'RaceName' in points_df.columns else 'race_col',
                    y='CumulativePoints',
                    color='Team',
                    markers=True,
                    title="Cumulative Team Points by Race"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select a data source to view championship standings.")
        
        # Example data structure
        st.subheader("Example Data Format")
        example_df = pd.DataFrame({
            'Race': ['Bahrain GP', 'Bahrain GP', 'Saudi Arabian GP', 'Saudi Arabian GP'],
            'DriverName': ['Max Verstappen', 'Sergio Perez', 'Max Verstappen', 'Sergio Perez'],
            'Team': ['Red Bull', 'Red Bull', 'Red Bull', 'Red Bull'],
            'Position': [1, 2, 2, 1],
            'Points': [25, 18, 18, 25]
        })
        st.dataframe(example_df)