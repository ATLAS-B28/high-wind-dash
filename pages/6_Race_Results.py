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
    page_title="F1 Race Results",
    page_icon="🏁",
    layout="wide"
)

st.title("Formula 1 Race Results")
st.write("View and analyze Formula 1 race results and standings")

# Function to load race results data
def load_race_results_data():
    base_path = os.path.join(os.getcwd(), "formulaarchive")
    
    # Dictionary to store all dataframes
    dfs = {}
    
    # Load necessary files
    try:
        # Load results data
        results_df = pd.read_csv(os.path.join(base_path, "results.csv"))
        dfs["results"] = results_df
        
        # Load drivers data
        drivers_df = pd.read_csv(os.path.join(base_path, "drivers.csv"))
        dfs["drivers"] = drivers_df
        
        # Load races data
        races_df = pd.read_csv(os.path.join(base_path, "races.csv"))
        dfs["races"] = races_df
        
        # Load constructors data
        constructors_df = pd.read_csv(os.path.join(base_path, "constructors.csv"))
        dfs["constructors"] = constructors_df
        
        # Merge data to get comprehensive race results
        merged_df = results_df.merge(drivers_df, on='driverId')
        merged_df = merged_df.merge(races_df, on='raceId')
        merged_df = merged_df.merge(constructors_df, on='constructorId')
        
        # Create full driver name
        merged_df['driverName'] = merged_df['forename'] + ' ' + merged_df['surname']
        dfs["race_results_complete"] = merged_df
        
        return dfs
    except Exception as e:
        st.error(f"Error loading race results data: {e}")
        return {}

# Data source selection
data_source = st.radio(
    "Choose data source",
    ["Formula Archive data", "Use existing dataset", "Upload new file"],
    horizontal=True,
    index=0  # Default to Formula Archive data
)

results_df = None
drivers_df = None
races_df = None
constructors_df = None
merged_df = None
data_loaded = False

if data_source == "Formula Archive data":
    try:
        # Load Formula Archive data
        f1_dfs = load_race_results_data()
        
        if f1_dfs and "race_results_complete" in f1_dfs:
            results_df = f1_dfs["results"]
            drivers_df = f1_dfs["drivers"]
            races_df = f1_dfs["races"]
            constructors_df = f1_dfs["constructors"]
            merged_df = f1_dfs["race_results_complete"]
            data_loaded = True
            
            st.success("Loaded Formula Archive race results data")
            
            # Option to save to session state
            save_to_session = st.checkbox("Save this dataset for use across all pages", value=False)
            
            if save_to_session:
                if st.button("Save Formula Archive Race Results Data"):
                    # Save individual datasets
                    for key, dataset in f1_dfs.items():
                        SessionStateManager.save_dataset(
                            f"F1_{key}", 
                            dataset, 
                            {'source': f'formulaarchive/{key}'}
                        )
                    st.success("Formula 1 race results datasets saved successfully!")
                    st.rerun()
    except Exception as e:
        st.error(f"Error loading Formula Archive data: {e}")
        st.info("Falling back to other data sources.")
        data_source = "Use existing dataset" if st.session_state.datasets else "Upload new file"

if data_source == "Use existing dataset" and not data_loaded:
    # Dataset selector
    selected_dataset = SessionStateManager.create_dataset_selector("Select a dataset with race results")
    
    if selected_dataset:
        merged_df = SessionStateManager.get_dataset(selected_dataset)
        
        # Check if this dataset has the required columns for race results analysis
        required_cols = ['position', 'points', 'grid']
        if all(col in merged_df.columns for col in required_cols):
            data_loaded = True
            st.success(f"Loaded dataset: {selected_dataset}")
            
            # Try to extract driver names if available
            if 'forename' in merged_df.columns and 'surname' in merged_df.columns and 'driverName' not in merged_df.columns:
                merged_df['driverName'] = merged_df['forename'] + ' ' + merged_df['surname']
            
            # If we don't have races_df but have race info in merged_df
            if 'year' in merged_df.columns:
                races_df = merged_df[['raceId', 'name', 'year', 'date']].drop_duplicates() if 'raceId' in merged_df.columns else None
        else:
            st.warning(f"Selected dataset doesn't have the required columns for race results analysis. Please select another dataset.")

elif data_source == "Upload new file" and not data_loaded:
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file with F1 results data", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load data
        merged_df = load_data(uploaded_file)
        
        if merged_df is not None:
            # Check if this dataset has the required columns for race results analysis
            required_cols = ['position', 'points', 'grid']
            if all(col in merged_df.columns for col in required_cols):
                data_loaded = True
                
                # Try to extract driver names if available
                if 'forename' in merged_df.columns and 'surname' in merged_df.columns and 'driverName' not in merged_df.columns:
                    merged_df['driverName'] = merged_df['forename'] + ' ' + merged_df['surname']
                
                # Option to save to session state
                save_to_session = st.checkbox("Save this dataset for use across all pages", value=True)
                
                if save_to_session:
                    dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])
                    
                    if st.button("Save Dataset"):
                        SessionStateManager.save_dataset(
                            dataset_name, 
                            merged_df, 
                            {'source': uploaded_file.name, 'type': 'race_results'}
                        )
                        st.success(f"Dataset '{dataset_name}' saved successfully!")
                        st.rerun()
            else:
                st.warning("Uploaded file doesn't have the required columns for race results analysis.")
                data_loaded = False

if data_loaded:
    # Analysis type selection
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Race Results", "Season Standings", "Constructor Performance"]
    )
    
    if analysis_type == "Race Results":
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
            
            # Display race results
            st.subheader("Race Results")
            
            # Sort by position
            race_results = race_data.sort_values('position').copy()
            
            # Select columns to display - check if they exist first
            available_cols = race_results.columns.tolist()
            display_cols = ['position', 'driverName']
            display_names = ['Position', 'Driver']
            
            # Add constructor name if available
            if 'name_y' in available_cols:
                display_cols.append('name_y')
                display_names.append('Constructor')
            
            # Add other columns if available
            if 'grid' in available_cols:
                display_cols.append('grid')
                display_names.append('Grid')
            
            if 'statusId' in available_cols:
                display_cols.append('statusId')
                display_names.append('Status')
            
            if 'points' in available_cols:
                display_cols.append('points')
                display_names.append('Points')
            
            # Create a clean dataframe for display
            display_df = race_results[display_cols].copy()
            display_df.columns = display_names
            
            # Display results
            st.dataframe(display_df, use_container_width=True)
            
            # Visualize race results
            st.subheader("Race Visualization")
            
            # Create bar chart of points
            fig = px.bar(
                race_results.sort_values('position'),
                x='driverName',
                y='points',
                color='name_y' if 'name_y' in available_cols else None,  # Constructor name
                title=f"Points by Driver - {race_info['name']} {race_info['year']}",
                labels={'driverName': 'Driver', 'points': 'Points', 'name_y': 'Constructor'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Grid vs Finish position
            st.subheader("Grid vs. Finish Position")
            
            # Create scatter plot
            fig_pos = px.scatter(
                race_results,
                x='grid',
                y='position',
                color='driverName',
                size='points',
                size_max=15,
                hover_name='driverName',
                title=f"Grid vs. Finish Position - {race_info['name']} {race_info['year']}",
                labels={'grid': 'Grid Position', 'position': 'Finish Position'}
            )
            
            # Add diagonal line for reference
            fig_pos.add_trace(
                go.Scatter(
                    x=[0, 20],
                    y=[0, 20],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='Grid = Finish'
                )
            )
            
            # Invert both axes (1 is best)
            fig_pos.update_yaxes(autorange="reversed")
            fig_pos.update_xaxes(autorange="reversed")
            
            st.plotly_chart(fig_pos, use_container_width=True)
            
    elif analysis_type == "Season Standings":
        # Season selection
        st.subheader("Select Season")
        
        # Get unique seasons
        seasons = sorted(races_df['year'].unique(), reverse=True)
        selected_season = st.selectbox("Select Season", seasons)
        
        # Filter data for selected season
        season_data = merged_df[merged_df['year'] == selected_season].copy()
        
        if not season_data.empty:
            # Calculate driver standings
            driver_standings = season_data.groupby(['driverId', 'driverName', 'name_y'])['points'].sum().reset_index()
            driver_standings = driver_standings.sort_values('points', ascending=False).copy()
            
            # Add position column
            driver_standings.loc[:, 'position'] = range(1, len(driver_standings) + 1)
            
            # Display driver standings
            st.subheader(f"{selected_season} Driver Standings")
            
            # Select columns to display
            display_cols = ['position', 'driverName', 'name_y', 'points']
            display_names = ['Position', 'Driver', 'Constructor', 'Points']
            
            # Create a clean dataframe for display
            display_df = driver_standings[display_cols].copy()
            display_df.columns = display_names
            
            # Display standings
            st.dataframe(display_df, use_container_width=True)
            
            # Visualize driver standings
            st.subheader("Driver Standings Visualization")
            
            # Create bar chart of points
            fig = px.bar(
                driver_standings.head(10),
                x='driverName',
                y='points',
                color='name_y',  # Constructor name
                title=f"Top 10 Drivers by Points - {selected_season}",
                labels={'driverName': 'Driver', 'points': 'Points', 'name_y': 'Constructor'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate constructor standings
            constructor_standings = season_data.groupby(['constructorId', 'name_y'])['points'].sum().reset_index()
            constructor_standings = constructor_standings.sort_values('points', ascending=False).copy()
            
            # Add position column
            constructor_standings.loc[:, 'position'] = range(1, len(constructor_standings) + 1)
            
            # Display constructor standings
            st.subheader(f"{selected_season} Constructor Standings")
            
            # Select columns to display
            display_cols = ['position', 'name_y', 'points']
            display_names = ['Position', 'Constructor', 'Points']
            
            # Create a clean dataframe for display
            display_df = constructor_standings[display_cols].copy()
            display_df.columns = display_names
            
            # Display standings
            st.dataframe(display_df, use_container_width=True)
            
            # Visualize constructor standings
            st.subheader("Constructor Standings Visualization")
            
            # Create bar chart of points
            fig = px.bar(
                constructor_standings,
                x='name_y',
                y='points',
                color='name_y',  # Constructor name
                title=f"Constructor Points - {selected_season}",
                labels={'name_y': 'Constructor', 'points': 'Points'}
            )
            
            fig.update_layout(showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
    elif analysis_type == "Constructor Performance":
        # Constructor selection
        st.subheader("Select Constructor")
        
        # Get unique constructors
        constructors = constructors_df[['constructorId', 'name']].sort_values('name')
        
        selected_constructor_id = st.selectbox(
            "Select Constructor",
            constructors['constructorId'].tolist(),
            format_func=lambda x: constructors[constructors['constructorId'] == x]['name'].iloc[0]
        )
        
        # Filter data for selected constructor
        constructor_data = merged_df[merged_df['constructorId'] == selected_constructor_id].copy()
        
        if not constructor_data.empty:
            # Get constructor info
            constructor_info = constructors[constructors['constructorId'] == selected_constructor_id].iloc[0]
            st.write(f"**Constructor:** {constructor_info['name']}")
            
            # Calculate statistics
            total_races = constructor_data['raceId'].nunique()
            total_wins = len(constructor_data[constructor_data['position'] == 1])
            total_podiums = len(constructor_data[constructor_data['position'].isin([1, 2, 3])])
            total_points = constructor_data['points'].sum()
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Races", total_races)
            col2.metric("Wins", total_wins)
            col3.metric("Podiums", total_podiums)
            col4.metric("Total Points", f"{total_points:.1f}")
            
            # Performance by season
            st.subheader("Performance by Season")
            
            # Calculate points by season
            season_performance = constructor_data.groupby('year')['points'].sum().reset_index()
            season_performance = season_performance.sort_values('year').copy()
            
            # Create line chart
            fig = px.line(
                season_performance,
                x='year',
                y='points',
                markers=True,
                title=f"{constructor_info['name']} Points by Season",
                labels={'year': 'Season', 'points': 'Points'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Driver performance
            st.subheader("Driver Performance")
            
            # Calculate points by driver
            driver_performance = constructor_data.groupby(['driverName', 'year'])['points'].sum().reset_index().copy()
            
            # Create grouped bar chart
            fig = px.bar(
                driver_performance,
                x='driverName',
                y='points',
                color='year',
                title=f"{constructor_info['name']} Points by Driver",
                labels={'driverName': 'Driver', 'points': 'Points', 'year': 'Season'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Save constructor analysis to session state
            if st.button("Save this constructor analysis"):
                analysis_name = st.text_input("Analysis Name", value=f"Constructor Analysis - {constructor_info['name']}")
                
                if analysis_name:
                    analysis_data = {
                        "type": "constructor_analysis",
                        "constructor_id": selected_constructor_id,
                        "constructor_name": constructor_info['name'],
                        "stats": {
                            "races": total_races,
                            "wins": total_wins,
                            "podiums": total_podiums,
                            "total_points": float(total_points)
                        },
                        "season_performance": season_performance.to_dict('records'),
                        "driver_performance": driver_performance.to_dict('records')
                    }
                    
                    SessionStateManager.save_analysis_result(
                        analysis_name,
                        analysis_data,
                        {"type": "constructor_analysis"}
                    )
                    st.success(f"Analysis '{analysis_name}' saved successfully!")
    
    # Save race results or season standings to session state
    if analysis_type in ["Race Results", "Season Standings"] and st.button("Save this analysis"):
        if analysis_type == "Race Results":
            analysis_name = st.text_input("Analysis Name", value=f"Race Results - {race_info['name']} {race_info['year']}")
            analysis_data = {
                "type": "race_results",
                "race_id": selected_race_id,
                "race_name": f"{race_info['name']} {race_info['year']}",
                "results": race_results.to_dict('records')
            }
            analysis_type_meta = "race_results"
        else:  # Season Standings
            analysis_name = st.text_input("Analysis Name", value=f"Season Standings - {selected_season}")
            analysis_data = {
                "type": "season_standings",
                "season": selected_season,
                "driver_standings": driver_standings.to_dict('records'),
                "constructor_standings": constructor_standings.to_dict('records')
            }
            analysis_type_meta = "season_standings"
        
        if analysis_name:
            SessionStateManager.save_analysis_result(
                analysis_name,
                analysis_data,
                {"type": analysis_type_meta}
            )
            st.success(f"Analysis '{analysis_name}' saved successfully!")
else:
    # Check if there are any saved race results analyses
    saved_analyses = [name for name in SessionStateManager.get_analysis_names() 
                     if 'type' in SessionStateManager.analysis_results[name]['metadata'] 
                     and SessionStateManager.analysis_results[name]['metadata']['type'] in 
                     ['race_results', 'season_standings', 'constructor_analysis']]
    
    if saved_analyses:
        st.subheader("Previously Saved Race Results Analyses")
        selected_analysis = st.selectbox("Select a previous analysis", saved_analyses)
        
        if st.button("Load Analysis"):
            analysis_data = SessionStateManager.get_analysis_result(selected_analysis)
            analysis_type = SessionStateManager.analysis_results[selected_analysis]['metadata']['type']
            
            st.success(f"Loaded analysis: {selected_analysis}")
            
            if analysis_type == 'race_results':
                st.subheader(f"Race Results - {analysis_data['race_name']}")
                results_df = pd.DataFrame(analysis_data['results'])
                
                # Display race results
                if 'driverName' in results_df.columns and 'position' in results_df.columns:
                    # Select columns to display
                    display_cols = ['position', 'driverName']
                    display_names = ['Position', 'Driver']
                    
                    # Add constructor if available
                    if 'name_y' in results_df.columns:
                        display_cols.append('name_y')
                        display_names.append('Constructor')
                    
                    # Add other columns if available
                    for col, name in zip(['grid', 'statusId', 'points'], ['Grid', 'Status', 'Points']):
                        if col in results_df.columns:
                            display_cols.append(col)
                            display_names.append(name)
                    
                    # Create a clean dataframe for display
                    display_df = results_df[display_cols].copy()
                    display_df.columns = display_names
                    
                    # Display results
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Create bar chart of points if available
                    if 'points' in results_df.columns:
                        fig = px.bar(
                            results_df.sort_values('position'),
                            x='driverName',
                            y='points',
                            color='name_y' if 'name_y' in results_df.columns else None,
                            title=f"Points by Driver - {analysis_data['race_name']}",
                            labels={'driverName': 'Driver', 'points': 'Points', 'name_y': 'Constructor'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(results_df, use_container_width=True)
                
            elif analysis_type == 'season_standings':
                st.subheader(f"Season Standings - {analysis_data['season']}")
                
                # Driver standings
                st.write("Driver Standings")
                driver_df = pd.DataFrame(analysis_data['driver_standings'])
                st.dataframe(driver_df, use_container_width=True)
                
                # Constructor standings
                st.write("Constructor Standings")
                constructor_df = pd.DataFrame(analysis_data['constructor_standings'])
                st.dataframe(constructor_df, use_container_width=True)
                
                # Create visualizations if possible
                if 'driverName' in driver_df.columns and 'points' in driver_df.columns:
                    fig = px.bar(
                        driver_df.head(10),
                        x='driverName',
                        y='points',
                        color='name_y' if 'name_y' in driver_df.columns else None,
                        title=f"Top 10 Drivers by Points - {analysis_data['season']}",
                        labels={'driverName': 'Driver', 'points': 'Points', 'name_y': 'Constructor'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_type == 'constructor_analysis':
                st.subheader(f"Constructor Analysis - {analysis_data['constructor_name']}")
                
                # Display statistics
                stats = analysis_data['stats']
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Races", stats['races'])
                col2.metric("Wins", stats['wins'])
                col3.metric("Podiums", stats['podiums'])
                col4.metric("Total Points", f"{stats['total_points']:.1f}")
                
                # Performance by season
                st.subheader("Performance by Season")
                season_df = pd.DataFrame(analysis_data['season_performance'])
                
                # Create line chart
                fig = px.line(
                    season_df,
                    x='year',
                    y='points',
                    markers=True,
                    title=f"{analysis_data['constructor_name']} Points by Season",
                    labels={'year': 'Season', 'points': 'Points'}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select a data source to view race results.")
        
        # Example data structure
        st.subheader("Example Data Format")
        example_df = pd.DataFrame({
            'raceId': [841, 841, 841, 841],
            'driverId': [20, 1, 17, 808],
            'constructorId': [6, 1, 3, 2],
            'position': [1, 2, 3, 4],
            'points': [25, 18, 15, 12],
            'grid': [1, 2, 3, 4],
            'statusId': [1, 1, 1, 1]
        })
        st.dataframe(example_df)