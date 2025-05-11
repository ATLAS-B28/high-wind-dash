import streamlit as st
import pandas as pd
import numpy as np
import os
from utils import load_data, create_filtered_dataframe, get_column_types, SessionStateManager

# Initialize session state
SessionStateManager.initialize_session_state()

st.set_page_config(
    page_title="Data Analysis",
    page_icon="🔍",
    layout="wide"
)

st.title("Data Analysis")
st.write("Analyze your data with statistical methods")

# Function to load F1 data from formulaarchive
def load_f1_data():
    base_path = os.path.join(os.getcwd(), "formulaarchive")
    
    # Dictionary to store all dataframes
    dfs = {}
    
    # List of CSV files to load
    csv_files = [
        "drivers.csv", "races.csv", "results.csv", "constructors.csv", 
        "constructor_standings.csv", "driver_standings.csv", "circuits.csv",
        "lap_times.csv", "pit_stops.csv", "qualifying.csv", "seasons.csv"
    ]
    
    # Load each CSV file
    for file in csv_files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            df_name = file.split('.')[0]
            dfs[df_name] = pd.read_csv(file_path)
    
    # Create merged dataframes for common analyses
    if all(key in dfs for key in ["results", "drivers", "races"]):
        # Merge results with drivers and races
        dfs["results_drivers_races"] = dfs["results"].merge(
            dfs["drivers"], on='driverId'
        ).merge(
            dfs["races"], on='raceId'
        )
    
    if all(key in dfs for key in ["driver_standings", "drivers", "races"]):
        # Merge driver standings with drivers and races
        dfs["driver_standings_complete"] = dfs["driver_standings"].merge(
            dfs["drivers"], on='driverId'
        ).merge(
            dfs["races"], on='raceId'
        )
    
    if all(key in dfs for key in ["constructor_standings", "constructors", "races"]):
        # Merge constructor standings with constructors and races
        dfs["constructor_standings_complete"] = dfs["constructor_standings"].merge(
            dfs["constructors"], on='constructorId'
        ).merge(
            dfs["races"], on='raceId'
        )
    
    if all(key in dfs for key in ["lap_times", "drivers", "races"]):
        # Merge lap times with drivers and races
        dfs["lap_times_complete"] = dfs["lap_times"].merge(
            dfs["drivers"], on='driverId'
        ).merge(
            dfs["races"], on='raceId'
        )
    
    return dfs

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
        f1_dfs = load_f1_data()
        
        if f1_dfs:
            # Let user select which dataset to analyze
            selected_f1_dataset = st.selectbox(
                "Select Formula 1 dataset to analyze",
                options=list(f1_dfs.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            df = f1_dfs[selected_f1_dataset]
            st.success(f"Loaded Formula Archive dataset: {selected_f1_dataset}")
            
            # Option to save to session state
            save_to_session = st.checkbox("Save this dataset for use across all pages", value=False)
            
            if save_to_session:
                if st.button("Save Formula Archive Dataset"):
                    SessionStateManager.save_dataset(
                        f"F1_{selected_f1_dataset}", 
                        df, 
                        {'source': f'formulaarchive/{selected_f1_dataset}'}
                    )
                    st.success(f"Dataset 'F1_{selected_f1_dataset}' saved successfully!")
                    st.rerun()
    except Exception as e:
        st.error(f"Error loading Formula Archive data: {e}")
        st.info("Falling back to other data sources.")
        data_source = "Use existing dataset" if st.session_state.datasets else "Upload new file"

if data_source == "Use existing dataset":
    # Dataset selector
    selected_dataset = SessionStateManager.create_dataset_selector("Select a dataset to analyze")
    
    if selected_dataset:
        df = SessionStateManager.get_dataset(selected_dataset)
        st.success(f"Loaded dataset: {selected_dataset}")

elif data_source == "Upload new file":
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
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
                        {'source': uploaded_file.name}
                    )
                    st.success(f"Dataset '{dataset_name}' saved successfully!")
                    st.rerun()

if df is not None:
        # Create filtered dataframe
        filtered_df = create_filtered_dataframe(df)
        
        # Get column types
        column_types = get_column_types(filtered_df)
        
        # Analysis options
        st.subheader("Analysis Options")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Correlation Analysis", "Group Analysis", "Time Series Analysis"]
        )
        
        if analysis_type == "Correlation Analysis":
            st.write("Analyze correlations between numeric variables")
            
            # Only show correlation for dataframes with at least 2 numeric columns
            if len(column_types['numeric']) >= 2:
                # Correlation matrix
                corr_matrix = filtered_df[column_types['numeric']].corr()
                
                # Display correlation heatmap
                st.subheader("Correlation Matrix")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
                
                # Select columns for scatter plot
                st.subheader("Correlation Scatter Plot")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("X-axis", column_types['numeric'])
                
                with col2:
                    y_col = st.selectbox("Y-axis", column_types['numeric'], 
                                        index=min(1, len(column_types['numeric'])-1))
                
                # Calculate correlation
                correlation = filtered_df[x_col].corr(filtered_df[y_col])
                st.metric("Correlation Coefficient", f"{correlation:.4f}")
                
                # Create scatter plot
                chart = st.scatter_chart(
                    filtered_df,
                    x=x_col,
                    y=y_col
                )
                
                # Save correlation analysis to session state
                if st.button("Save this correlation analysis"):
                    analysis_name = st.text_input("Analysis Name", value=f"Correlation of {x_col} vs {y_col}")
                    
                    if analysis_name:
                        analysis_data = {
                            "type": "correlation_analysis",
                            "x_column": x_col,
                            "y_column": y_col,
                            "correlation_coefficient": correlation,
                            "correlation_matrix": corr_matrix.to_dict()
                        }
                        
                        SessionStateManager.save_analysis_result(
                            analysis_name,
                            analysis_data,
                            {"type": "correlation_analysis"}
                        )
                        st.success(f"Analysis '{analysis_name}' saved successfully!")
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        elif analysis_type == "Group Analysis":
            st.write("Analyze data grouped by categorical variables")
            
            if len(column_types['categorical']) > 0:
                # Select grouping column
                group_col = st.selectbox("Group by", column_types['categorical'])
                
                # Select aggregation columns
                agg_cols = st.multiselect("Select columns to aggregate", 
                                         column_types['numeric'], 
                                         default=column_types['numeric'][:min(3, len(column_types['numeric']))])
                
                if agg_cols:
                    # Select aggregation functions
                    agg_funcs = st.multiselect(
                        "Select aggregation functions",
                        ["mean", "sum", "count", "min", "max", "std"],
                        default=["mean", "sum", "count"]
                    )
                    
                    if agg_funcs:
                        # Create aggregation dictionary
                        agg_dict = {col: agg_funcs for col in agg_cols}
                        
                        # Perform groupby operation
                        grouped_df = filtered_df.groupby(group_col).agg(agg_dict)
                        
                        # Display results
                        st.subheader("Group Analysis Results")
                        st.dataframe(grouped_df, use_container_width=True)
                        
                        # Download grouped data
                        st.download_button(
                            label="Download grouped data as CSV",
                            data=grouped_df.to_csv().encode('utf-8'),
                            file_name=f"grouped_analysis.csv",
                            mime="text/csv"
                        )
                        
                        # Save analysis to session state
                        if st.button("Save this analysis to session state"):
                            analysis_name = st.text_input("Analysis Name", value=f"Group Analysis by {group_col}")
                            
                            if analysis_name:
                                analysis_data = {
                                    "type": "group_analysis",
                                    "group_column": group_col,
                                    "aggregation_columns": agg_cols,
                                    "aggregation_functions": agg_funcs,
                                    "results": grouped_df.reset_index().to_dict('records')
                                }
                                
                                SessionStateManager.save_analysis_result(
                                    analysis_name,
                                    analysis_data,
                                    {"type": "group_analysis"}
                                )
                                st.success(f"Analysis '{analysis_name}' saved successfully!")
                        
                        # Visualize grouped data
                        st.subheader("Group Visualization")
                        
                        # Select specific aggregation to visualize
                        if len(agg_funcs) > 1:
                            selected_agg = st.selectbox("Select aggregation to visualize", agg_funcs)
                            
                            # Get column names with selected aggregation
                            viz_cols = [col for col in grouped_df.columns if col[1] == selected_agg]
                            
                            # Select column to visualize
                            if viz_cols:
                                viz_col = st.selectbox("Select column to visualize", 
                                                     [col[0] for col in viz_cols])
                                
                                # Create bar chart
                                chart_data = grouped_df.xs(selected_agg, level=1, axis=1)[viz_col].reset_index()
                                st.bar_chart(chart_data, x=group_col, y=viz_col)
                    else:
                        st.info("Please select at least one aggregation function.")
                else:
                    st.info("Please select at least one column to aggregate.")
            else:
                st.warning("Need at least one categorical column for group analysis.")
        
        elif analysis_type == "Time Series Analysis":
            st.write("Analyze time series data")
            
            if len(column_types['datetime']) > 0:
                # Select datetime column
                time_col = st.selectbox("Select time column", column_types['datetime'])
                
                # Select value column
                value_col = st.selectbox("Select value column", column_types['numeric'])
                
                # Select time frequency for resampling
                freq = st.selectbox(
                    "Select time frequency",
                    ["Day", "Week", "Month", "Quarter", "Year"],
                    index=0
                )
                
                # Map selected frequency to pandas frequency string
                freq_map = {
                    "Day": "D",
                    "Week": "W",
                    "Month": "M",
                    "Quarter": "Q",
                    "Year": "Y"
                }
                
                # Set datetime as index
                ts_df = filtered_df.copy()
                ts_df[time_col] = pd.to_datetime(ts_df[time_col])
                ts_df = ts_df.set_index(time_col)
                
                # Resample time series
                resampled = ts_df[value_col].resample(freq_map[freq]).agg(['mean', 'sum', 'count'])
                
                # Display resampled data
                st.subheader(f"Time Series Analysis ({freq} Frequency)")
                st.dataframe(resampled, use_container_width=True)
                
                # Visualize time series
                st.subheader("Time Series Visualization")
                
                # Select aggregation to visualize
                agg_method = st.selectbox("Select aggregation method", ["mean", "sum", "count"])
                
                # Create line chart
                chart_data = resampled[[agg_method]].reset_index()
                st.line_chart(chart_data, x=time_col, y=agg_method)
                
                # Save time series analysis to session state
                if st.button("Save this time series analysis"):
                    analysis_name = st.text_input("Analysis Name", value=f"Time Series Analysis of {value_col} by {freq}")
                    
                    if analysis_name:
                        analysis_data = {
                            "type": "time_series_analysis",
                            "time_column": time_col,
                            "value_column": value_col,
                            "frequency": freq,
                            "aggregation_method": agg_method,
                            "results": resampled.reset_index().to_dict('records')
                        }
                        
                        SessionStateManager.save_analysis_result(
                            analysis_name,
                            analysis_data,
                            {"type": "time_series_analysis"}
                        )
                        st.success(f"Analysis '{analysis_name}' saved successfully!")
            else:
                st.warning("Need at least one datetime column for time series analysis.")
        
        # Special F1 Analysis section for Formula Archive data
        if data_source == "Formula Archive data":
            st.subheader("Formula 1 Specific Analysis")
            
            f1_analysis_type = st.selectbox(
                "Select F1 Analysis Type",
                ["Driver Performance", "Constructor Performance", "Circuit Analysis", "Season Comparison"]
            )
            
            if f1_analysis_type == "Driver Performance":
                # Check if we have the necessary columns for driver performance analysis
                required_cols = ['driverId', 'position', 'points']
                if all(col in df.columns for col in required_cols):
                    # Select driver(s) to analyze
                    if 'forename' in df.columns and 'surname' in df.columns:
                        # Create full name for better selection
                        driver_options = df.apply(lambda x: f"{x['forename']} {x['surname']} (ID: {x['driverId']})", axis=1).unique()
                        selected_drivers = st.multiselect("Select drivers to analyze", options=driver_options)
                        
                        if selected_drivers:
                            # Extract driver IDs from selection
                            driver_ids = [int(driver.split("(ID: ")[1].split(")")[0]) for driver in selected_drivers]
                            
                            # Filter data for selected drivers
                            driver_data = df[df['driverId'].isin(driver_ids)]
                            
                            # Analyze performance
                            st.subheader("Driver Performance Analysis")
                            
                            # Group by driver and calculate stats
                            driver_stats = driver_data.groupby('driverId').agg({
                                'points': ['sum', 'mean'],
                                'position': ['mean', 'min']
                            })
                            
                            # Rename columns for clarity
                            driver_stats.columns = ['Total Points', 'Avg Points per Race', 'Avg Position', 'Best Position']
                            
                            # Display stats
                            st.dataframe(driver_stats, use_container_width=True)
                            
                            # Visualize points progression if we have race dates
                            if 'date' in df.columns:
                                st.subheader("Points Progression")
                                
                                # Prepare data for visualization
                                points_data = driver_data.sort_values('date')
                                
                                # Create cumulative points column
                                for driver_id in driver_ids:
                                    mask = points_data['driverId'] == driver_id
                                    points_data.loc[mask, 'cumulative_points'] = points_data.loc[mask, 'points'].cumsum()
                                
                                # Create line chart
                                driver_names = {
                                    driver_id: next(name for name in selected_drivers if f"(ID: {driver_id})" in name)
                                    for driver_id in driver_ids
                                }
                                
                                # Create chart data with driver names
                                chart_data = pd.DataFrame()
                                for driver_id in driver_ids:
                                    driver_points = points_data[points_data['driverId'] == driver_id]
                                    if not driver_points.empty:
                                        driver_chart_data = pd.DataFrame({
                                            'date': driver_points['date'],
                                            'points': driver_points['cumulative_points'],
                                            'driver': driver_names[driver_id].split(" (ID:")[0]
                                        })
                                        chart_data = pd.concat([chart_data, driver_chart_data])
                                
                                # Plot the chart
                                st.line_chart(chart_data, x='date', y='points', color='driver')
                    else:
                        st.warning("Driver name columns not found in the dataset.")
                else:
                    st.warning("Required columns for driver performance analysis not found in the dataset.")
            
            elif f1_analysis_type == "Constructor Performance":
                # Check if we have constructor data
                if 'constructorId' in df.columns:
                    # Select constructor(s) to analyze
                    if 'name' in df.columns and 'constructorId' in df.columns:
                        constructor_options = df[['constructorId', 'name']].drop_duplicates().apply(
                            lambda x: f"{x['name']} (ID: {x['constructorId']})", axis=1
                        ).tolist()
                        
                        selected_constructors = st.multiselect("Select constructors to analyze", options=constructor_options)
                        
                        if selected_constructors:
                            # Extract constructor IDs from selection
                            constructor_ids = [int(constructor.split("(ID: ")[1].split(")")[0]) for constructor in selected_constructors]
                            
                            # Filter data for selected constructors
                            constructor_data = df[df['constructorId'].isin(constructor_ids)]
                            
                            # Analyze performance
                            st.subheader("Constructor Performance Analysis")
                            
                            # Group by constructor and calculate stats
                            if 'points' in df.columns:
                                constructor_stats = constructor_data.groupby('constructorId').agg({
                                    'points': ['sum', 'mean']
                                })
                                
                                # Rename columns for clarity
                                constructor_stats.columns = ['Total Points', 'Avg Points per Race']
                                
                                # Display stats
                                st.dataframe(constructor_stats, use_container_width=True)
                                
                                # Visualize points progression if we have race dates
                                if 'date' in df.columns:
                                    st.subheader("Points Progression")
                                    
                                    # Prepare data for visualization
                                    points_data = constructor_data.sort_values('date')
                                    
                                    # Create cumulative points column
                                    for constructor_id in constructor_ids:
                                        mask = points_data['constructorId'] == constructor_id
                                        points_data.loc[mask, 'cumulative_points'] = points_data.loc[mask, 'points'].cumsum()
                                    
                                    # Create line chart
                                    constructor_names = {
                                        constructor_id: next(name.split(" (ID:")[0] for name in selected_constructors if f"(ID: {constructor_id})" in name)
                                        for constructor_id in constructor_ids
                                    }
                                    
                                    # Create chart data with constructor names
                                    chart_data = pd.DataFrame()
                                    for constructor_id in constructor_ids:
                                        constructor_points = points_data[points_data['constructorId'] == constructor_id]
                                        if not constructor_points.empty:
                                            constructor_chart_data = pd.DataFrame({
                                                'date': constructor_points['date'],
                                                'points': constructor_points['cumulative_points'],
                                                'constructor': constructor_names[constructor_id]
                                            })
                                            chart_data = pd.concat([chart_data, constructor_chart_data])
                                    
                                    # Plot the chart
                                    st.line_chart(chart_data, x='date', y='points', color='constructor')
                            else:
                                st.warning("Points column not found in the dataset.")
                    else:
                        st.warning("Constructor name columns not found in the dataset.")
                else:
                    st.warning("Constructor data not found in the dataset.")
            
            elif f1_analysis_type == "Circuit Analysis":
                # Check if we have circuit data
                if 'circuitId' in df.columns:
                    # Select circuit(s) to analyze
                    circuit_options = df[['circuitId', 'name']].drop_duplicates().apply(
                        lambda x: f"{x['name']} (ID: {x['circuitId']})", axis=1
                    ).tolist()
                    
                    selected_circuit = st.selectbox("Select circuit to analyze", options=circuit_options)
                    
                    if selected_circuit:
                        # Extract circuit ID from selection
                        circuit_id = int(selected_circuit.split("(ID: ")[1].split(")")[0])
                        
                        # Filter data for selected circuit
                        circuit_data = df[df['circuitId'] == circuit_id]
                        
                        # Analyze circuit performance
                        st.subheader("Circuit Analysis")
                        
                        # Show race winners at this circuit if we have position data
                        if 'position' in df.columns and 'forename' in df.columns and 'surname' in df.columns:
                            winners = circuit_data[circuit_data['position'] == 1].sort_values('date', ascending=False)
                            
                            if not winners.empty:
                                st.write("Race Winners at this Circuit:")
                                
                                # Create a dataframe with winner information
                                winner_df = winners.apply(
                                    lambda x: pd.Series({
                                        'Date': x['date'],
                                        'Driver': f"{x['forename']} {x['surname']}",
                                        'Constructor': x['name'] if 'name' in x else 'Unknown',
                                        'Grid': x['grid'] if 'grid' in x else 'Unknown',
                                        'Fastest Lap': x['fastestLapTime'] if 'fastestLapTime' in x else 'Unknown'
                                    }), axis=1
                                )
                                
                                st.dataframe(winner_df, use_container_width=True)
                        else:
                            st.warning("Position or driver name columns not found in the dataset.")
                else:
                    st.warning("Circuit data not found in the dataset.")
            
            elif f1_analysis_type == "Season Comparison":
                # Check if we have season/year data
                if 'year' in df.columns or 'season' in df.columns:
                    # Determine which column to use
                    season_col = 'year' if 'year' in df.columns else 'season'
                    
                    # Get list of seasons
                    seasons = sorted(df[season_col].unique())
                    
                    # Select seasons to compare
                    selected_seasons = st.multiselect("Select seasons to compare", options=seasons, default=seasons[-2:] if len(seasons) >= 2 else seasons)
                    
                    if len(selected_seasons) >= 2:
                        # Filter data for selected seasons
                        season_data = df[df[season_col].isin(selected_seasons)]
                        
                        # Analyze season comparison
                        st.subheader("Season Comparison")
                        
                        # Compare driver performance across seasons
                        if 'driverId' in df.columns and 'points' in df.columns:
                            # Group by season and driver
                            season_driver_points = season_data.groupby([season_col, 'driverId']).agg({
                                'points': 'sum'
                            }).reset_index()
                            
                            # Add driver names if available
                            if 'forename' in df.columns and 'surname' in df.columns:
                                # Create a mapping of driver IDs to names
                                driver_names = df[['driverId', 'forename', 'surname']].drop_duplicates()
                                driver_names['driver_name'] = driver_names.apply(lambda x: f"{x['forename']} {x['surname']}", axis=1)
                                
                                # Merge with season_driver_points
                                season_driver_points = season_driver_points.merge(
                                    driver_names[['driverId', 'driver_name']], 
                                    on='driverId'
                                )
                                
                                # Pivot the data for comparison
                                pivot_df = season_driver_points.pivot(
                                    index='driver_name', 
                                    columns=season_col, 
                                    values='points'
                                ).fillna(0)
                                
                                # Sort by the most recent season
                                pivot_df = pivot_df.sort_values(by=selected_seasons[-1], ascending=False)
                                
                                # Display the comparison
                                st.write("Driver Points Comparison Across Seasons:")
                                st.dataframe(pivot_df, use_container_width=True)
                                
                                # Visualize top drivers across seasons
                                top_n = st.slider("Select number of top drivers to visualize", min_value=3, max_value=20, value=10)
                                top_drivers = pivot_df.head(top_n).index.tolist()
                                
                                # Filter data for top drivers
                                top_driver_data = season_driver_points[season_driver_points['driver_name'].isin(top_drivers)]
                                
                                # Create bar chart
                                st.subheader(f"Top {top_n} Drivers Across Seasons")
                                st.bar_chart(top_driver_data, x='driver_name', y='points', color=season_col)
                            else:
                                st.warning("Driver name columns not found in the dataset.")
                        else:
                            st.warning("Driver ID or points columns not found in the dataset.")
                    else:
                        st.info("Please select at least two seasons to compare.")
                else:
                    st.warning("Season/year data not found in the dataset.")
            
            # Save F1 analysis to session state
            if st.button("Save this F1 analysis"):
                analysis_name = st.text_input("Analysis Name", value=f"F1 {f1_analysis_type} Analysis")
                
                if analysis_name:
                    # Create a simplified version of the analysis data
                    analysis_data = {
                        "type": "f1_analysis",
                        "analysis_type": f1_analysis_type,
                        "dataset": selected_f1_dataset if 'selected_f1_dataset' in locals() else None
                    }
                    
                    SessionStateManager.save_analysis_result(
                        analysis_name,
                        analysis_data,
                        {"type": "f1_analysis", "subtype": f1_analysis_type}
                    )
                    st.success(f"Analysis '{analysis_name}' saved successfully!")
else:
    # Check if there are any saved analyses
    saved_analyses = [name for name in SessionStateManager.get_analysis_names() 
                     if 'type' in SessionStateManager.analysis_results[name]['metadata']]
    
    if saved_analyses:
        st.subheader("Previously Saved Analyses")
        selected_analysis = st.selectbox("Select a previous analysis", saved_analyses)
        
        if st.button("Load Analysis"):
            analysis_data = SessionStateManager.get_analysis_result(selected_analysis)
            analysis_type = SessionStateManager.analysis_results[selected_analysis]['metadata'].get('type')
            
            st.success(f"Loaded analysis: {selected_analysis}")
            st.json(analysis_data)
    else:
        st.info("Please select a data source to analyze data.")