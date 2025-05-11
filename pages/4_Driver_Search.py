import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from utils import load_data, search_driver, get_driver_stats, create_driver_position_chart, SessionStateManager

# Initialize session state
SessionStateManager.initialize_session_state()

st.set_page_config(
    page_title="F1 Driver Search",
    page_icon="🏎️",
    layout="wide"
)

st.title("Formula 1 Driver Search")
st.write("Search for F1 drivers and view their statistics")

# Function to load F1 data from formulaarchive
def load_f1_data():
    base_path = os.path.join(os.getcwd(), "formulaarchive")
    
    # Load drivers data
    drivers_df = pd.read_csv(os.path.join(base_path, "drivers.csv"))
    
    # Load races data
    races_df = pd.read_csv(os.path.join(base_path, "races.csv"))
    
    # Load results data
    results_df = pd.read_csv(os.path.join(base_path, "results.csv"))
    
    # Merge data to get comprehensive driver information
    merged_df = results_df.merge(drivers_df, on='driverId')
    merged_df = merged_df.merge(races_df, on='raceId')
    
    return drivers_df, races_df, results_df, merged_df

# Data source selection
data_source = st.radio(
    "Choose data source",
    ["Use existing dataset", "Use Formula Archive data", "Upload new file"],
    horizontal=True,
    index=0 if st.session_state.datasets else 1
)

data_loaded = False
drivers_df = None
races_df = None
results_df = None
merged_df = None

if data_source == "Use existing dataset":
    # Dataset selector
    selected_dataset = SessionStateManager.create_dataset_selector("Select a dataset")
    
    if selected_dataset:
        merged_df = SessionStateManager.get_dataset(selected_dataset)
        st.success(f"Loaded dataset: {selected_dataset}")
        data_loaded = True
        
        # Check if this is a merged dataset or just drivers
        if 'forename' in merged_df.columns and 'surname' in merged_df.columns:
            drivers_df = merged_df
        else:
            st.warning("Selected dataset doesn't appear to contain driver information. Please select another dataset.")
            data_loaded = False

elif data_source == "Use Formula Archive data":
    try:
        drivers_df, races_df, results_df, merged_df = load_f1_data()
        data_loaded = True
        
        # Option to save to session state
        save_to_session = st.checkbox("Save this dataset for use across all pages", value=True)
        
        if save_to_session:
            if st.button("Save Formula Archive Data"):
                # Save individual datasets
                SessionStateManager.save_dataset("F1_Drivers", drivers_df, {'source': 'formulaarchive/drivers.csv'})
                SessionStateManager.save_dataset("F1_Races", races_df, {'source': 'formulaarchive/races.csv'})
                SessionStateManager.save_dataset("F1_Results", results_df, {'source': 'formulaarchive/results.csv'})
                SessionStateManager.save_dataset("F1_Merged", merged_df, {'source': 'formulaarchive merged data'})
                st.success("Formula 1 datasets saved successfully!")
                st.rerun()
    except Exception as e:
        st.error(f"Error loading F1 data: {e}")
        data_loaded = False

else:  # Upload new file
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file with F1 data", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load data
        merged_df = load_data(uploaded_file)
        if merged_df is not None:
            data_loaded = True
            
            # Check if this is a merged dataset or just drivers
            if 'forename' in merged_df.columns and 'surname' in merged_df.columns:
                drivers_df = merged_df
            else:
                st.warning("Uploaded file doesn't appear to contain driver information.")
                data_loaded = False
            
            # Option to save to session state
            save_to_session = st.checkbox("Save this dataset for use across all pages", value=True)
            
            if save_to_session:
                dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])
                
                if st.button("Save Dataset"):
                    SessionStateManager.save_dataset(
                        dataset_name, 
                        merged_df, 
                        {'source': uploaded_file.name}
                    )
                    st.success(f"Dataset '{dataset_name}' saved successfully!")
                    st.rerun()

if data_loaded:
    # Search interface
    st.subheader("Search for a Driver")
    
    # Create a more comprehensive search
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_term = st.text_input("Enter driver name, number, or nationality")
    
    with search_col2:
        search_by = st.selectbox("Search by", ["Name", "Number", "Nationality", "All"])
    
    if search_term:
        # Search for driver based on selected criteria
        if drivers_df is not None:
            if search_by == "Name":
                search_results = drivers_df[drivers_df['forename'].str.contains(search_term, case=False, na=False) | 
                                          drivers_df['surname'].str.contains(search_term, case=False, na=False)]
            elif search_by == "Number":
                try:
                    search_results = drivers_df[drivers_df['number'] == int(search_term)]
                except ValueError:
                    search_results = pd.DataFrame()
            elif search_by == "Nationality":
                search_results = drivers_df[drivers_df['nationality'].str.contains(search_term, case=False, na=False)]
            else:  # All
                search_results = drivers_df[drivers_df['forename'].str.contains(search_term, case=False, na=False) | 
                                          drivers_df['surname'].str.contains(search_term, case=False, na=False) |
                                          drivers_df['nationality'].str.contains(search_term, case=False, na=False) |
                                          drivers_df['code'].str.contains(search_term, case=False, na=False)]
                try:
                    number_results = drivers_df[drivers_df['number'] == int(search_term)]
                    search_results = pd.concat([search_results, number_results]).drop_duplicates()
                except ValueError:
                    pass
                    
            # Save search results to session state for use in other pages
            if not search_results.empty:
                SessionStateManager.save_analysis_result(
                    f"Driver_Search_{search_term}",
                    search_results,
                    {"type": "driver_search", "search_term": search_term}
                )
        
        if not search_results.empty:
            st.success(f"Found {len(search_results)} results for '{search_term}'")
            
            # Display search results
            st.dataframe(search_results[['driverId', 'code', 'number', 'forename', 'surname', 'nationality', 'dob']], use_container_width=True)
            
            # If multiple results, let user select one
            if len(search_results) > 1:
                selected_driver_id = st.selectbox("Select a driver to view details", 
                                               search_results['driverId'].tolist(),
                                               format_func=lambda x: f"{search_results[search_results['driverId']==x]['forename'].iloc[0]} {search_results[search_results['driverId']==x]['surname'].iloc[0]}")
            else:
                selected_driver_id = search_results['driverId'].iloc[0]
            
            # Get driver details
            driver_info = search_results[search_results['driverId'] == selected_driver_id].iloc[0]
            driver_name = f"{driver_info['forename']} {driver_info['surname']}"
            
            # Display driver statistics
            st.subheader(f"{driver_name} Statistics")
            
            # Filter results for this driver
            driver_results = merged_df[merged_df['driverId'] == selected_driver_id]
            
            # Calculate statistics
            total_races = len(driver_results)
            wins = len(driver_results[driver_results['position'] == 1])
            podiums = len(driver_results[driver_results['position'].isin([1, 2, 3])])
            poles = len(driver_results[driver_results['grid'] == 1])
            total_points = driver_results['points'].sum()
            
            # Display stats in columns
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Races", total_races)
            col2.metric("Wins", wins)
            col3.metric("Podiums", podiums)
            col4.metric("Poles", poles)
            col5.metric("Total Points", f"{total_points:.1f}")
            
            # Create position chart
            st.subheader(f"{driver_name}'s Race Positions")
            
            # Prepare data for position chart
            position_data = driver_results.sort_values('date')[['name', 'date', 'position', 'grid', 'points']]
            
            # Create chart
            fig = px.line(
                position_data, 
                x='date', 
                y='position',
                markers=True,
                title=f"{driver_name}'s Race Positions",
                labels={'date': 'Race Date', 'position': 'Position'}
            )
            
            # Invert y-axis for race positions (1 is best)
            fig.update_yaxes(autorange="reversed")
            
            # Add grid position as another line
            fig.add_trace(
                go.Scatter(
                    x=position_data['date'],
                    y=position_data['grid'],
                    mode='lines+markers',
                    name='Grid Position',
                    line=dict(dash='dash')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show race results table
            st.subheader(f"{driver_name}'s Race Results")
            st.dataframe(
                position_data.rename(columns={
                    'name': 'Race',
                    'date': 'Date',
                    'position': 'Finish Position',
                    'grid': 'Grid Position',
                    'points': 'Points'
                }),
                use_container_width=True
            )
            
            # Show driver info
            with st.expander("Driver Details"):
                st.write(f"**Driver ID:** {driver_info['driverId']}")
                st.write(f"**Code:** {driver_info['code']}")
                st.write(f"**Number:** {driver_info['number']}")
                st.write(f"**Date of Birth:** {driver_info['dob']}")
                st.write(f"**Nationality:** {driver_info['nationality']}")
                st.write(f"**Wikipedia:** [{driver_info['url']}]({driver_info['url']})")
                
            # Save driver details to session state
            if st.button("Save Driver Details for Use in Other Pages"):
                driver_data = {
                    'driver_id': int(driver_info['driverId']),
                    'name': driver_name,
                    'code': driver_info['code'],
                    'number': int(driver_info['number']) if not pd.isna(driver_info['number']) else None,
                    'nationality': driver_info['nationality'],
                    'results': position_data.to_dict('records')
                }
                
                SessionStateManager.save_analysis_result(
                    f"Driver_{driver_name.replace(' ', '_')}",
                    driver_data,
                    {"type": "driver_details"}
                )
                st.success(f"Driver details for {driver_name} saved successfully!")
        else:
            st.warning(f"No results found for '{search_term}'")
else:
    # Check if there are any saved driver searches
    saved_searches = [name for name in SessionStateManager.get_analysis_names() 
                     if 'type' in SessionStateManager.analysis_results[name]['metadata'] 
                     and SessionStateManager.analysis_results[name]['metadata']['type'] == 'driver_search']
    
    if saved_searches:
        st.subheader("Previously Searched Drivers")
        selected_search = st.selectbox("Select a previous search", saved_searches)
        
        if st.button("Load Search Results"):
            search_results = SessionStateManager.get_analysis_result(selected_search)
            st.success(f"Loaded search results for '{SessionStateManager.analysis_results[selected_search]['metadata']['search_term']}'")
            st.dataframe(search_results[['driverId', 'code', 'number', 'forename', 'surname', 'nationality', 'dob']], use_container_width=True)
    else:
        st.info("Please select a dataset or upload a CSV or Excel file with Formula 1 data to search for drivers.")
        
        # Example data structure
        st.subheader("Example Data Format")
        example_df = pd.DataFrame({
            'driverId': [1, 20, 830],
            'code': ['HAM', 'VET', 'VER'],
            'number': [44, 5, 33],
            'forename': ['Lewis', 'Sebastian', 'Max'],
            'surname': ['Hamilton', 'Vettel', 'Verstappen'],
            'nationality': ['British', 'German', 'Dutch'],
            'dob': ['1985-01-07', '1987-07-03', '1997-09-30']
        })
        st.dataframe(example_df)