import streamlit as st
import pandas as pd
import os
from utils import load_data, create_filtered_dataframe, create_chart, get_column_types, SessionStateManager

# Initialize session state
SessionStateManager.initialize_session_state()

st.set_page_config(
    page_title="Chart Visualization",
    page_icon="📈",
    layout="wide"
)

st.title("Chart Visualization")
st.write("Create interactive charts from your data")

# Function to load Formula Archive data
def load_formula_archive_data():
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
    
    return dfs

# Dataset selection or upload
data_source = st.radio(
    "Choose data source",
    ["Formula Archive data", "Use existing dataset", "Upload new file"],
    horizontal=True,
    index=0  # Default to Formula Archive data
    horizontal=True,
    index=0 if st.session_state.datasets else 1
)

df = None

if data_source == "Formula Archive data":
    try:
        # Load Formula Archive data
        f1_dfs = load_formula_archive_data()
        
        if f1_dfs:
            # Let user select which dataset to visualize
            selected_f1_dataset = st.selectbox(
                "Select Formula 1 dataset to visualize",
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

if data_source == "Use existing dataset" and df is None:
    # Dataset selector
    selected_dataset = SessionStateManager.create_dataset_selector("Select a dataset to visualize")
    
    if selected_dataset:
        df = SessionStateManager.get_dataset(selected_dataset)
        st.success(f"Loaded dataset: {selected_dataset}")

elif data_source == "Upload new file" and df is None:
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
    
    # Get column types for chart selection
    column_types = get_column_types(filtered_df)
    
    # Chart configuration
    st.subheader("Chart Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Scatter Plot", "Bar Chart", "Line Chart", "Histogram", "Box Plot", "Heatmap", "Race Position Chart", "Lap Time Comparison"]
        )
        
        # Select columns based on chart type
        all_columns = list(filtered_df.columns)
        
        if chart_type == "Histogram":
            x_col = st.selectbox("Select column for histogram", column_types['numeric'])
            y_col = None
        else:
            x_options = all_columns
            y_options = column_types['numeric']
            
            x_col = st.selectbox("X-axis", x_options)
            y_col = st.selectbox("Y-axis", y_options)
    
    with col2:
        # Optional encodings
        color_col = st.selectbox("Color by (optional)", ["None"] + column_types['categorical'], index=0)
        color_col = None if color_col == "None" else color_col
        
        if chart_type == "Scatter Plot":
            size_col = st.selectbox("Size by (optional)", ["None"] + column_types['numeric'], index=0)
            size_col = None if size_col == "None" else size_col
        else:
            size_col = None
    
    # Create and display chart
    st.subheader("Visualization")
    
    if chart_type == "Histogram":
        chart = create_chart(filtered_df, chart_type, x_col, None)
    else:
        chart = create_chart(filtered_df, chart_type, x_col, y_col, color_col, size_col)
    
    if chart is not None:
        # Handle both Altair and Plotly charts
        if chart_type in ["Heatmap", "Race Position Chart", "Lap Time Comparison"]:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.altair_chart(chart, use_container_width=True)
        
        # Option to save chart configuration
        with st.expander("Save this chart configuration"):
            chart_name = st.text_input("Chart Name", value=f"{chart_type} of {x_col} vs {y_col}")
            
            if st.button("Save Chart Configuration"):
                chart_config = {
                    "chart_type": chart_type,
                    "x_col": x_col,
                    "y_col": y_col,
                    "color_col": color_col,
                    "size_col": size_col,
                    "dataset": selected_dataset if 'selected_dataset' in locals() else "uploaded_data"
                }
                
                SessionStateManager.save_analysis_result(
                    chart_name,
                    chart_config,
                    {"type": "chart_config"}
                )
                st.success(f"Chart configuration '{chart_name}' saved!")
    
    # Show data sample
    with st.expander("View Data Sample"):
        st.dataframe(filtered_df.head(10))
    
    # Previously saved chart configurations
    saved_charts = [name for name in SessionStateManager.get_analysis_names() 
                   if 'type' in SessionStateManager.analysis_results[name]['metadata'] 
                   and SessionStateManager.analysis_results[name]['metadata']['type'] == 'chart_config']
    
    if saved_charts:
        st.subheader("Saved Chart Configurations")
        selected_saved_chart = st.selectbox("Select a saved chart", saved_charts)
        
        if st.button("Load Chart Configuration"):
            chart_config = SessionStateManager.get_analysis_result(selected_saved_chart)
            st.success(f"Loaded chart configuration: {selected_saved_chart}")
            st.write(f"Chart type: {chart_config['chart_type']}")
            st.write(f"X-axis: {chart_config['x_col']}")
            st.write(f"Y-axis: {chart_config['y_col']}")
            if chart_config['color_col']:
                st.write(f"Color by: {chart_config['color_col']}")
            if chart_config['size_col']:
                st.write(f"Size by: {chart_config['size_col']}")
else:
    st.info("Please select an existing dataset or upload a CSV or Excel file to create charts.")
    
    # Example visualization
    st.subheader("Example Chart")
    example_df = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D', 'E'],
        'Value': [10, 25, 15, 30, 22]
    })
    
    example_chart = create_chart(example_df, "Bar Chart", "Category", "Value")
    st.altair_chart(example_chart, use_container_width=True)