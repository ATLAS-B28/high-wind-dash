import streamlit as st
import pandas as pd
import os
from utils import load_data, create_filtered_dataframe, SessionStateManager

# Initialize session state
SessionStateManager.initialize_session_state()

st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("Data Explorer")
st.write("Explore your data with dynamic filters")

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
    
    return dfs

# Dataset selection or upload
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
        f1_dfs = load_formula_archive_data()
        
        if f1_dfs:
            # Let user select which dataset to explore
            selected_f1_dataset = st.selectbox(
                "Select Formula 1 dataset to explore",
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
    selected_dataset = SessionStateManager.create_dataset_selector("Select a dataset to explore")
    
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
    # Display basic info
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())
    
    # Create filtered dataframe
    filtered_df = create_filtered_dataframe(df)
    
    # Option to save filtered data as a new dataset
    if st.checkbox("Save filtered data as a new dataset"):
        filtered_name = st.text_input("Filtered Dataset Name", value=f"filtered_{selected_dataset if 'selected_dataset' in locals() else 'data'}")
        
        if st.button("Save Filtered Dataset"):
            SessionStateManager.save_dataset(
                filtered_name, 
                filtered_df, 
                {'source': 'filtered data', 'parent': selected_dataset if 'selected_dataset' in locals() else 'uploaded file'}
            )
            st.success(f"Filtered dataset '{filtered_name}' saved successfully!")
    
    # Display data statistics
    st.subheader("Data Statistics")
    tab1, tab2, tab3 = st.tabs(["Summary", "Head", "Tail"])
    
    with tab1:
        st.write(filtered_df.describe())
    
    with tab2:
        st.write(filtered_df.head(10))
    
    with tab3:
        st.write(filtered_df.tail(10))
    
    # Display full filtered dataframe with pagination
    st.subheader("Filtered Data")
    
    # Pagination controls
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100])
    total_pages = (len(filtered_df) - 1) // page_size + 1
    
    if total_pages > 1:
        page_number = st.slider("Page", 1, total_pages, 1)
        start_idx = (page_number - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        st.write(f"Showing rows {start_idx + 1} to {end_idx} of {len(filtered_df)}")
        st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
    else:
        st.dataframe(filtered_df, use_container_width=True)
    
    # Download filtered data
    st.download_button(
        label="Download filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name=f"filtered_data.csv",
        mime="text/csv"
    )
    
    # Save filter configuration
    if st.checkbox("Save this filter configuration"):
        filter_name = st.text_input("Filter Configuration Name", value="My Filter")
        
        if st.button("Save Filter"):
            # Get current filter state from sidebar
            # This is a simplified version - in a real app you'd capture the actual filter values
            filter_config = {
                "dataset": selected_dataset if 'selected_dataset' in locals() else "uploaded_file",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            SessionStateManager.save_filter(filter_name, filter_config)
            st.success(f"Filter configuration '{filter_name}' saved!")
else:
    st.info("Please select an existing dataset or upload a CSV or Excel file to get started.")
    
    # Example data
    st.subheader("Example Data Format")
    example_df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=5),
        'Category': ['A', 'B', 'A', 'C', 'B'],
        'Value': [10, 25, 15, 30, 22],
        'Count': [100, 200, 150, 300, 220]
    })
    st.dataframe(example_df)