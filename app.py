import streamlit as st
import pandas as pd
from utils import SessionStateManager

# Initialize session state for data persistence
SessionStateManager.initialize_session_state()

st.set_page_config(
    page_title="Highwind F1 Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Highwind Formula 1 Dashboard")
st.markdown("""
# Welcome to Highwind Formula 1 Dashboard
Upload your Formula 1 data files to visualize and analyze race results, driver statistics, and more.

## Features
- Dynamic data tables with filtering capabilities
- Interactive charts and visualizations
- Driver search and statistics
- Race analysis and comparisons
- Multi-page interface for organized analysis

Get started by navigating to the pages in the sidebar.
""")

# Add data management section to sidebar
st.sidebar.header("Data Management")

# Display available datasets
if st.session_state.datasets:
    st.sidebar.subheader("Available Datasets")
    for name, dataset_info in st.session_state.datasets.items():
        with st.sidebar.expander(f"{name} ({dataset_info['metadata']['rows']} rows)"):
            st.write(f"Uploaded: {dataset_info['metadata']['timestamp']}")
            st.write(f"Columns: {dataset_info['metadata']['columns']}")
            
            # Option to delete dataset
            if st.button(f"Delete {name}", key=f"delete_{name}"):
                SessionStateManager.delete_dataset(name)
                st.rerun()

# File uploader in sidebar for global data
with st.sidebar.expander("Upload New Dataset", expanded=not bool(st.session_state.datasets)):
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    dataset_name = st.text_input("Dataset Name (optional)", value="")
    
    if uploaded_file is not None:
        try:
            # Get file extension to determine how to read it
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                df = None
                
            if df is not None:
                # Use filename as dataset name if not provided
                if not dataset_name:
                    dataset_name = uploaded_file.name.split('.')[0]
                
                # Save to session state
                SessionStateManager.save_dataset(
                    dataset_name, 
                    df, 
                    {'source': uploaded_file.name}
                )
                
                st.success(f"Dataset '{dataset_name}' loaded successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error loading file: {e}")

st.sidebar.success("Select a page above.")

# Display sample data info
st.subheader("Data Format")
st.write("""
This dashboard works with Formula 1 data in CSV or Excel format. The data should include:

- Driver information (name, number, team)
- Race results (position, points)
- Optional: Lap times, qualifying results, etc.

You can upload your data files here or on any page.
""")

# Display available datasets in the main area
if st.session_state.datasets:
    st.subheader("Your Datasets")
    
    # Create tabs for each dataset
    tabs = st.tabs(list(st.session_state.datasets.keys()))
    
    for i, (name, dataset_info) in enumerate(st.session_state.datasets.items()):
        with tabs[i]:
            df = dataset_info['data']
            st.write(f"Preview of {name} dataset:")
            st.dataframe(df.head(5), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isna().sum().sum())

# Display quick links
st.subheader("Quick Links")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📊 [Data Explorer](/Data_Explorer)")
    st.write("Upload and explore F1 data with dynamic filters")

with col2:
    st.info("🏎️ [Driver Search](/Driver_Search)")
    st.write("Search for drivers and view their statistics")

with col3:
    st.info("🏁 [Race Analysis](/Race_Analysis)")
    st.write("Analyze race results and performance data")