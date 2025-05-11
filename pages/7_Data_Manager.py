import streamlit as st
import pandas as pd
import json
from utils import SessionStateManager

# Initialize session state
SessionStateManager.initialize_session_state()

st.set_page_config(
    page_title="Data Manager",
    page_icon="💾",
    layout="wide"
)

st.title("Data Manager")
st.write("Manage all your saved data across the application")

# Create tabs for different types of data
tab1, tab2, tab3, tab4 = st.tabs(["Datasets", "Analysis Results", "Filters", "User Preferences"])

with tab1:
    st.header("Datasets")
    
    if st.session_state.datasets:
        st.write(f"You have {len(st.session_state.datasets)} datasets saved.")
        
        # Create a table of datasets
        dataset_info = []
        for name, dataset in st.session_state.datasets.items():
            dataset_info.append({
                "Name": name,
                "Rows": dataset['metadata']['rows'],
                "Columns": dataset['metadata']['columns'],
                "Uploaded": dataset['metadata'].get('timestamp', 'Unknown'),
                "Source": dataset['metadata'].get('source', 'Unknown')
            })
        
        dataset_df = pd.DataFrame(dataset_info)
        st.dataframe(dataset_df, use_container_width=True)
        
        # Dataset management
        st.subheader("Dataset Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rename dataset
            st.write("Rename Dataset")
            dataset_to_rename = st.selectbox("Select dataset to rename", list(st.session_state.datasets.keys()), key="rename_select")
            new_name = st.text_input("New name", value=dataset_to_rename)
            
            if st.button("Rename"):
                if new_name != dataset_to_rename:
                    if new_name in st.session_state.datasets:
                        st.error(f"Dataset '{new_name}' already exists!")
                    else:
                        # Copy dataset with new name
                        st.session_state.datasets[new_name] = st.session_state.datasets[dataset_to_rename]
                        # Delete old dataset
                        del st.session_state.datasets[dataset_to_rename]
                        # Update selected dataset if needed
                        if st.session_state.selected_dataset == dataset_to_rename:
                            st.session_state.selected_dataset = new_name
                        st.success(f"Dataset renamed from '{dataset_to_rename}' to '{new_name}'")
                        st.rerun()
        
        with col2:
            # Delete dataset
            st.write("Delete Dataset")
            dataset_to_delete = st.selectbox("Select dataset to delete", list(st.session_state.datasets.keys()), key="delete_select")
            
            if st.button("Delete", type="primary", use_container_width=True):
                SessionStateManager.delete_dataset(dataset_to_delete)
                st.success(f"Dataset '{dataset_to_delete}' deleted successfully!")
                st.rerun()
        
        # Dataset preview
        st.subheader("Dataset Preview")
        dataset_to_preview = st.selectbox("Select dataset to preview", list(st.session_state.datasets.keys()), key="preview_select")
        
        if dataset_to_preview:
            df = st.session_state.datasets[dataset_to_preview]['data']
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download option
            st.download_button(
                label="Download dataset as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"{dataset_to_preview}.csv",
                mime="text/csv"
            )
    else:
        st.info("No datasets saved yet. Upload data from the main page or any other page.")

with tab2:
    st.header("Analysis Results")
    
    if st.session_state.analysis_results:
        st.write(f"You have {len(st.session_state.analysis_results)} saved analysis results.")
        
        # Group by type
        result_types = {}
        for name, result in st.session_state.analysis_results.items():
            result_type = result['metadata'].get('type', 'Other')
            if result_type not in result_types:
                result_types[result_type] = []
            result_types[result_type].append(name)
        
        # Create expandable sections for each type
        for result_type, names in result_types.items():
            with st.expander(f"{result_type.title()} ({len(names)})"):
                for name in names:
                    result = st.session_state.analysis_results[name]
                    
                    st.subheader(name)
                    st.write(f"Created: {result['metadata'].get('timestamp', 'Unknown')}")
                    
                    # Display different types of results differently
                    if result_type == 'chart_config':
                        st.json(result['result'])
                    elif result_type == 'driver_search':
                        st.dataframe(result['result'].head(5))
                    elif result_type == 'driver_details':
                        st.json(result['result'])
                    else:
                        st.write(result['result'])
                    
                    # Delete button
                    if st.button(f"Delete {name}", key=f"delete_{name}"):
                        del st.session_state.analysis_results[name]
                        st.success(f"Analysis result '{name}' deleted!")
                        st.rerun()
    else:
        st.info("No analysis results saved yet. Create charts, search for drivers, or perform analysis on other pages.")

with tab3:
    st.header("Saved Filters")
    
    if st.session_state.filters:
        st.write(f"You have {len(st.session_state.filters)} saved filters.")
        
        for name, filter_config in st.session_state.filters.items():
            with st.expander(name):
                st.json(filter_config)
                
                if st.button(f"Delete filter '{name}'", key=f"delete_filter_{name}"):
                    del st.session_state.filters[name]
                    st.success(f"Filter '{name}' deleted!")
                    st.rerun()
    else:
        st.info("No filters saved yet. Create and save filters on the Data Explorer page.")

with tab4:
    st.header("User Preferences")
    
    # Display current preferences
    st.subheader("Current Preferences")
    st.json(st.session_state.user_preferences)
    
    # Edit preferences
    st.subheader("Edit Preferences")
    
    # Theme preference
    theme = st.selectbox(
        "Theme",
        ["light", "dark"],
        index=0 if st.session_state.user_preferences.get('theme') == 'light' else 1
    )
    
    # Default chart type
    default_chart = st.selectbox(
        "Default Chart Type",
        ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Heatmap"],
        index=["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Heatmap"].index(
            st.session_state.user_preferences.get('default_chart', 'Line Chart')
        )
    )
    
    # Save preferences
    if st.button("Save Preferences"):
        SessionStateManager.set_user_preference('theme', theme)
        SessionStateManager.set_user_preference('default_chart', default_chart)
        st.success("Preferences saved successfully!")

# Export/Import section
st.header("Export/Import Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Export All Data")
    st.write("Export all your datasets, analysis results, filters, and preferences to a JSON file.")
    
    if st.button("Export Data"):
        # Create export data structure
        export_data = {
            "datasets": {},
            "analysis_results": st.session_state.analysis_results,
            "filters": st.session_state.filters,
            "user_preferences": st.session_state.user_preferences
        }
        
        # Convert datasets (can't directly serialize pandas DataFrames)
        for name, dataset in st.session_state.datasets.items():
            export_data["datasets"][name] = {
                "metadata": dataset["metadata"],
                "data_csv": dataset["data"].to_csv(index=False)
            }
        
        # Convert to JSON
        export_json = json.dumps(export_data)
        
        # Provide download button
        st.download_button(
            label="Download JSON",
            data=export_json,
            file_name="highwind_data_export.json",
            mime="application/json"
        )

with col2:
    st.subheader("Import Data")
    st.write("Import previously exported data.")
    
    uploaded_file = st.file_uploader("Upload JSON export file", type=["json"])
    
    if uploaded_file is not None:
        try:
            import_data = json.load(uploaded_file)
            
            # Validate structure
            required_keys = ["datasets", "analysis_results", "filters", "user_preferences"]
            if not all(key in import_data for key in required_keys):
                st.error("Invalid export file format!")
            else:
                if st.button("Import Data"):
                    # Import datasets
                    for name, dataset_info in import_data["datasets"].items():
                        df = pd.read_csv(pd.StringIO(dataset_info["data_csv"]))
                        SessionStateManager.save_dataset(name, df, dataset_info["metadata"])
                    
                    # Import analysis results
                    st.session_state.analysis_results = import_data["analysis_results"]
                    
                    # Import filters
                    st.session_state.filters = import_data["filters"]
                    
                    # Import user preferences
                    st.session_state.user_preferences = import_data["user_preferences"]
                    
                    st.success("Data imported successfully!")
                    st.rerun()
        except Exception as e:
            st.error(f"Error importing data: {e}")