import streamlit as st
import pandas as pd
from utils import load_data, create_filtered_dataframe

st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("Data Explorer")
st.write("Upload your data file and explore it with dynamic filters")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        # Display basic info
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())
        
        # Create filtered dataframe
        filtered_df = create_filtered_dataframe(df)
        
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
            file_name=f"filtered_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a CSV or Excel file to get started.")
    
    # Example data
    st.subheader("Example Data Format")
    example_df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=5),
        'Category': ['A', 'B', 'A', 'C', 'B'],
        'Value': [10, 25, 15, 30, 22],
        'Count': [100, 200, 150, 300, 220]
    })
    st.dataframe(example_df)