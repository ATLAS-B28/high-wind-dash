import streamlit as st
import pandas as pd
from utils import load_data, create_filtered_dataframe, create_chart, get_column_types

st.set_page_config(
    page_title="Chart Visualization",
    page_icon="📈",
    layout="wide"
)

st.title("Chart Visualization")
st.write("Create interactive charts from your data")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
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
                ["Scatter Plot", "Bar Chart", "Line Chart", "Histogram", "Box Plot", "Heatmap"]
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
            st.altair_chart(chart, use_container_width=True)
        
        # Show data sample
        with st.expander("View Data Sample"):
            st.dataframe(filtered_df.head(10))
else:
    st.info("Please upload a CSV or Excel file to create charts.")
    
    # Example visualization
    st.subheader("Example Chart")
    example_df = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D', 'E'],
        'Value': [10, 25, 15, 30, 22]
    })
    
    example_chart = create_chart(example_df, "Bar Chart", "Category", "Value")
    st.altair_chart(example_chart, use_container_width=True)