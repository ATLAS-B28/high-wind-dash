import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data, create_filtered_dataframe, get_column_types

st.set_page_config(
    page_title="Data Analysis",
    page_icon="🔍",
    layout="wide"
)

st.title("Data Analysis")
st.write("Analyze your data with statistical methods")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
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
                            file_name=f"grouped_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv"
                        )
                        
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
            else:
                st.warning("Need at least one datetime column for time series analysis.")
else:
    st.info("Please upload a CSV or Excel file to analyze data.")