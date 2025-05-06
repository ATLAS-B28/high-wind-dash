import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
from typing import List, Dict, Any, Tuple, Optional

def load_data(uploaded_file) -> pd.DataFrame:
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
                return None
                
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    
    if df is None:
        return {}
        
    column_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': []
    }
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types['numeric'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types['datetime'].append(col)
        elif df[col].nunique() < 0.5 * len(df) and df[col].nunique() < 100:
            column_types['categorical'].append(col)
        else:
            column_types['text'].append(col)
            
    return column_types

def create_filtered_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    if df is None or df.empty:
        return None
        
    st.sidebar.header("Filters")
    
    filtered_df = df.copy()
    column_types = get_column_types(df)
    
    # Filter for categorical columns
    for col in column_types['categorical']:
        unique_values = sorted(df[col].unique())
        selected_values = st.sidebar.multiselect(
            f"Select {col}",
            unique_values,
            default=unique_values
        )
        if selected_values:
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
    # Filter for numeric columns
    for col in column_types['numeric']:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        step = (max_val - min_val) / 100
        selected_range = st.sidebar.slider(
            f"Select range for {col}",
            min_val, max_val, (min_val, max_val),
            step=step
        )
        filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & 
                                 (filtered_df[col] <= selected_range[1])]
    
    # Filter for datetime columns
    for col in column_types['datetime']:
        try:
            min_date = df[col].min().date()
            max_date = df[col].max().date()
            selected_dates = st.sidebar.date_input(
                f"Select date range for {col}",
                value=(min_date, max_date)
            )
            
            if len(selected_dates) == 2:
                start_date, end_date = selected_dates
                filtered_df = filtered_df[(filtered_df[col].dt.date >= start_date) & 
                                         (filtered_df[col].dt.date <= end_date)]
        except:
            st.sidebar.write(f"Could not create date filter for {col}")
    
    return filtered_df

def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str, 
                color_col: Optional[str] = None, size_col: Optional[str] = None) -> Any:
   
    if df is None or df.empty:
        return None
    
    if chart_type == "Scatter Plot":
        if color_col and size_col:
            chart = alt.Chart(df).mark_circle().encode(
                x=x_col,
                y=y_col,
                color=color_col,
                size=size_col,
                tooltip=list(df.columns)
            ).interactive()
        elif color_col:
            chart = alt.Chart(df).mark_circle().encode(
                x=x_col,
                y=y_col,
                color=color_col,
                tooltip=list(df.columns)
            ).interactive()
        else:
            chart = alt.Chart(df).mark_circle().encode(
                x=x_col,
                y=y_col,
                tooltip=list(df.columns)
            ).interactive()
            
    elif chart_type == "Bar Chart":
        chart = alt.Chart(df).mark_bar().encode(
            x=x_col,
            y=y_col,
            color=color_col if color_col else alt.value('steelblue'),
            tooltip=list(df.columns)
        ).interactive()
        
    elif chart_type == "Line Chart":
        chart = alt.Chart(df).mark_line().encode(
            x=x_col,
            y=y_col,
            color=color_col if color_col else alt.value('steelblue'),
            tooltip=list(df.columns)
        ).interactive()
        
    elif chart_type == "Heatmap":
        # Using Plotly for heatmap as it's more flexible
        fig = px.density_heatmap(
            df, 
            x=x_col, 
            y=y_col,
            color_continuous_scale="Viridis"
        )
        return fig
        
    elif chart_type == "Histogram":
        chart = alt.Chart(df).mark_bar().encode(
            alt.X(f"{x_col}:Q", bin=True),
            y='count()',
            tooltip=['count()']
        ).interactive()
        
    elif chart_type == "Box Plot":
        chart = alt.Chart(df).mark_boxplot().encode(
            x=x_col,
            y=y_col,
            color=color_col if color_col else alt.value('steelblue'),
        ).interactive()
        
    else:
        st.error(f"Unsupported chart type: {chart_type}")
        return None
        
    return chart