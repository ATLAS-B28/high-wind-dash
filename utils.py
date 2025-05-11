import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import os
import json
from datetime import datetime

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

def search_driver(df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """
    Search for a driver in the DataFrame based on name or number
    
    Args:
        df: DataFrame containing driver data
        search_term: Search term (driver name or number)
        
    Returns:
        Filtered DataFrame with matching drivers
    """
    if df is None or df.empty:
        return None
        
    # Try to convert search term to number for driver number search
    try:
        driver_number = int(search_term)
        number_search = df[df['DriverNumber'] == driver_number]
        if not number_search.empty:
            return number_search
    except (ValueError, KeyError):
        pass
    
    # Search by driver name (case insensitive)
    name_columns = ['DriverName', 'Driver', 'Name', 'FullName', 'Fullname']
    result = pd.DataFrame()
    
    for col in name_columns:
        if col in df.columns:
            matches = df[df[col].str.contains(search_term, case=False, na=False)]
            result = pd.concat([result, matches])
    
    return result.drop_duplicates()

def get_driver_stats(df: pd.DataFrame, driver_name: str) -> Dict[str, Any]:
    """
    Calculate statistics for a specific driver
    
    Args:
        df: DataFrame containing race results
        driver_name: Name of the driver
        
    Returns:
        Dictionary with driver statistics
    """
    if df is None or df.empty:
        return {}
    
    # Find driver name column
    driver_col = None
    for col in ['DriverName', 'Driver', 'Name', 'FullName', 'Fullname']:
        if col in df.columns:
            driver_col = col
            break
    
    if not driver_col:
        return {}
    
    # Filter for the specific driver
    driver_df = df[df[driver_col].str.contains(driver_name, case=False, na=False)]
    
    if driver_df.empty:
        return {}
    
    # Calculate statistics
    stats = {}
    
    # Races participated
    stats['races'] = len(driver_df)
    
    # Positions
    if 'Position' in driver_df.columns:
        position_col = 'Position'
    elif 'FinalPosition' in driver_df.columns:
        position_col = 'FinalPosition'
    elif 'GridPosition' in driver_df.columns:
        position_col = 'GridPosition'
    else:
        position_col = None
    
    if position_col:
        try:
            # Convert position to numeric, handling any non-numeric values
            driver_df[position_col] = pd.to_numeric(driver_df[position_col], errors='coerce')
            
            # Calculate position stats
            stats['wins'] = len(driver_df[driver_df[position_col] == 1])
            stats['podiums'] = len(driver_df[driver_df[position_col] <= 3])
            stats['avg_position'] = driver_df[position_col].mean()
            stats['best_position'] = driver_df[position_col].min()
            stats['worst_position'] = driver_df[position_col].max()
        except:
            pass
    
    # Points
    if 'Points' in driver_df.columns:
        try:
            stats['total_points'] = driver_df['Points'].sum()
            stats['avg_points'] = driver_df['Points'].mean()
        except:
            pass
    
    return stats

def create_driver_position_chart(df: pd.DataFrame, driver_name: str) -> Any:
    """
    Create a chart showing a driver's positions over races
    
    Args:
        df: DataFrame containing race results
        driver_name: Name of the driver
        
    Returns:
        Plotly chart object
    """
    if df is None or df.empty:
        return None
    
    # Find driver name column
    driver_col = None
    for col in ['DriverName', 'Driver', 'Name', 'FullName', 'Fullname']:
        if col in df.columns:
            driver_col = col
            break
    
    if not driver_col:
        return None
    
    # Filter for the specific driver
    driver_df = df[df[driver_col].str.contains(driver_name, case=False, na=False)]
    
    if driver_df.empty:
        return None
    
    # Find position column
    position_col = None
    for col in ['Position', 'FinalPosition', 'GridPosition']:
        if col in driver_df.columns:
            position_col = col
            break
    
    if not position_col:
        return None
    
    # Find race or date column
    race_col = None
    for col in ['Race', 'RaceName', 'Grand Prix', 'GrandPrix', 'Date']:
        if col in driver_df.columns:
            race_col = col
            break
    
    if not race_col:
        return None
    
    # Create chart
    fig = px.line(
        driver_df.sort_values(race_col), 
        x=race_col, 
        y=position_col,
        markers=True,
        title=f"{driver_name}'s Race Positions"
    )
    
    # Invert y-axis for race positions (1 is best)
    fig.update_yaxes(autorange="reversed")
    
    return fig

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
    
    elif chart_type == "Race Position Chart":
        # Special chart for F1 race positions
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col,
            color=color_col,
            markers=True,
            line_shape="linear",
            labels={x_col: x_col, y_col: y_col},
            title=f"Position by {x_col}"
        )
        # Invert y-axis for race positions (1 is best)
        fig.update_yaxes(autorange="reversed")
        return fig
        
    elif chart_type == "Lap Time Comparison":
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            markers=True,
            title=f"Lap Time Comparison"
        )
        return fig
        
    else:
        st.error(f"Unsupported chart type: {chart_type}")
        return None
        
    return chart


# Session State Manager for data persistence across pages
class SessionStateManager:
    """
    A class to manage session state data persistence across all pages of the application.
    This allows data to be shared between different pages without requiring re-uploads.
    """
    
    @staticmethod
    def initialize_session_state():
        """Initialize the session state with default values if they don't exist"""
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
        
        if 'selected_dataset' not in st.session_state:
            st.session_state.selected_dataset = None
            
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
            
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
            
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'theme': 'light',
                'default_chart': 'Line Chart'
            }
    
    @staticmethod
    def save_dataset(name: str, df: pd.DataFrame, metadata: Dict = None) -> None:
        """
        Save a dataset to the session state
        
        Args:
            name: Name of the dataset
            df: DataFrame to save
            metadata: Optional metadata about the dataset
        """
        if metadata is None:
            metadata = {}
            
        # Add timestamp
        metadata['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata['rows'] = df.shape[0]
        metadata['columns'] = df.shape[1]
        
        # Store in session state
        st.session_state.datasets[name] = {
            'data': df,
            'metadata': metadata
        }
        
        # Set as selected dataset if none is selected
        if st.session_state.selected_dataset is None:
            st.session_state.selected_dataset = name
    
    @staticmethod
    def get_dataset(name: str = None) -> Union[pd.DataFrame, None]:
        """
        Get a dataset from the session state
        
        Args:
            name: Name of the dataset to retrieve. If None, returns the currently selected dataset.
            
        Returns:
            DataFrame if found, None otherwise
        """
        if name is None:
            name = st.session_state.selected_dataset
            
        if name is None or name not in st.session_state.datasets:
            return None
            
        return st.session_state.datasets[name]['data']
    
    @staticmethod
    def get_dataset_names() -> List[str]:
        """Get a list of all dataset names in the session state"""
        return list(st.session_state.datasets.keys())
    
    @staticmethod
    def select_dataset(name: str) -> None:
        """Set the currently selected dataset"""
        if name in st.session_state.datasets:
            st.session_state.selected_dataset = name
    
    @staticmethod
    def delete_dataset(name: str) -> None:
        """Delete a dataset from the session state"""
        if name in st.session_state.datasets:
            del st.session_state.datasets[name]
            
            # Reset selected dataset if it was the one deleted
            if st.session_state.selected_dataset == name:
                if st.session_state.datasets:
                    st.session_state.selected_dataset = list(st.session_state.datasets.keys())[0]
                else:
                    st.session_state.selected_dataset = None
    
    @staticmethod
    def save_filter(name: str, filter_config: Dict) -> None:
        """Save a filter configuration to the session state"""
        st.session_state.filters[name] = filter_config
    
    @staticmethod
    def get_filter(name: str) -> Union[Dict, None]:
        """Get a filter configuration from the session state"""
        return st.session_state.filters.get(name)
    
    @staticmethod
    def save_analysis_result(name: str, result: Any, metadata: Dict = None) -> None:
        """
        Save an analysis result to the session state
        
        Args:
            name: Name of the analysis result
            result: The analysis result to save
            metadata: Optional metadata about the result
        """
        if metadata is None:
            metadata = {}
            
        # Add timestamp
        metadata['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        st.session_state.analysis_results[name] = {
            'result': result,
            'metadata': metadata
        }
    
    @staticmethod
    def get_analysis_result(name: str) -> Union[Any, None]:
        """Get an analysis result from the session state"""
        if name in st.session_state.analysis_results:
            return st.session_state.analysis_results[name]['result']
        return None
    
    @staticmethod
    def get_analysis_names() -> List[str]:
        """Get a list of all analysis result names in the session state"""
        return list(st.session_state.analysis_results.keys())
    
    @staticmethod
    def set_user_preference(key: str, value: Any) -> None:
        """Set a user preference"""
        st.session_state.user_preferences[key] = value
    
    @staticmethod
    def get_user_preference(key: str, default: Any = None) -> Any:
        """Get a user preference"""
        return st.session_state.user_preferences.get(key, default)
    
    @staticmethod
    def create_dataset_selector(label: str = "Select Dataset", key: str = "dataset_selector") -> str:
        """
        Create a selectbox for choosing a dataset
        
        Args:
            label: Label for the selectbox
            key: Unique key for the selectbox
            
        Returns:
            Name of the selected dataset
        """
        dataset_names = SessionStateManager.get_dataset_names()
        
        if not dataset_names:
            st.info("No datasets available. Please upload data first.")
            return None
            
        selected = st.selectbox(
            label,
            options=dataset_names,
            index=dataset_names.index(st.session_state.selected_dataset) if st.session_state.selected_dataset in dataset_names else 0,
            key=key
        )
        
        # Update the selected dataset in session state
        SessionStateManager.select_dataset(selected)
        
        return selected