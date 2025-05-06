import streamlit as st

st.set_page_config(
    page_title="Highwind Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Highwind Dashboard")
st.markdown("""
# Welcome to Highwind Dashboard
Upload your CSV or Excel files to visualize and analyze your data.

## Features
- Dynamic data tables with filtering capabilities
- Interactive charts and visualizations
- Multi-page interface for organized analysis

Get started by navigating to the pages in the sidebar.
""")

st.sidebar.success("Select a page above.")