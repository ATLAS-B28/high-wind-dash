# Highwind Formula 1 Dashboard

A multi-page Streamlit application for analyzing Formula 1 data with data persistence across pages.

## Features

- **Data Persistence**: Upload data once and use it across all pages
- **Data Explorer**: Upload and explore F1 data with dynamic filters
- **Chart Visualization**: Create interactive charts from your data
- **Data Analysis**: Perform statistical analysis on F1 data
- **Driver Search**: Search for drivers and view their statistics
- **Lap Time Analysis**: Analyze lap times and performance
- **Race Analysis**: Analyze race results and performance data
- **Championship Standings**: View and analyze championship standings
- **Data Manager**: Centralized management of all saved data

## Data Persistence Features

- Upload datasets once and access them from any page
- Save filtered datasets for later use
- Save chart configurations
- Save driver search results and details
- Export and import all your data
- Manage user preferences across the application

## Getting Started

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
4. Upload your Formula 1 data or use the included sample data

## Data Format

This dashboard works with Formula 1 data in CSV or Excel format. The data should include:

- Driver information (name, number, team)
- Race results (position, points)
- Optional: Lap times, qualifying results, etc.

## Pages

1. **Home**: Welcome page with quick links and data management
2. **Data Explorer**: Upload and explore data with dynamic filters
3. **Chart Visualization**: Create interactive charts from your data
4. **Data Analysis**: Perform statistical analysis on your data
5. **Driver Search**: Search for drivers and view their statistics
6. **Lap Time Analysis**: Analyze lap times and performance
7. **Race Analysis**: Analyze race results and performance data
8. **Championship Standings**: View and analyze championship standings
9. **Data Manager**: Manage all your saved data across the application

## Technologies Used

- Streamlit
- Pandas
- Altair
- Plotly
- NumPy