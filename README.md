# Highwind Dashboard

A multi-page Streamlit dashboard application for data visualization and analysis.

## Features

- Upload and process CSV and Excel files
- Dynamic data tables with filtering capabilities
- Interactive charts and visualizations (scatter plots, bar charts, line charts, etc.)
- Data analysis tools (correlation, grouping, time series)
- Multi-page interface for organized analysis

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Activate the virtual environment (if not already activated)
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Pages

1. **Home** - Welcome page with application overview
2. **Data Explorer** - Upload and explore data with dynamic filters
3. **Chart Visualization** - Create interactive charts from your data
4. **Data Analysis** - Perform statistical analysis on your data

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Altair
- Plotly
- NumPy
- OpenPyXL

## License

MIT