# PCA Visualization App

A simple app for creating PCA visualizations from CSV data, with two implementations: Streamlit and Dash.

## Features

- Upload a CSV file
- Select columns for PCA analysis
- Visualize the first two principal components
- Color code points based on a selected column
- Display percentage variance explained by the first two components
- Customize plot title, axis labels, color palette, and point opacity

## Implementations

### Streamlit Version (easy_pca_vis.py)

Best for smaller datasets.

Run with: `streamlit run easy_pca_vis.py`

### Dash Version (pca_alternative.py)

Better suited for larger datasets, using Dash and Plotly for improved performance.

Run with: `python pca_alternative.py`

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the app:
   ```
   streamlit run easy_pca_vis.py
   ```

2. Open the provided URL in your browser (usually http://localhost:8501)

3. Upload your CSV file

4. Select the columns to include in the PCA

5. Choose a column for coloring the points

6. Customize the plot title and labels if desired

7. View the PCA plot and variance explained

## Requirements

- Python 3.7+
- The dependencies listed in requirements.txt

## Troubleshooting

- Ensure your CSV file has numeric columns for PCA
- Handle missing values appropriately (the app drops rows with NaN in selected PCA columns)
- For coloring, both categorical and numeric columns are supported
