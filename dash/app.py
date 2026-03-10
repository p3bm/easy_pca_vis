import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import base64
import io

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("PCA Visualization App"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or Click to select a CSV file.']),
        multiple=False
    ),
    dcc.Store(id='stored-data'),
    html.Div(id='data-preview'),
    dcc.Checklist(
        id='use-all-cols',
        options=[{'label': 'Use all numeric columns for PCA', 'value': 'all'}],
        value=['all']
    ),
    dcc.Dropdown(id='pca-cols', multi=True, placeholder="Select columns for PCA"),
    dcc.Dropdown(id='color-col', placeholder="Select column for color coding"),
    dcc.Input(id='title', type='text', placeholder='Plot Title', value='PCA Plot'),
    dcc.Input(id='xlabel', type='text', placeholder='X Label', value='PC1'),
    dcc.Input(id='ylabel', type='text', placeholder='Y Label', value='PC2'),
    dcc.Dropdown(
        id='palette',
        options=['viridis', 'plasma', 'coolwarm', 'Set1', 'tab10'],
        value='viridis'
    ),
    html.Label("Point Opacity"),
    dcc.Slider(id='opacity', min=0.1, max=1.0, step=0.1, value=0.8),
    html.Button('Generate PCA Plot', id='generate-btn'),
    html.Div(id='variance-info'),
    dcc.Graph(id='pca-plot')
])

@app.callback(
    Output('stored-data', 'data'),
    Output('data-preview', 'children'),
    Output('pca-cols', 'options'),
    Output('pca-cols', 'value'),
    Output('color-col', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('use-all-cols', 'value')
)
def update_data(contents, filename, use_all):
    if contents is None:
        return None, "No data uploaded", [], [], []
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    # Preview
    preview = html.Table([
        html.Tr([html.Th(col) for col in df.columns])
    ] + [
        html.Tr([html.Td(str(df.iloc[i][col])) for col in df.columns]) for i in range(min(5, len(df)))
    ])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    pca_options = [{'label': col, 'value': col} for col in numeric_cols]
    pca_value = numeric_cols if 'all' in use_all else []
    
    color_options = [{'label': col, 'value': col} for col in df.columns]
    
    return df.to_json(date_format='iso', orient='split'), preview, pca_options, pca_value, color_options

@app.callback(
    Output('pca-cols', 'value'),
    Input('use-all-cols', 'value'),
    State('stored-data', 'data')
)
def update_pca_cols(use_all, data):
    if data and 'all' in use_all:
        df = pd.read_json(io.StringIO(data), orient='split')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    return []

@app.callback(
    Output('pca-plot', 'figure'),
    Output('variance-info', 'children'),
    Input('generate-btn', 'n_clicks'),
    State('stored-data', 'data'),
    State('pca-cols', 'value'),
    State('color-col', 'value'),
    State('title', 'value'),
    State('xlabel', 'value'),
    State('ylabel', 'value'),
    State('palette', 'value'),
    State('opacity', 'value')
)
def generate_plot(n_clicks, data, pca_cols, color_col, title, xlabel, ylabel, palette, opacity):
    if n_clicks is None or not data or not pca_cols or not color_col:
        return {}, ""
    
    df = pd.read_json(io.StringIO(data), orient='split')
    
    # Clean data
    df_pca = df.dropna(subset=pca_cols).drop_duplicates()
    if df_pca.empty:
        return {}, "No valid data after cleaning."
    
    # PCA
    X = df_pca[pca_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    
    pc_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    pc_df[color_col] = df_pca[color_col].values
    
    # Variance
    var_explained = pca.explained_variance_ratio_ * 100
    variance_text = f"Variance Explained: PC1: {var_explained[0]:.2f}%, PC2: {var_explained[1]:.2f}%"
    
    # Plot
    fig = px.scatter(
        pc_df, x='PC1', y='PC2', color=color_col,
        color_continuous_scale=palette if pc_df[color_col].dtype in ['int64', 'float64'] else None,
        color_discrete_sequence=px.colors.qualitative.__dict__[palette] if pc_df[color_col].dtype == 'object' else None,
        opacity=opacity,
        title=title,
        labels={'PC1': xlabel, 'PC2': ylabel}
    )
    
    return fig, variance_text

if __name__ == '__main__':
    app.run_server(debug=True)
