import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update

from PIL import Image
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=4, random_state=42)
plot_df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "cluster": y})


# Loading dataset
ind_df = pd.read_csv('data/indigenous_collection_processed.csv', index_col='id')

sampled_ind_df = ind_df[~ind_df['image_path'].isnull()].sample(len(plot_df))
plot_df['image_path'] = sampled_ind_df['image_path'].values
plot_df['image_path'] = plot_df['image_path'].apply(lambda path: 'data/br_images/'+path.split('/')[-1].split('.')[0]+'.png')
plot_df['povo'] = sampled_ind_df['povo'].values

# Dash app setup
app = Dash(__name__)

# Updating layout for any given figure
def fig_update_layout(fig, x_range=(-12,12), y_range=(-12,12)):
    # Customizing layout
    fig.update_layout(
        # Adjusting style of the graph
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        
        font=dict(family="Roboto, sans-serif", size=16, color="black"),
        
        title_x=0.5,
        margin=dict(l=0, r=0, t=10, b=0),
        
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
        coloraxis_showscale=False,

        # Mouse default configuration (panning instead of zooming)
        dragmode='pan'
    )

    # Hide axes' labels and ticks
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=x_range
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=y_range
    )

# Create empty figure for initialization
def empty_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig_update_layout(fig)
    return fig

# Create scatter plot with markers
def plot_with_markers(df, x_range, y_range):
    # Creating Plotly figure
    fig = px.scatter(df, x='x', y='y', color='cluster')

    # Deactivating default hovering
    fig.update_traces(hoverinfo='none', hovertemplate=None, marker=dict(size=10))

    fig_update_layout(fig, x_range, y_range)
    return fig

# Create scatter plot with the images themselves
def plot_with_images(df, x_range, y_range):
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_layout_image(
            dict(
                source=Image.open(row['image_path']),
                x=row['x'],
                y=row['y'],
                xref="x",
                yref="y",
                sizex=1.0,
                sizey=1.0,
                xanchor="center",
                yanchor="middle",
            )
        )
    
    fig_update_layout(fig, x_range, y_range)
    return fig

# Dash graph configurations
config = {
    'scrollZoom': True,
    'displayModeBar': False,
    'displaylogo': False
}

# Dash layout
app.layout = html.Div(
    className='base-background',
    children=[
        html.Div(
            className='tool-container',
            children=[
                html.H1("Agrupamentos do Acervo", className='graph-title'),
                html.Div(
                    className='sidebar-graph',
                    children=[
                        html.Div(
                            className='sidebar',
                            children=[
                                dcc.RadioItems(
                                    id='toggle-view',
                                    options=[
                                        {'label': 'Pontos', 'value': 'markers'},
                                        {'label': 'Imagens', 'value': 'images'}
                                    ],
                                    value='markers'
                                ),
                                html.P('MAIS BOTÕES!')
                            ]
                        ),
                        html.Div(
                            className='graph',
                            children=[
                                dcc.Graph(id='cluster-plot', config=config, figure=empty_figure(), clear_on_unhover=True),
                                dcc.Tooltip(id='graph-tooltip')
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Callback for hovering
@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("cluster-plot", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # Extracting plotly dash information
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    
    # Acessing the dataframe to get the data we actually want to display
    df_row = plot_df.iloc[num]
    img_src = df_row['image_path']
    povo = df_row['povo']

    children = [
        html.Div([
            html.Img(src=Image.open(img_src), style={"width": "100%"}),
            html.P(f'Povo {povo}', className='hover-box')
        ], style={'width': '100%'})
    ]

    return True, bbox, children

# Callback for collapsing points that are close together and to switch between points and images (because of duplicate ['cluster-plot', 'figure'] output)
@app.callback(
    Output('cluster-plot', 'figure'),
    Input('toggle-view', 'value'),
    Input('cluster-plot', 'relayoutData'),
)
def update_scatter_plot(view_type, relayout_data):
    # Handling (potential) constant trigerring and app crashing
    if relayout_data is None:
        return no_update
    
    # Default zoom range
    x_range = (-12, 12)
    y_range = (-12, 12)

    if relayout_data and 'xaxis.range[0]' in relayout_data:
        x_range = (relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]'])
        y_range = (relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]'])

    # Calculating the visible "scale" to dynamically adjust sensitivity
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]

    # Dynamic epsilon based on zoom level
    eps = 0.01 * min(x_span, y_span)

    # Computing collapses
    coords = plot_df[['x', 'y']].to_numpy()
    db = DBSCAN(eps=eps, min_samples=2).fit(coords)
    labels = db.labels_

    # Splitting clusters (inliers) and outliers for collapsing
    collapse_df = pd.DataFrame(coords, columns=["x", "y"])
    collapse_df["cluster"] = labels
    
    outliers = collapse_df[collapse_df['cluster'] == -1]
    outliers['cluster'] = plot_df[collapse_df['cluster'] == -1]['cluster'].values
    outliers['image_path'] = plot_df[collapse_df['cluster'] == -1]['image_path'].values

    visible_outliers = outliers[
        (outliers['x'] >= x_range[0]) & (outliers['x'] <= x_range[1]) &
        (outliers['y'] >= y_range[0]) & (outliers['y'] <= y_range[1])
    ]

    inliers = collapse_df[collapse_df['cluster'] != -1]

    # Replotting outliers
    if view_type == 'markers':
        fig = plot_with_markers(outliers, x_range, y_range)
    else:
        fig = plot_with_images(visible_outliers, x_range, y_range)
    
    # Plotting collapsed points
    centroids_df = inliers.groupby('cluster').agg({'x': 'mean', 'y': 'mean'})
    centroids_df['count'] = inliers.groupby('cluster').size().values
    centroids_df['marker_size'] = centroids_df['count'].apply(lambda c: min(150, max(25, 15*np.log(c))))

    plot_df['image_path'] = plot_df['image_path'].apply(lambda path: 'data/br_images/'+path.split('/')[-1].split('.')[0]+'.png')

    fig.add_trace(go.Scatter(
        x=centroids_df['x'],
        y=centroids_df['y'],
        mode='markers+text',
        marker=dict(color='#062a57', size=centroids_df['marker_size'], symbol='circle'),
        text=centroids_df['count'],
        textposition='middle center',
        textfont=dict(color='#ffffff', size=16)
    ))
    fig.update_traces(hoverinfo='none', hovertemplate=None)

    return fig

# Running app
if __name__ == '__main__':
    app.run_server(debug=True)