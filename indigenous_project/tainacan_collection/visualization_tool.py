import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update
from dash_extensions.enrich import DashProxy, MultiplexerTransform

from PIL import Image
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=4, random_state=42)
plot_df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "cluster": y})


######### DATA LOADING AND PROCESSING #########
# Loading dataset
ind_df = pd.read_csv('data/indigenous_collection_processed.csv', index_col='id')

sampled_ind_df = ind_df[~ind_df['image_path'].isnull()].sample(len(plot_df))
plot_df['image_path'] = sampled_ind_df['image_path'].values
plot_df['image_path'] = plot_df['image_path'].apply(lambda path: 'data/br_images/'+path.split('/')[-1].split('.')[0]+'.png')
plot_df['url'] = sampled_ind_df['url'].values
plot_df['nome_do_item'] = sampled_ind_df['nome_do_item'].values
plot_df['povo'] = sampled_ind_df['povo'].values

# Brazilian states dataset
brazil_states = pd.DataFrame({
    'state': ['Acre', 'Alagoas', 'Amapá', 'Amazonas', 'Bahia', 'Ceará', 'Distrito Federal', 'Espírito Santo', 'Goiás', 'Maranhão', 'Mato Grosso', 'Mato Grosso do Sul', 'Minas Gerais', 'Pará', 'Paraíba', 'Paraná', 'Pernambuco', 'Piauí', 'Rio de Janeiro', 'Rio Grande do Norte', 'Rio Grande do Sul', 'Rondônia', 'Roraima', 'Santa Catarina', 'São Paulo', 'Sergipe', 'Tocantins'],

    'latitude': [-8.77, -9.62, 1.41, -3.07, -12.96, -5.20, -15.83, -19.19, -15.98, -4.96, -12.64, -20.51, -18.10, -3.79, -7.12, -24.89, -8.28, -6.60, -22.91, -5.81, -30.03, -10.90, 2.82, -27.33, -23.55, -10.57, -10.25],
    'longitude': [-70.55, -36.82, -51.77, -61.66, -38.51, -39.53, -47.86, -40.34, -49.86, -44.30, -55.42, -54.54, -44.38, -52.49, -34.83, -51.55, -34.88, -42.28, -43.20, -36.59, -51.22, -62.80, -60.67, -49.44, -46.63, -37.06, -48.25]
})

# Dash app setup. DashPRoxy used for multiple callbacks with the same output, but made the app a bit buggy (multiple triggers of the same callback in a row)
# app = Dash(__name__)
app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()])

########## UTIL FUNCTIONS FOR FIGURE ##########
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
        dragmode='pan',
        hoverdistance = 5
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

# Creating the map of Brazil and plotting markers on states
def brazil_figure():
    fig = px.scatter_mapbox(brazil_states, lat='latitude', lon='longitude', hover_name='state', zoom=3.5, center={'lat': -14.2350, 'lon': -51.9253}, width=1350, height=600)

    fig.update_layout(
        mapbox_style="carto-positron",  # options are 'open-street-map', 'stamen-terrain', 'carto-positron', etc.
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Mapa do Brasil e suas Comunidades Indígenas"
    )

    fig.update_traces(marker=dict(size=20, color='#062a57'))

    return fig

# Create scatter plot with markers
def plot_with_markers(df, x_range, y_range):
    # Creating Plotly figure
    fig = px.scatter(df, x='x', y='y', color='cluster', color_continuous_scale='rainbow', width=1030, height=500)

    # Configuring default hovering
    fig.update_traces(hoverinfo='none', hovertemplate=None, marker=dict(size=20, opacity=1), line=dict(width=0, color='rgb(255, 212, 110)'))

    fig_update_layout(fig, x_range, y_range)
    return fig

# Create scatter plot with the images themselves
def plot_with_images(df, x_range, y_range):
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_layout_image(
            dict(source=Image.open(row['image_path']), x=row['x'], y=row['y'], xref="x", yref="y", sizex=1.0, sizey=1.0, xanchor="center",yanchor="middle")
        )
    
    fig_update_layout(fig, x_range, y_range)
    return fig

# Dash graph configurations
config = {
    'scrollZoom': True,
    'displayModeBar': False,
    'displaylogo': False
}

################# APP LAYOUT #################
# Dash layout
app.layout = html.Div(
    dcc.Tabs([
        dcc.Tab(
            label='Agrupamentos do Acervo',
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
                                        dcc.Location(id='url', refresh=True),
                                        dcc.Store(id='trigger-url'),
                                        dcc.Graph(id='cluster-plot', config=config, figure=empty_figure(), clear_on_unhover=True),
                                        dcc.Tooltip(id='graph-tooltip')
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        dcc.Tab(
            label='Coleção no Brasil',
            children=[
                html.Div(
                    className='map-container',
                    children=[
                            dcc.Graph(id='brazil-map', figure=brazil_figure())
                    ]
                )
            ]
        )
    ]),
    className='base-background'
)

################## CALLBCAKS ##################
# Callback for hovering
@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Output('cluster-plot', 'figure'),
    Input("cluster-plot", "hoverData"),
    State('cluster-plot', 'figure'),
    prevent_initial_call=True
)
def display_hover(hover_data, fig):
    fig = go.Figure(fig)

    if hover_data is None:
        # Resetting markers on hover-out
        old_sizes = list(np.full((len(fig.data[0].x)), 20))
        old_line_widths = list(np.full((len(fig.data[0].x)), 0))

        fig.data[0].marker.size = old_sizes
        fig.data[0].marker.line.width = old_line_widths

        return False, no_update, no_update, fig 

    # Extracting plotly dash information
    pt = hover_data["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    
    # Adding hovering effects
    new_sizes = np.full((len(fig.data[0].x)), 20)
    new_sizes[num] = 35
    new_sizes = list(new_sizes)

    new_line_widths = np.full((len(fig.data[0].x)), 0)
    new_line_widths[num] = 6
    new_line_widths = list(new_line_widths)

    fig.data[0].marker.size = new_sizes
    fig.data[0].marker.line.width = new_line_widths

    # Acessing the dataframe to get the data we actually want to display
    df_row = plot_df.iloc[num]
    img_src = df_row['image_path']
    nome_do_item = df_row['nome_do_item']
    povo = df_row['povo']

    children = [
        html.Div([
            html.Img(src=Image.open(img_src), style={"width": "100%"}),
            html.P(f'{nome_do_item.title()}', className='hover-box'),
            html.P(f'Povo {povo.title()}', className='hover-box')
        ], style={'width': '100px'})
    ]

    return True, bbox, children, fig

# Callback for clicking
@app.callback(
    Output("url", "href"),
    Input("cluster-plot", "clickData"),
)
def open_click(click_data):
    if click_data is None:
        return no_update

    # Extracting plotly dash information
    num = click_data["points"][0]['pointIndex']
    
    # Acessing the dataframe to get the URL we want
    df_row = plot_df.iloc[num]
    url = df_row['url']

    return url

# Callback for collapsing points that are close together and to switch between points and images
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
    eps = 0.03 * min(x_span, y_span)

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
    centroids_df['marker_size'] = centroids_df['count'].apply(lambda c: min(200, max(30, 20*np.log(c))))

    fig.add_trace(go.Scatter(
        x=centroids_df['x'],
        y=centroids_df['y'],
        mode='markers+text',
        marker=dict(color='#062a57', size=centroids_df['marker_size'], symbol='circle'),
        text=centroids_df['count'],
        textposition='middle center',
        textfont=dict(color='#ffffff', size=16),
        hoverinfo='skip',
        hovertemplate=None
    ))

    return fig

# Running app
if __name__ == '__main__':
    app.run_server(debug=True)