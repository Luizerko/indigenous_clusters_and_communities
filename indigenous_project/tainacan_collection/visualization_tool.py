import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update
from dash_extensions.enrich import DashProxy, MultiplexerTransform
import dash_bootstrap_components as dbc

from PIL import Image
# import base64
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

from utils import brazil_states_dict, norm_factor, empty_figure, brazil_figure, plot_with_markers, plot_with_images, collapse_cluster_points, generate_color_map, get_dropdown_options

from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=20965, centers=10, random_state=42, center_box=(-norm_factor,norm_factor))
plot_df = pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "cluster": y})

######### DATA LOADING AND PROCESSING #########
# Loading dataset
ind_df = pd.read_csv('data/indigenous_collection_processed.csv', index_col='id')

tipo_materia_prima_baseline_df = pd.read_csv('data/clusters/tipo_materia_prima_baseline.csv', index_col='id')

# Creating artificial index to interact with our dataframe
plot_df['ind_index'] = ind_df.index

# Setting point visibility
plot_df['visibility'] = True

# Extracting and processing information that will be used from dataframe
plot_df['image_path'] = ind_df['image_path'].values
plot_df['image_path_br'] = ind_df['image_path'].values
plot_df.loc[plot_df['image_path_br'].notna(), 'image_path_br'] = plot_df.loc[plot_df['image_path_br'].notna(), 'image_path'].apply(lambda path: f"data/br_images/{path.split('/')[-1].split('.')[0]}.png")
plot_df.fillna({'image_path': 'data/placeholder_square.png'}, inplace=True)
plot_df.fillna({'image_path_br': 'data/placeholder_square.png'}, inplace=True)

plot_df['url'] = ind_df['url'].values

plot_df['nome_do_item'] = ind_df['nome_do_item'].values

plot_df['povo'] = ind_df['povo'].values

plot_df['categoria'] = ind_df['categoria'].values

plot_df['ano_de_aquisicao'] = ind_df['ano_de_aquisicao'].values
plot_df.fillna({'ano_de_aquisicao': '----'}, inplace=True)

plot_df['estado_de_origem'] = ind_df['estado_de_origem'].values

plot_df['thumbnail'] = ind_df['thumbnail'].values

plot_df['cluster_names'] = ''
plot_df.set_index('ind_index', inplace=True)
plot_df.loc[tipo_materia_prima_baseline_df.index, 'cluster_names'] = tipo_materia_prima_baseline_df['cluster_names'].values
plot_df.reset_index(inplace=True)

# Creating initial color map for clusters
color_map = generate_color_map(y)
plot_df['color'] = [color_map[label] for label in plot_df['cluster'].values]

# Dash app setup. DashPRoxy used for multiple callbacks with the same output, but made the app a bit buggy (multiple triggers of the same callback in a row)
# app = Dash(__name__)
app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()], external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dash graph configurations
config = {
    'scrollZoom': True,
    'displayModeBar': False,
    'displaylogo': False
}

################# APP LAYOUT #################
# Dash layout
app.layout = html.Div([
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
                                        html.Label("Opções de Exibição", style={'fontWeight': 'bold'}),
                                        dcc.RadioItems(
                                            id='toggle-view',
                                            options=[
                                                {'label': 'Pontos', 'value': 'markers'},
                                                {'label': 'Imagens', 'value': 'images'}
                                            ],
                                            value='markers'
                                        ),

                                        html.Label("Filtragem de Dados", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                                        html.Div(
                                            className='filter-dropdown',
                                            children=[
                                                html.Label("Categoria:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                dcc.Dropdown(
                                                    id='categoria-filter',
                                                    options=get_dropdown_options(ind_df, 'categoria'),
                                                    multi=False,
                                                    placeholder='Filtrar por Categoria',
                                                    value='all',
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className='filter-dropdown',
                                            children=[
                                                html.Label("Povo:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                dcc.Dropdown(
                                                    id='povo-filter',
                                                    options=get_dropdown_options(ind_df, 'povo'),
                                                    multi=False,
                                                    placeholder='Filtrar por Povo',
                                                    value='all',
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className='filter-dropdown',
                                            children=[
                                                html.Label("Estado de Origem:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                dcc.Dropdown(
                                                    id='estado-filter',
                                                    options=get_dropdown_options(ind_df, 'estado_de_origem'),
                                                    multi=False,
                                                    placeholder='Filtrar por Estado de Origem',
                                                    value='all',
                                                ),
                                            ]
                                        ),

                                        html.Label('Opções de Agrupamento', style={'fontWeight': 'bold', 'marginTop': '20px'}),
                                        dcc.Dropdown(
                                            id='single-option-dropdown',
                                            options=[
                                                {'label': 'Teste', 'value': 'cluster_1'},
                                                {'label': 'Tipo de Materia Prima', 'value': 'cluster_2'},
                                            ],
                                            multi=False,
                                            placeholder='Selecione uma opção',
                                            value='cluster_1'
                                        ),
                                        dcc.Store(id='zoom-update')
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
                        dcc.Graph(id='brazil-map', config=config, figure=brazil_figure()),
                    ]
                ),
            ]
        )
    ]),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle('Itens Advindos do', style={'fontWeight': 'bold', 'color': '#062a57'}, id='state-header')),
        dbc.ModalBody(id='state-items'),
    ], id='modal-items', scrollable=True, is_open=False, backdrop=True)
], className='base-background')

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
    df_row = plot_df.iloc[fig.data[0].customdata[num]]
    img_src = df_row['image_path']
    nome_do_item = df_row['nome_do_item']
    povo = df_row['povo']
    try:
        ano_de_aquisicao = int(df_row['ano_de_aquisicao'])
    except:
        ano_de_aquisicao = df_row['ano_de_aquisicao']

    # Hovering box with image only for points with image
    if img_src == 'data/placeholder_square.png':
        children = [
        html.Div(
            className='hover-box',
            children=[
                html.P(f'{nome_do_item.title()}', className='hover-box-text'),
                html.P(f'{povo.title()}, {ano_de_aquisicao}', className='hover-box-text')
            ], style={'width': '160px'})
    ]

    else:
        children = [
            html.Div(
                className='hover-box',
                children=[
                    html.Img(src=Image.open(img_src), className='hover-box-image'),
                    html.P(f'{nome_do_item.title()}', className='hover-box-text'),
                    html.P(f'{povo.title()}, {ano_de_aquisicao}', className='hover-box-text')
                ], style={'width': '160px'})
        ]

    return True, bbox, children, fig

# Callback for clicking
@app.callback(
    Output("url", "href"),
    Input("cluster-plot", "clickData"),
    State('cluster-plot', 'figure')
)
def open_click(click_data, fig):
    if click_data is None:
        return no_update

    fig = go.Figure(fig)

    # Extracting plotly dash information
    num = click_data["points"][0]['pointIndex']
    
    # Acessing the dataframe to get the URL we want
    df_row = plot_df.iloc[fig.data[0].customdata[num]]
    url = df_row['url']

    return url

# Callback for collapsing points that are close together and to switch between points and images
@app.callback(
    Output('cluster-plot', 'figure'),
    Input('toggle-view', 'value'),
    Input('cluster-plot', 'relayoutData'),
    Input('zoom-update', 'data'),
)
def update_scatter_plot(view_type, relayout_data, zoom_update):
    # Handling (potential) constant trigerring and app crashing
    if relayout_data is None:
        return no_update
    
    # Default zoom range
    x_range = (-norm_factor, norm_factor)
    y_range = (-norm_factor, norm_factor)

    if relayout_data and 'xaxis.range[0]' in relayout_data:
        x_range = (relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]'])
        y_range = (relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]'])

    # Getting filtered points only
    filtered_plot_df = plot_df[plot_df['visibility'] == True]

    # Computing collapses
    coords = filtered_plot_df[['x', 'y']].to_numpy()
    labels = collapse_cluster_points(coords, x_range, y_range, threshold=0.05)

    # Splitting clusters (inliers) and outliers for collapsing
    collapse_df = pd.DataFrame(coords, columns=["x", "y"], index=filtered_plot_df.index)
    collapse_df["cluster"] = labels
    
    # Generating colormap to avoid cluster color reassignment with lazy plotting
    color_map = generate_color_map(filtered_plot_df['cluster'].values)
    collapse_df["color"] = [color_map[label] for label in filtered_plot_df['cluster'].values]

    # Getting cluster names for legend
    collapse_df["cluster_names"] = filtered_plot_df['cluster_names']

    outliers = collapse_df[collapse_df['cluster'] == -1]
    outliers = outliers.copy()
    outliers.loc[:, 'cluster'] = filtered_plot_df.loc[collapse_df['cluster'] == -1, 'cluster'].values
    outliers.loc[:, 'image_path_br'] = filtered_plot_df.loc[collapse_df['cluster'] == -1, 'image_path_br'].values

    # Lazy plotting for speed
    visible_outliers = outliers[
        (outliers['x'] >= x_range[0]) & (outliers['x'] <= x_range[1]) &
        (outliers['y'] >= y_range[0]) & (outliers['y'] <= y_range[1])
    ]

    inliers = collapse_df[collapse_df['cluster'] != -1]

    # Replotting outliers
    if len(visible_outliers) > 0:
        if view_type == 'markers':
            fig = plot_with_markers(visible_outliers, x_range, y_range)
        else:
            fig = plot_with_images(visible_outliers, x_range, y_range)
    else:
        fig = empty_figure(x_range, y_range)
    
    # Plotting collapsed points
    centroids_df = inliers.groupby('cluster').agg({'x': 'mean', 'y': 'mean'})
    centroids_df['count'] = inliers.groupby('cluster').size().values
    centroids_df['marker_size'] = centroids_df['count'].apply(lambda c: min(60, max(30, 15*np.log(c))))

    fig.add_trace(go.Scatter(
        x=centroids_df['x'],
        y=centroids_df['y'],
        mode='markers+text',
        marker=dict(color='#062a57', size=centroids_df['marker_size'], symbol='circle'),
        text=centroids_df['count'],
        textposition='middle center',
        textfont=dict(color='#ffffff', size=16),
        hoverinfo='skip',
        hovertemplate=None,
        showlegend=False
    ))

    return fig

# Callback for changing clustering option
@app.callback(
    Output('zoom-update', 'data'),
    Input('single-option-dropdown', 'value')
)
def update_cluster(selected_option):
    plot_df['visibility'] = False

    if selected_option == 'cluster_1':
        plot_df['visibility'] = True
        
        plot_df["x"] = x[:, 0]
        plot_df["y"] = x[:, 1]
        plot_df["cluster"] = y
    
    elif selected_option == 'cluster_2':
        plot_df.set_index('ind_index', inplace=True)
        
        indices = tipo_materia_prima_baseline_df.index
        plot_df.loc[indices, 'visibility'] = True

        plot_df.loc[indices, "x"] = tipo_materia_prima_baseline_df['x'].values
        plot_df.loc[indices, "y"] = tipo_materia_prima_baseline_df['y'].values
        plot_df.loc[indices, "cluster"] = tipo_materia_prima_baseline_df['cluster'].values
        plot_df.loc[indices, 'cluster_names'] = tipo_materia_prima_baseline_df['cluster_names'].values

        plot_df.reset_index(inplace=True)

    return True

# Callback for filtering data
@app.callback(
    Output('zoom-update', 'data'),
    Input('categoria-filter', 'value'),
    Input('povo-filter', 'value'),
    Input('estado-filter', 'value')
)
def filter_data(selected_categoria, selected_povo, selected_estado):
    filtered_df = plot_df.copy()

    # Applying filters if a selection is made
    if selected_categoria != 'all':
        filtered_df = filtered_df[filtered_df['categoria'] == selected_categoria]
    
    if selected_povo != 'all':
        filtered_df = filtered_df[filtered_df['povo'] == selected_povo]
        
    if selected_estado != 'all':
        filtered_df = filtered_df[filtered_df['estado_de_origem'] == selected_estado]

    # Updating dataframe visibility
    plot_df['visibility'] = plot_df.index.isin(filtered_df.index)

    return True

# Callback for map state-items modal
@app.callback(
    Output("modal-items", "is_open"),
    Output("state-items", "children"),
    Output("state-header", "children"),
    Input("brazil-map", "clickData"),
    State("modal-items", "is_open")
)
def display_state_items(clickData, is_open):
    # Extracting clicked state and filtering items belonging to that state
    state_name = clickData["points"][0]["hovertext"]
    state_symb = brazil_states_dict[state_name]

    state_items = plot_df[plot_df['estado_de_origem'].apply(lambda x: state_symb in x if pd.notna(x) else False)]

    # Create a list of items to display
    items_list = html.Ul([
        html.Li([
            html.Img(src=row['thumbnail'], style={'width': '80px'}),
            html.A(f"{row['nome_do_item'].title()} - {row['povo'].title()} - {row['ano_de_aquisicao']}", href=row['url'], target="_blank", style={'font-weight': 'bold', 'text-decoration': 'none', 'color': '#062a57', 'margin-left': '10px'})
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '25px'})
        for index, row in state_items.iterrows()
    ], style={'list-style-type': 'none', 'padding': '0'})

    header_title = f'Itens Advindos do {state_name}'

    return not is_open, items_list, header_title

# Running app
if __name__ == '__main__':
    app.run_server(debug=True)