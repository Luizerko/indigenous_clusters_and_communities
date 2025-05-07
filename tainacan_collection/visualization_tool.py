import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State, no_update
from dash_extensions.enrich import DashProxy, MultiplexerTransform
import dash_bootstrap_components as dbc

import ast
import re
from PIL import Image
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

from google.cloud import storage

from visual_utils import *

from sklearn.datasets import make_blobs

######### DATA LOADING AND PROCESSING #########
# Loading dataset and initializing plot grpah
ind_df = pd.read_csv('data/indigenous_collection_processed.csv', index_col='id')
x, y = make_blobs(n_samples=len(ind_df), centers=10, random_state=42, center_box=(-norm_factor,norm_factor))
plot_df = pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "cluster": y})

# Loading projections
tipo_materia_prima_baseline_df = pd.read_csv('data/projections/tipo_materia_prima_baseline.csv', index_col='id')

# vanilla_vit_df = pd.read_csv('data/projections/vanilla_vit.csv', index_col='id')
povo_vit_df = pd.read_csv('data/projections/povo_vit.csv', index_col='id')
categoria_vit_df = pd.read_csv('data/projections/categoria_vit.csv', index_col='id')
# multihead_vit_df = pd.read_csv('data/projections/multihead_vit.csv', index_col='id')

vanilla_dino_df = pd.read_csv('data/projections/vanilla_dino.csv', index_col='id')
# povo_dino_df = pd.read_csv('data/projections/povo_dino.csv', index_col='id')
# categoria_dino_df = pd.read_csv('data/projections/categoria_dino.csv', index_col='id')
multihead_dino_df = pd.read_csv('data/projections/multihead_dino.csv', index_col='id')

# vanilla_bertimbau_df = pd.read_csv('data/projections/vanilla_bertimbau_trimap.csv', index_col='id')
vanilla_bertimbau_df = pd.read_csv('data/projections/vanilla_bertimbau_umap.csv', index_col='id')

# simcse_bertimbau_df = pd.read_csv('data/projections/simcse_bertimbau_trimap.csv', index_col='id')
simcse_bertimbau_df = pd.read_csv('data/projections/simcse_bertimbau_umap.csv', index_col='id')

# Creating artificial index to interact with our dataframe
plot_df['ind_index'] = ind_df.index

# Extracting and processing information that will be used from dataframe
plot_df['image_path'] = ind_df['image_path'].values

# Initializing the storage client with my service account key
storage_client = storage.Client.from_service_account_json('data/master-thesis-454117-c3204ebce791.json')

# Creating temporary URL for lazy loading images
plot_df['temporary_br_url'] = pd.NA
plot_df.loc[plot_df['image_path'].notna(), 'temporary_br_url'] = plot_df.loc[plot_df['image_path'].notna(), 'image_path'].apply(lambda path: generate_signed_url(storage_client, 'background-removed-tainacan-images', f"{path.split('/')[-1].split('.')[0]}.png", expiration_minutes=10))
plot_df.loc[plot_df['temporary_br_url'].isna(), 'temporary_br_url'] = generate_signed_url(storage_client, 'background-removed-tainacan-images', 'placeholder_square.png', expiration_minutes=10)

plot_df['image_path_br'] = ind_df['image_path'].values
plot_df.loc[plot_df['image_path_br'].notna(), 'image_path_br'] = plot_df.loc[plot_df['image_path_br'].notna(), 'image_path'].apply(lambda path: f"data/br_images/{path.split('/')[-1].split('.')[0]}.png")
plot_df.fillna({'image_path': 'data/placeholder_square.png'}, inplace=True)
plot_df.fillna({'image_path_br': 'data/placeholder_square.png'}, inplace=True)

plot_df['url'] = ind_df['url'].values

plot_df['nome_do_item'] = ind_df['nome_do_item'].values
plot_df.fillna({'nome_do_item': '----'}, inplace=True)

plot_df['povo'] = ind_df['povo'].values
plot_df.fillna({'povo': '----'}, inplace=True)

plot_df['categoria'] = ind_df['categoria'].values

plot_df['ano_de_aquisicao'] = ind_df['ano_de_aquisicao'].values
plot_df.fillna({'ano_de_aquisicao': '0'}, inplace=True)
plot_df['ano_de_aquisicao'] = plot_df['ano_de_aquisicao'].astype(int)

plot_df['data_de_aquisicao'] = ind_df['data_de_aquisicao'].values
plot_df.fillna({'data_de_aquisicao': '0001-01-01'}, inplace=True)
plot_df.loc[plot_df['data_de_aquisicao'].str[:4] != plot_df['ano_de_aquisicao'].astype(str), 'data_de_aquisicao'] = '0001-01-01'

plot_df['estado_de_origem'] = ind_df['estado_de_origem'].values

plot_df['colecao'] = ind_df['colecao'].values
plot_df['coletor'] = ind_df['coletor'].values

plot_df['thumbnail'] = ind_df['thumbnail'].values
plot_df.fillna({'thumbnail': 'https://tainacan.museudoindio.gov.br/wp-content/plugins/tainacan/assets/images/placeholder_square.png'}, inplace=True)

plot_df['descricao'] = ind_df['descricao'].values
plot_df.fillna({'descricao': 'Item sem descrição.'}, inplace=True)

# Creating extra filters
plot_df.set_index('ind_index', inplace=True)

plot_df['tipo_materia_prima'] = pd.NA
indices = tipo_materia_prima_baseline_df.index
plot_df.loc[indices, 'tipo_materia_prima'] = tipo_materia_prima_baseline_df['cluster_names'].values

plot_df['comprimento'] = 0.0
plot_df['largura'] = 0.0
plot_df['altura'] = 0.0
plot_df['diametro'] = 0.0
dimensoes = {0 : [], 1: [], 2: [], 3: []}
noise_threshold = 1000
for row in ind_df['dimensoes'].dropna():
    for i, d in enumerate(ast.literal_eval(row)):
        if d > 0 and d < noise_threshold:
            dimensoes[i].append(round(d, 1))
        else:
            dimensoes[i].append(0.0)
plot_df.loc[ind_df['dimensoes'].notna(), 'comprimento'] = dimensoes[0]
plot_df.loc[ind_df['dimensoes'].notna(), 'largura'] = dimensoes[1]
plot_df.loc[ind_df['dimensoes'].notna(), 'altura'] = dimensoes[2]
plot_df.loc[ind_df['dimensoes'].notna(), 'diametro'] = dimensoes[3]

# Fixing cluster names for tipo_materia_prima, but only after giving proper data structure (list) so plot_df can process dropdown options
tipo_materia_prima_baseline_df['cluster_names'] = tipo_materia_prima_baseline_df['cluster_names'].apply(lambda x: ', '.join(ast.literal_eval(x)))

plot_df.reset_index(inplace=True)

# Setting point visibility and initializing first plot
plot_df['visibility'] = False
plot_df.set_index('ind_index', inplace=True)
indices = categoria_vit_df.index
plot_df.loc[indices, 'visibility'] = True
plot_df.loc[indices, "x"] = categoria_vit_df['x'].values
plot_df.loc[indices, "y"] = categoria_vit_df['y'].values
plot_df.loc[indices, "cluster"] = categoria_vit_df['cluster'].values
plot_df.loc[indices, 'cluster_names'] = categoria_vit_df['cluster_names'].values

# Getting first cluster names and creating initial color map
plot_df['cluster_names'] = ''
plot_df.loc[indices, 'cluster_names'] = categoria_vit_df['cluster_names'].values
plot_df.fillna({'cluster_names': ''}, inplace=True)

plot_df['color'] = 'rgba(255, 255, 255, 1)'
color_map = generate_color_map(plot_df.loc[indices, 'cluster_names'].values)
plot_df.loc[indices, 'color'] = [color_map[label] for label in plot_df.loc[indices, 'cluster_names'].values]

plot_df.reset_index(inplace=True)

# Dash app setup. DashPRoxy used for multiple callbacks with the same output, but made the app a bit buggy (multiple triggers of the same callback in a row)
external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css"]
app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()], external_stylesheets=external_stylesheets)

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
            id='aba-agrupamentos',
            className='tab-option',
            children=[
                dbc.Tooltip(
                    "Essa aba permite que o usuário explore diferentes agrupamentos nos dados do acervo, baseados tanto em similaridade imagética dos itens como em semelhanças textuais em suas descrições. Além disso, o usuário pode filtrar os dados a serem visualizados como bem entender, fazendo da página basicamente uma versão interativa do Tainacan.",
                    target="aba-agrupamentos",
                    trigger='click',
                    id="tooltip-info-agrupamentos",
                    placement="bottom",
                    is_open=False,
                    className='tab-tooltip',
                ),
                html.Div(
                    className='tool-container',
                    children=[
                        html.Div(
                            className='sidebar-graph',
                            children=[
                                html.Div(
                                    className='sidebar',
                                    children=[
                                        html.Label("Opções de Exibição", id='exhibition-options', style={'fontWeight': 'bold'}),
                                        dbc.Tooltip(
                                            "Essa opção permite que o usuário escolha o tipo de visualização do acervo: nuvem de pontos no espaço associados a itens ou nuvem imagens dos itens. Note que a nuvem de imagens oferece menos interatividade, sendo designada apenas para visualização.",
                                            target="exhibition-options",
                                            trigger='click',
                                            id="tooltip-info-exhibition-options",
                                            placement="right-start",
                                            is_open=False,
                                            className='filter-tooltip',
                                        ),
                                        dcc.RadioItems(
                                            id='toggle-view',
                                            options=[
                                                {'label': 'Pontos', 'value': 'markers'},
                                                {'label': 'Imagens', 'value': 'images'}
                                            ],
                                            value='markers',
                                            className='toggle-view'
                                        ),

                                        html.Label('Opções de Agrupamento', id='grouping-options', style={'fontWeight': 'bold', 'marginTop': '20px', 'marginBottom': '5px'}),
                                        dbc.Tooltip(
                                            'Essa opção permite ao usuário decidir como explorar a nuvem de pontos em relação a seus agrupamentos: por similaridade imagética, similaridade descritiva ou por similaridade de algum outro atributo. Esses agrupamentos, produzidos através da ajuda de inteligência artificial, vêm em diversas modalidades, podendo ressaltar, ou não, categorias contidas nos dados.',
                                            target="grouping-options",
                                            trigger='click',
                                            id="tooltip-info-grouping-options",
                                            placement="right-start",
                                            is_open=False,
                                            className='filter-tooltip',
                                        ),
                                        dcc.Dropdown(
                                            id='cluster-options',
                                            options=[
                                                {'label': 'Tipo de Materia Prima', 'value': 'cluster_1'},
                                                {'label': 'Similaridade Imagética', 'value': 'cluster_2'},
                                                {'label': 'Similaridade Imagética (por Categoria)', 'value': 'cluster_3'},
                                                {'label': 'Similaridade Imagética (por Povo)', 'value': 'cluster_4'},
                                                {'label': 'Similaridade Imagética (por Categoria e Povo)', 'value': 'cluster_5'},
                                                {'label': 'Similaridade Textual', 'value': 'cluster_6'}
                                            ],
                                            multi=False,
                                            placeholder='Selecione uma opção de agrupamento',
                                            value='cluster_3',
                                            clearable=False,
                                            # className='cluster-dropup'
                                        ),

                                        html.Label('Granularidade da Nuvem', id='granularity-slider', style={'fontWeight': 'bold', 'marginTop': '30px', 'marginBottom': '5px'}),
                                        dbc.Tooltip(
                                            'Esse controle permite ajustar o nível de detalhe com que os pontos da nuvem são agrupados. Quando a granularidade está baixa, os pontos próximos são agrupados em blocos maiores e mais espaçados - isso melhora o desempenho e é ideal quando você quer focar em uma área específica e pode dar bastante zoom para ver os detalhes. Já com a granularidade alta, os agrupamentos são menores e mais numerosos, o que mostra a nuvem com mais detalhes em regiões maiores ou até no geral. No entanto, isso pode deixar o sistema mais pesado, já que mais pontos precisam ser exibidos ao mesmo tempo.',
                                            target="granularity-slider",
                                            trigger='click',
                                            id="tooltip-info-granularity-slider",
                                            placement="right-start",
                                            is_open=False,
                                            className='filter-tooltip',
                                        ),
                                        dcc.Slider(0, 4,
                                            step=None,
                                            marks={0: 'Muito\nBaixa', 1: 'Baixa', 2: 'Média', 3: 'Alta', 4: 'Muito\nAlta'},
                                            value=2,
                                            className='filter-slider',
                                            id='granularity-filter'
                                        ),

                                        html.Label("Filtragem de Dados", id='data-filtering', style={'fontWeight': 'bold', 'marginTop': '40px'}),
                                        dbc.Tooltip(
                                            'As opções dessa seção permitem que o usuário filtre os dados a serem observados de maneira tal a granularizar a pesquisa e visualização pensando no seu objeto alvo. Note que, para os filtros numéricos, o valor "0" indica ausência do dado, então um item de Comprimento 0cm, por exemplo, é um item para o qual não existe informação de Comprimento.',
                                            target="data-filtering",
                                            trigger='click',
                                            id="tooltip-info-data-filtering",
                                            placement="right-start",
                                            is_open=False,
                                            className='filter-tooltip',
                                        ),
                                        html.Div(
                                            className='filter-dropdown',
                                            children=[
                                                html.Label("Categoria:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                dcc.Dropdown(
                                                    id='categoria-filter',
                                                    options=get_dropdown_options(ind_df, 'categoria'),
                                                    multi=True,
                                                    placeholder='Filtrar por categoria',
                                                    value=[],
                                                    clearable=False
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
                                                    multi=True,
                                                    placeholder='Filtrar por povo',
                                                    value=[],
                                                    clearable=False
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
                                                    multi=True,
                                                    placeholder='Filtrar por estado de origem',
                                                    value=[],
                                                    clearable=False
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className='filter-dropdown',
                                            children=[
                                                html.Label("Tipo de Matéria Prima:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                dcc.Dropdown(
                                                    id='materia-filter',
                                                    options=get_dropdown_options(plot_df, 'tipo_materia_prima'),
                                                    multi=True,
                                                    placeholder='Filtrar por tipo de matéria prima',
                                                    value=[],
                                                    clearable=False
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className='filter-range',
                                            children=[
                                                html.Label("Ano de Aquisição:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                html.Div(
                                                    className='input-container',
                                                    children=[
                                                        html.Label("Min", style={'fontSize': '14px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='ano-min',
                                                            type='number',
                                                            placeholder=f"{plot_df['ano_de_aquisicao'].min()}",
                                                            value=plot_df['ano_de_aquisicao'].min(),
                                                            step=1,
                                                        ),
                                                        html.Label("Max", style={'fontSize': '14px', 'marginLeft': '20px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='ano-max',
                                                            type='number',
                                                            placeholder=f"{plot_df['ano_de_aquisicao'].max()}",
                                                            value=plot_df['ano_de_aquisicao'].max(),
                                                            step=1,
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className='filter-range',
                                            children=[
                                                html.Label("Comprimento (cm):", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                html.Div(
                                                    className='input-container',
                                                    children=[
                                                        html.Label("Min", style={'fontSize': '14px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='comprimento-min',
                                                            type='number',
                                                            placeholder=f"{plot_df['comprimento'].min():.1f}",
                                                            value=round(plot_df['comprimento'].min(), 1),
                                                            step="any",
                                                        ),
                                                        html.Label("Max", style={'fontSize': '14px', 'marginLeft': '20px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='comprimento-max',
                                                            type='number',
                                                            placeholder=f"{plot_df['comprimento'].max():.1f}",
                                                            value=round(plot_df['comprimento'].max(), 1),
                                                            step="any",
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className='filter-range',
                                            children=[
                                                html.Label("Largura (cm):", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                html.Div(
                                                    className='input-container',
                                                    children=[
                                                        html.Label("Min", style={'fontSize': '14px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='largura-min',
                                                            type='number',
                                                            placeholder=f"{plot_df['largura'].min():.1f}",
                                                            value=round(plot_df['largura'].min(), 1),
                                                            step="any",
                                                        ),
                                                        html.Label("Max", style={'fontSize': '14px', 'marginLeft': '20px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='largura-max',
                                                            type='number',
                                                            placeholder=f"{plot_df['largura'].max():.1f}",
                                                            value=round(plot_df['largura'].max(), 1),
                                                            step="any",
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className='filter-range',
                                            children=[
                                                html.Label("Altura (cm):", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                html.Div(
                                                    className='input-container',
                                                    children=[
                                                        html.Label("Min", style={'fontSize': '14px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='altura-min',
                                                            type='number',
                                                            placeholder=f"{plot_df['altura'].min():.1f}",
                                                            value=round(plot_df['altura'].min(), 1),
                                                            step="any",
                                                        ),
                                                        html.Label("Max", style={'fontSize': '14px', 'marginLeft': '20px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='altura-max',
                                                            type='number',
                                                            placeholder=f"{plot_df['altura'].max():.1f}",
                                                            value=round(plot_df['altura'].max(), 1),
                                                            step="any",
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className='filter-range',
                                            children=[
                                                html.Label("Diâmetro (cm):", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                                                html.Div(
                                                    className='input-container',
                                                    children=[
                                                        html.Label("Min", style={'fontSize': '14px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='diametro-min',
                                                            type='number',
                                                            placeholder=f"{plot_df['diametro'].min():.1f}",
                                                            value=round(plot_df['diametro'].min(), 1),
                                                            step="any",
                                                        ),
                                                        html.Label("Max", style={'fontSize': '14px', 'marginLeft': '20px', 'marginRight': '5px'}),
                                                        dcc.Input(
                                                            id='diametro-max',
                                                            type='number',
                                                            placeholder=f"{plot_df['diametro'].max():.1f}",
                                                            value=round(plot_df['diametro'].max(), 1),
                                                            step="any",
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),

                                        dcc.Store(id='zoom-update')
                                    ]
                                ),
                                html.Div(
                                    className='graph',
                                    children=[
                                        dcc.Store(id='url-store', data=[]),
                                        html.Div(id='dummy'),
                                        
                                        dcc.Graph(id='cluster-plot', config=config, figure=empty_figure(), clear_on_unhover=True),

                                        dcc.Interval(id='fade-in', interval=100, n_intervals=0, disabled=True),
                                        
                                        dcc.Tooltip(id='graph-tooltip')
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ],
        ),
        dcc.Tab(
            label='Coleção no Tempo',
            id='aba-timeline',
            className='tab-option',
            children=[
                dbc.Tooltip(
                    "Essa aba permite que o usuário navegue pela coleção através de sua linha do tempo, selecionando coleções de itens por seu ano de aquisição pelo museu. Note que o tamanho dos marcadores é proporcional à quantidade de itens adquiridos naquele ano.",
                    target="aba-timeline",
                    trigger='click',
                    id="tooltip-info-timeline",
                    placement="bottom",
                    is_open=False,
                    className='tab-tooltip',
                ),
                html.Div(
                    className='timeline-container',
                    children=[
                        dcc.Store(id='timeline-url-store', data=[]),
                        html.Div(id='timeline-dummy'),

                        dcc.Graph(id='timeline', config=config, clear_on_unhover=True, figure=timeline_figure_zigzag(ind_df['ano_de_aquisicao'])),
                        dcc.Store(id='turn-grid', data=1),
                        dcc.Tooltip(id='timeline-tooltip'),
                    ]
                ),
            ],
        ),
        dcc.Tab(
            label='Coleção no Brasil',
            id='aba-mapa',
            className='tab-option',
            children=[
                dbc.Tooltip(
                    "Essa aba permite que o usuário estude a coleção geograficamente, entendendo a regionalidade dos povos e localizando os itens no mapa do Brasil.",
                    target="aba-mapa",
                    trigger='click',
                    id="tooltip-info-mapa",
                    placement="bottom",
                    is_open=False,
                    className='tab-tooltip',
                ),
                html.Div(
                    className='map-container',
                    children=[
                        dcc.Graph(id='brazil-map', config=config, figure=brazil_figure()),
                        dcc.Store(id='current-name', data=None),
                        dcc.Store(id='current-page', data=0),
                        dcc.Store(id='current-curve', data=0)
                    ]
                ),
            ],
        )
    ]),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle('Itens Advindos do', style={'fontWeight': 'bold', 'color': '#062a57'}, id='state-header')),
        dbc.ModalBody([
            html.Div(id='state-items'),
            html.Div(id="pagination-text", style={"text-align": "center", "margin-top": "10px", "font-weight": "bold", "color": "#062a57"}),
            html.Div([
                dbc.Button("Anterior", id="prev-page", style={"margin-right": "20px", "fontWeight": "bold", "backgroundColor": "#062a57", "color": "white", "borderRadius": "10px", "border": "none", "padding": "10px 10px"}),
                dbc.Button("Próxima", id="next-page", style={"fontWeight": "bold", "backgroundColor": "#062a57", "color": "white", "borderRadius": "10px", "border": "none", "padding": "10px 10px"})
            ], style={"display": "flex", "justify-content": "center", "margin-top": "20px"})
        ])
    ], id='modal-items', scrollable=True, is_open=False, backdrop=True, size='lg')
], className='base-background')

################## CALLBCAKS ##################
# Callback for hovering on 'Agrupamentos do Acervo'
@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Output('cluster-plot', 'figure'),
    Input("cluster-plot", "hoverData"),
    State('cluster-plot', 'figure'),
    State('cluster-options', 'value'),
    prevent_initial_call=True
)
def display_hover(hover_data, fig, grouping):
    fig = go.Figure(fig)

    # (Almost) remove transition from hover callback (should we do that?)
    fig.update_layout(
        transition=dict(duration=25)
    )

    if hover_data is None:
        # Resetting markers on hover-out
        old_sizes = list(np.full((len(fig.data[0].x)), 15))
        old_line_widths = list(np.full((len(fig.data[0].x)), 0))

        fig.data[0].marker.size = old_sizes
        fig.data[0].marker.line.width = old_line_widths

        return False, no_update, no_update, fig 

    # Extracting plotly dash information
    pt = hover_data["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    
    # Acessing the dataframe to get the data we actually want to display
    df_row = plot_df.iloc[fig.data[0].customdata[num]]

    # Adding hovering effects
    new_sizes = np.full((len(fig.data[0].x)), 15)
    new_sizes[num] = 25
    new_sizes = list(new_sizes)

    new_line_widths = np.full((len(fig.data[0].x)), 0)
    new_line_widths[num] = 10
    new_line_widths = list(new_line_widths)

    color = df_row['color']
    color = color.replace('1)', '0.5)')

    fig.data[0].marker.size = new_sizes
    fig.data[0].marker.line.width = new_line_widths
    fig.data[0].marker.line.color = color

    # Getting hovering information for box display
    nome_do_item = df_row['nome_do_item']
    povo = df_row['povo']
    ano_de_aquisicao = df_row['ano_de_aquisicao']
    if ano_de_aquisicao == 0:
        ano_de_aquisicao = '----'
    img_src = df_row['image_path']

    # Building tooltip for text similarity visualization
    if grouping == 'cluster_6':
        card_width = 300

        # Getting specific hovering information (text)
        ind_index = df_row['ind_index']
        token_attribution_map = ast.literal_eval(vanilla_bertimbau_df.loc[ind_index, 'token_attribution_map'])

        # getting parameters for attribution normalization (to the [0, 1] interval) for better visuals
        attributions = list(token_attribution_map.values())
        min_attribution, max_attribution = min(attributions), max(attributions)

        # Creating a <span> for each token, with opacity of background proportional to the attribution of the respective token
        spans = []
        for token, attribution in token_attribution_map.items():
            # Normalizing attribution
            attribution = (attribution - min_attribution)/(max_attribution-min_attribution)
            opacity = attribution*0.7
            spans.append(
                html.Span(
                    token,
                    style={
                        'backgroundColor': f'rgba(255,40,0,{opacity})',
                        'padding': '0 2px',
                        'borderRadius': '2px',
                        'margin': '0 1px',
                        'display': 'inline-block'
                    }
                )
            )

        # Wrapping all spans in a Div so it flows beneath the title
        description_div = html.Div(
            spans,
            className='hover-box-text',
            style={
                'whiteSpace': 'normal',
                'overflowY': 'hidden',
                'textOverflow': 'ellipsis',
                'display': '-webkit-box',
                'WebkitLineClamp': '8',
                'WebkitBoxOrient': 'vertical',
                'maxHeight': '150px',
                'marginTop': '8px',
                'marginLeft': '8px',
                'marginRight': '8px',
                'textAlign': 'justify'
            }
        )

        # Divider for the parts of the card
        divider = html.Hr(style={
            'borderTop': '2px dashed',
            'marginTop': '10px',
            'marginBottom': '-2.2px',
            'padding': '0',
            'position': 'relative',
            'zIndex': 1,
        })

        # Info row with tiny image + text
        info_div = html.Div(
            children=[
                html.Img(
                    src=Image.open(img_src),
                    style={
                        'width': '130px',
                        'height': '80px',
                        'objectFit': 'cover',
                        'borderRadius': '2px',
                        'position': 'relative',
                        'zIndex': 3
                    }
                ),
                html.Div([
                    html.P(nome_do_item.title(), className='hover-box-text-textual', style={'margin': '0'}),
                    html.P(f'{povo.title()}, {ano_de_aquisicao}', className='hover-box-text-textual', style={'margin': '0'})
                ], style={'display':'flex','flexDirection':'column', 'margin': '0', 'padding': '0', 'flex': '1 1 auto'})
            ],
            style={
                'display': 'flex',
                'alignItems': 'center',
                'padding': '0',
                'zIndex': 2,
                'position': 'relative',
                'justifyContent': 'space-between'
            },
        )

        # Assembling the final children like the image-card, but with text
        card = html.Div(
            className='hover-box',
            children=[description_div, divider, info_div],
            style={'width': f'{card_width}px', 'overflow': 'hidden'}
        )
        children = [card]

    # Building tooltip for image similarity visualizations
    else:
        card_width = 200

        # Hovering box with image only for points with image
        if img_src == 'data/placeholder_square.png':
            children = [
            html.Div(
                className='hover-box',
                children=[
                    html.P(f'{nome_do_item.title()}', className='hover-box-text'),
                    html.P(f'{povo.title()}, {ano_de_aquisicao}', className='hover-box-text')
                ], style={'width': f'{card_width}px'})
        ]

        else:
            children = [
                html.Div(
                    className='hover-box',
                    children=[
                        html.Img(src=Image.open(img_src), className='hover-box-image'),
                        html.P(f'{nome_do_item.title()}', className='hover-box-text'),
                        html.P(f'{povo.title()}, {ano_de_aquisicao}', className='hover-box-text')
                    ], style={'width': f'{card_width}px'})
            ]

    # Changing sied of bbox in case we are in the right side of the image to prevent page breaks
    arrow_size = 8
    gap = 18
    x_data = pt['x']
    x_min, x_max = fig.layout['xaxis']['range']
    move_left = x_data > (x_min+x_max)/2
    
    shift = card_width + gap + arrow_size
    arrow_base = {
        'position':     'absolute',
        'width':        0,
        'height':       0,
        'borderTop':    f'{arrow_size}px solid transparent',
        'borderBottom': f'{arrow_size}px solid transparent',
        'top':          '50%',
        'transform':    'translateY(-50%)'
    }

    if move_left:
        # Shifting bounding box to the left
        bbox['x0'] -= shift
        bbox['x1'] -= shift

        # Arrow sitting on the right edge of our card
        arrow_style = {
            **arrow_base,
            'borderLeft': f'{arrow_size}px solid white',
            'right':      f'-{arrow_size}px'
        }
    else:
        # Arrow sitting on the left edge of our card
        arrow_style = {
            **arrow_base,
            'borderRight': f'{arrow_size}px solid white',
            'left':        f'-{arrow_size}px'
        }

    arrow = html.Div(style=arrow_style)

    # Wrapping existing card and the arrow in a relative container so the arrow absolute-positions correctly:
    card_container = html.Div(
        [arrow, children[0]],
        style={
            'position': 'relative',
            'display':  'inline-block'
        }
    )
    children = [card_container]

    return True, bbox, children, fig

# Callback for clicking on 'Agrupamentos do Acervo'
@app.callback(
    Output("url-store", "data"),
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

# Helper client-side callback so we can open the item on a new tab
app.clientside_callback(
    """
    function(url) {
        if (!url) {
            return "";
        }
        try {
            window.open(url, '_blank');
        } catch(e) {
            console.error("Error opening new tab:", e);
        }
        return "";
    }
    """,
    Output('dummy', 'children'),
    Input('url-store', 'data')
)

# Callback for collapsing points that are close together and to switch between points and images
@app.callback(
    Output('cluster-plot', 'figure'),
    Output('fade-in', 'disabled'),
    Output('zoom-update', 'data'),
    Input('toggle-view', 'value'),
    Input('cluster-plot', 'relayoutData'),
    Input('zoom-update', 'data'),
    Input('granularity-filter', 'value'),
    State('cluster-options', 'value'),
)
def update_scatter_plot(view_type, relayout_data, zoom_update, granularity, grouping):
    # Handling (potential) constant trigerring and app crashing
    if relayout_data is None:
        return no_update, False, False
    
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
    if granularity == 0:
        labels = collapse_cluster_points(coords, x_range, y_range, threshold=0.065)
    elif granularity == 1:
        labels = collapse_cluster_points(coords, x_range, y_range, threshold=0.055)
    elif granularity == 2:
        labels = collapse_cluster_points(coords, x_range, y_range, threshold=0.045)
    elif granularity == 3:
        labels = collapse_cluster_points(coords, x_range, y_range, threshold=0.035)
    elif granularity == 4:
        labels = collapse_cluster_points(coords, x_range, y_range, threshold=0.025)

    # Splitting clusters (inliers) and outliers for collapsing
    collapse_df = pd.DataFrame(coords, columns=["x", "y"], index=filtered_plot_df.index)
    collapse_df["cluster"] = labels
    
    # Getting cluster colors for later painting the collapses
    collapse_df["color"] = filtered_plot_df['color']

    # Getting cluster names for legend
    collapse_df["cluster_names"] = filtered_plot_df['cluster_names']

    # Getting URLs for lazy image loading
    # collapse_df["thumbnail"] = filtered_plot_df['thumbnail']
    collapse_df["temporary_br_url"] = filtered_plot_df['temporary_br_url']

    outliers = collapse_df[collapse_df['cluster'] == -1]
    outliers = outliers.copy()
    outliers.loc[:, 'cluster'] = filtered_plot_df.loc[collapse_df['cluster'] == -1, 'cluster'].values
    outliers.loc[:, 'image_path_br'] = filtered_plot_df.loc[collapse_df['cluster'] == -1, 'image_path_br'].values

    # Lazy plotting outliers for speed
    visible_outliers = outliers[
        (outliers['x'] >= x_range[0]) & (outliers['x'] <= x_range[1]) &
        (outliers['y'] >= y_range[0]) & (outliers['y'] <= y_range[1])
    ]

    # Computing collapses
    inliers = collapse_df[collapse_df['cluster'] != -1]
    centroids_df = inliers.groupby('cluster').agg({'x': 'mean', 'y': 'mean'})
    centroids_df['count'] = inliers.groupby('cluster').size().values
    centroids_df['marker_size'] = centroids_df['count'].apply(lambda c: min(70, max(28, 15*np.log(c))))

    # Computing dominant collapse clusters and colors
    centroids_cluster_names = inliers.groupby('cluster')['cluster_names'].agg(lambda x: x.mode()[0])
    centroids_df['cluster_names'] = centroids_cluster_names
    dominant_colors = inliers.groupby('cluster')['color'].agg(lambda x: x.mode()[0])
    centroids_df['color'] = dominant_colors

    # Lazy plotting collapses for speed
    visible_centroids = centroids_df[
        (centroids_df['x'] >= x_range[0]) & (centroids_df['x'] <= x_range[1]) &
        (centroids_df['y'] >= y_range[0]) & (centroids_df['y'] <= y_range[1])
    ]

    # Extracting (cluster_names, color) for plotting legend later
    visible_outliers_color_df = visible_outliers[['cluster_names', 'color']].drop_duplicates()
    visible_centroids_color_df = visible_centroids[['cluster_names', 'color']].drop_duplicates()
    color_df = pd.concat([visible_outliers_color_df, visible_centroids_color_df]).drop_duplicates()

    # Replotting outliers
    if len(visible_outliers) > 0:
        if view_type == 'markers':
            fig = plot_with_markers(visible_outliers, len(collapse_df), color_df, x_range, y_range, grouping!='cluster_1' and grouping!='cluster_6', grouping=='cluster_2' or grouping=='cluster_5' or grouping=='cluster_6')
        else:
            num_points = len(filtered_plot_df.loc[filtered_plot_df['image_path_br'] != 'data/placeholder_square.png'])
            fig = plot_with_images(visible_outliers, num_points, color_df, x_range, y_range, grouping=='cluster_2' or grouping=='cluster_5' or grouping=='cluster_6')
    else:
        if len(color_df) > 0 and grouping != 'cluster_2' and grouping != 'cluster_5' and grouping != 'cluster_6' :
            fig = empty_figure_legend(color_df, x_range, y_range, len(collapse_df), grouping!='cluster_1' and grouping!='cluster_6')
        else:
            fig = empty_figure(x_range, y_range, len(collapse_df), grouping!='cluster_1' and grouping!='cluster_6')

    # Plotting collapsed points
    fig.add_trace(go.Scatter(
        x=visible_centroids['x'],
        y=visible_centroids['y'],
        mode='markers+text',
        marker=dict(color=visible_centroids['color'], size=visible_centroids['marker_size'], symbol='circle'),
        text=visible_centroids['count'],
        textposition='middle center',
        textfont=dict(color='#ffffff', size=16),
        hoverinfo='skip',
        hovertemplate=None,
        showlegend=False,
        name='collapse_trace'
    ))

    # Fade-out effect for animation
    fade_in = True
    if zoom_update:
        fig.update_traces(
            selector=dict(name='marker_trace'),
            marker=dict(opacity=0)
        )
        fig.update_traces(
            selector=dict(name='collapse_trace'),
            textfont=dict(size=1),
            marker=dict(opacity=0)
        )
        fig.update_layout(
            transition=dict(duration=0, easing='linear')
        )
        fade_in = False

    return fig, fade_in, False

# Callback for fade-in animation effect on graph
@app.callback(
    Output('cluster-plot', 'figure'),
    Input('fade-in', 'disabled'),
    State('cluster-plot', 'figure'),
)
def fade_in_update(fade_in, fig):
    fig = go.Figure(fig)
    
    # No selection, just panning or zooming
    if fade_in:
        return no_update

    # Making particles fade-in after replotting them on update_scatter_plot callback
    fig.update_traces(
        selector=dict(name='marker_trace'),
        marker=dict(opacity=0.65),
    )
    fig.update_traces(
        selector=dict(name='collapse_trace'),
        textfont=dict(size=16),
        marker=dict(opacity=0.7),
    )
    fig.update_layout(
        transition=dict(duration=300, easing='linear')
    )

    return fig

# Callback for changing clustering option
@app.callback(
    Output('zoom-update', 'data'),
    Output('categoria-filter', 'value'),
    Output('povo-filter', 'value'),
    Output('estado-filter', 'value'),
    Output('materia-filter', 'value'),
    Output('ano-min', 'value'),
    Output('ano-max', 'value'),
    Output('comprimento-min', 'value'),
    Output('comprimento-max', 'value'),
    Output('largura-min', 'value'),
    Output('largura-max', 'value'),
    Output('altura-min', 'value'),
    Output('altura-max', 'value'),
    Output('diametro-min', 'value'),
    Output('diametro-max', 'value'),
    Input('cluster-options', 'value')
)
def update_cluster(selected_option):
    # Reseting visibility
    plot_df['visibility'] = False

    # Updating cluster values, positions and colors depending on the chosen clustering option 
    if selected_option == 'cluster_1':
        update_cluster_selection(plot_df, tipo_materia_prima_baseline_df)

    elif selected_option == 'cluster_2':
        update_cluster_selection(plot_df, vanilla_dino_df, no_clusters=True)

    elif selected_option == 'cluster_3':
        update_cluster_selection(plot_df, categoria_vit_df)

    elif selected_option == 'cluster_4':
        update_cluster_selection(plot_df, povo_vit_df)

    elif selected_option == 'cluster_5':
        update_cluster_selection(plot_df, multihead_dino_df, no_clusters=True)

    elif selected_option == 'cluster_6':
        update_cluster_selection(plot_df, vanilla_bertimbau_df, no_clusters=True)

    return True, [], [], [], [], plot_df['ano_de_aquisicao'].min(), plot_df['ano_de_aquisicao'].max(), plot_df['comprimento'].min(), plot_df['comprimento'].max(), plot_df['largura'].min(), plot_df['largura'].max(), plot_df['altura'].min(), plot_df['altura'].max(), plot_df['diametro'].min(), plot_df['diametro'].max()

# Callback for filtering data
@app.callback(
    Output('zoom-update', 'data'),
    Input('categoria-filter', 'value'),
    Input('povo-filter', 'value'),
    Input('estado-filter', 'value'),
    Input('materia-filter', 'value'),
    Input('ano-min', 'value'),
    Input('ano-max', 'value'),
    Input('comprimento-min', 'value'),
    Input('comprimento-max', 'value'),
    Input('largura-min', 'value'),
    Input('largura-max', 'value'),
    Input('altura-min', 'value'),
    Input('altura-max', 'value'),
    Input('diametro-min', 'value'),
    Input('diametro-max', 'value'),
    State('cluster-options', 'value')
)
def filter_data(selected_categorias, selected_povos, selected_estados, selected_materias, ano_min, ano_max, comprimento_min, comprimento_max, largura_min, largura_max, altura_min, altura_max, diametro_min, diametro_max, grouping):
    # Preserving visibility indices
    if grouping == 'cluster_1':
        filtered_df = plot_df[plot_df['ind_index'].isin(tipo_materia_prima_baseline_df.index)].copy()
    elif grouping == 'cluster_2':
        filtered_df = plot_df[plot_df['ind_index'].isin(vanilla_dino_df.index)].copy()
    elif grouping == 'cluster_3':
        filtered_df = plot_df[plot_df['ind_index'].isin(categoria_vit_df.index)].copy()
    elif grouping == 'cluster_4':
        filtered_df = plot_df[plot_df['ind_index'].isin(povo_vit_df.index)].copy()
    elif grouping == 'cluster_5':
        filtered_df = plot_df[plot_df['ind_index'].isin(multihead_dino_df.index)].copy()
    elif grouping == 'cluster_6':
        filtered_df = plot_df[plot_df['ind_index'].isin(vanilla_bertimbau_df.index)].copy()

    # Applying filters if a selection is made
    if selected_categorias is not None and len(selected_categorias) > 0:
        filtered_df = filtered_df[filtered_df['categoria'].isin(selected_categorias)]
    
    if selected_categorias is not None and len(selected_povos) > 0:
        filtered_df = filtered_df[filtered_df['povo'].isin(selected_povos)]
        
    if selected_categorias is not None and len(selected_estados) > 0:
        selected_estados = '|'.join(selected_estados)
        filtered_df = filtered_df[filtered_df['estado_de_origem'].str.contains(selected_estados, na=False)]

    if selected_categorias is not None and len(selected_materias) > 0:
        selected_materias = '|'.join(selected_materias)
        filtered_df = filtered_df[filtered_df['tipo_materia_prima'].str.contains(selected_materias, na=False)]

    # Filtering by year of acquisition
    filtered_df = filtered_df[(filtered_df['ano_de_aquisicao'] >= ano_min) & (filtered_df['ano_de_aquisicao'] <= ano_max)]

    # Filtering by dimensions
    comprimento_min = round(comprimento_min, 1) if comprimento_min is not None else 0.0
    comprimento_max = round(comprimento_max, 1) if comprimento_max is not None else 0.0
    largura_min = round(largura_min, 1) if largura_min is not None else 0.0
    largura_max = round(largura_max, 1) if largura_max is not None else 0.0
    altura_min = round(altura_min, 1) if altura_min is not None else 0.0
    altura_max = round(altura_max, 1) if altura_max is not None else 0.0
    diametro_min = round(diametro_min, 1) if diametro_min is not None else 0.0
    diametro_max = round(diametro_max, 1) if diametro_max is not None else 0.0
    
    filtered_df = filtered_df[(filtered_df['comprimento'] >= comprimento_min) & (filtered_df['comprimento'] <= comprimento_max)]
    filtered_df = filtered_df[(filtered_df['largura'] >= largura_min) & (filtered_df['largura'] <= largura_max)]
    filtered_df = filtered_df[(filtered_df['altura'] >= altura_min) & (filtered_df['altura'] <= altura_max)]
    filtered_df = filtered_df[(filtered_df['diametro'] >= diametro_min) & (filtered_df['diametro'] <= diametro_max)]

    # Updating dataframe visibility
    plot_df['visibility'] = plot_df.index.isin(filtered_df.index)

    return True

# Callback for hovering in both possibilities of timeline tab
@app.callback(
    Output('timeline', 'figure'),
    Output("timeline-tooltip", "show"),
    Output("timeline-tooltip", "bbox"),
    Output("timeline-tooltip", "children"),
    Input('timeline', 'hoverData'),
    State('turn-grid', 'data'),
    State('timeline', 'figure'),
    prevent_initial_call=True
)
def resize_timeline_marker_on_hover(hover_data, turn_grid, fig):
    fig = go.Figure(fig)

    # Adding small transition from hover callback
    fig.update_layout(
        transition=dict(duration=25)
    )

    if turn_grid == 1:
        # Getting counts to compute proper marker sizes
        counts = ind_df['ano_de_aquisicao'].value_counts()
        key_sorted_counts = counts.sort_index(ascending=False)
        year_counts = np.array(key_sorted_counts.tolist())
        marker_sizes = np.maximum(25, 1.3*np.sqrt(year_counts))
        line_sizes = np.maximum(5, 0.2*np.sqrt(year_counts))

        # Resetting markers when hovering another point
        old_sizes = list(marker_sizes)
        old_line_widths = list(line_sizes)
        fig.data[1].marker.size = old_sizes
        fig.data[1].marker.line.width = old_line_widths
        fig.data[1].marker.opacity = 1

        # Resetting hidden text on markers when hovering another point
        text_labels = ['']*len(old_sizes)
        fig.data[2].text = text_labels

        if hover_data and hover_data["points"][0]["curveNumber"] == 1:
            # Extracting plotly dash information and changing size on hover
            num = hover_data["points"][0]["pointNumber"]

            old_sizes[num] = np.max(old_sizes)+2
            old_line_widths[num] = 1
            fig.data[1].marker.size = old_sizes
            fig.data[1].marker.line.width = old_line_widths

            text_labels[num] = year_counts[num]
            fig.data[2].text = text_labels

        return fig, False, no_update, no_update
    
    else:
        # Finding current page on slider
        for i, trace in enumerate (fig.data):
            try:
                if 'year_timeline' in trace['name'] and trace['visible']:
                    page = i
                    break
            except:
                continue

        # Resetting markers when hovering another point
        old_widths = list(np.full((len(fig.data[page].x)), 5))
        fig.data[page].marker.line.width = old_widths
        
        square_sizes = list(np.full((len(fig.data[page].x)), 35))
        fig.data[page+1].marker.size = square_sizes

        colors = list(fig.data[page].marker.line.color)
        square_colors = list(fig.data[page+1].marker.color)
        image_opacities = list(np.full((len(fig.layout.images)), 1))
        image_sizes = list(np.full((len(fig.layout.images)), 0.6))

        for (i, color), square_color, image_opacity, image_size in zip(enumerate(colors), square_colors, image_opacities, image_sizes):
            if '0)' in color:
                colors[i] = color.replace('0)', '1)')
            else:
                colors[i] = color.replace('0.3)', '1)')

            square_colors[i] = square_color.replace('1)', '0)')

            fig.layout.images[i].opacity = image_opacity

            fig.layout.images[i].sizex = image_size
            fig.layout.images[i].sizey = image_size

        fig.data[page].marker.line.color = colors
        fig.data[page+1].marker.color = square_colors

        month_colors = list(fig.data[-1].marker.color)
        for i, month_color in enumerate(month_colors):
            month_colors[i] = month_color.replace('0.3)', '1)')
        fig.data[-1].marker.color = month_colors

        # Grid plot
        if hover_data and hover_data["points"][0]["curveNumber"] == page:
            # Extracting plotly dash information and highlighting on hover
            num = hover_data["points"][0]["pointNumber"]
            
            old_widths[num] = 15
            fig.data[page].marker.line.width = old_widths

            square_sizes[num] *= 2.4
            fig.data[page+1].marker.size = square_sizes

            square_colors[num] = square_colors[num].replace('0)', '1)')
            fig.data[page+1].marker.color = square_colors

            fig.layout.images[num].sizex *= 1.3
            fig.layout.images[num].sizey *= 1.3

            for (i, color), image_opacity in zip(enumerate(colors), image_opacities):
                if i != num:
                    colors[i] = color.replace('1)', '0.3)')
                    fig.layout.images[i].opacity = 0.3
                else:
                    colors[i] = color.replace('1)', '0)')
            fig.data[page].marker.line.color = colors

            # Building hover tooltip
            bbox = hover_data['points'][0]['bbox']
            bbox['x0'] += 30
            bbox['x1'] += 30

            # Acessing the dataframe to get the data we actually want to display
            df_row = plot_df.iloc[fig.data[page].customdata[num]]
            nome_do_item = df_row['nome_do_item']
            povo = df_row['povo']

            if df_row['data_de_aquisicao'] == '0001-01-01':
                data_de_aquisicao = 'Sem Data Exata'
            else:
                data_de_aquisicao = df_row['data_de_aquisicao']
                data_de_aquisicao = data_de_aquisicao[-2:] + '-' + data_de_aquisicao[-5:-3] + '-' + data_de_aquisicao[:4]
            
            if pd.isna(df_row['colecao']):
                colecao = 'Sem Coleção'
            else:
                colecao = df_row['colecao']
            
            if pd.isna(df_row['coletor']):
                coletor = 'Sem Coletor'
            else:
                coletor = df_row['coletor']

            # Plotting the hovering card with item information
            children = [
                html.Div(
                    className='hover-box',
                    children=[
                        html.P(f'{nome_do_item.title()}', className='hover-box-text'),
                        html.P(f'{povo.title()}, {data_de_aquisicao}', className='hover-box-text'),
                        html.P(f'{colecao.title()}', className='hover-box-text'),
                        html.P(f'{coletor.title()}', className='hover-box-text')
                    ], style={'width': '130px'})
            ]

            return fig, True, bbox, children
    
        # Histogram (bar) plot
        elif hover_data and hover_data["points"][0]["curveNumber"] == len(fig.data)-1:
            month = hover_data["points"][0]["pointNumber"]

            # Changing colors for hovered bar plot
            for i, month_color in enumerate(month_colors):
                if i != month:
                    month_colors[i] = month_color.replace('1)', '0.3)')
            fig.data[-1].marker.color = month_colors

            # Changing colors for grid points that belong to the hovered month
            df_rows = plot_df.loc[list(fig.data[page].customdata)]
            df_rows = df_rows.sort_values(by='data_de_aquisicao')
            indices = list(df_rows.index)
            if month == 0:
                interval = [i for i in range(len(df_rows.loc[df_rows['data_de_aquisicao'] == '0001-01-01']))]
            else:
                safe_df = df_rows.loc[df_rows['data_de_aquisicao'] != '0001-01-01']
                safe_rows = safe_df[pd.to_datetime(safe_df['data_de_aquisicao']).dt.month == month]
                interval = list(safe_rows.index)
                interval = [i for i, ind in enumerate(indices) if ind in interval]

            for i in interval:
                old_widths[i] = 10
                square_sizes[i] *= 1.9
                square_colors[i] = square_colors[i].replace('0)', '1)')
                fig.layout.images[i].sizex *= 1.2
                fig.layout.images[i].sizey *= 1.2

            fig.data[page].marker.line.width = old_widths
            fig.data[page+1].marker.size = square_sizes
            fig.data[page+1].marker.color = square_colors

            for (i, color), image_opacity in zip(enumerate(colors), image_opacities):
                if i not in interval:
                    colors[i] = color.replace('1)', '0.3)')
                    fig.layout.images[i].opacity = 0.3
                else:
                    colors[i] = color.replace('1)', '0)')
            fig.data[page].marker.line.color = colors

            return fig, False, no_update, no_update


    return fig, False, no_update, no_update

# Callback to switch between the zigzag timeline and the grid
@app.callback(
    Output('turn-grid', 'data'),
    Output('timeline', 'figure'),
    Output('timeline-url-store', 'data'),
    Input('timeline', 'clickData'),
    State('turn-grid', 'data'),
    State('timeline', 'figure'),
    prevent_initial_call=True
)
def switch_timeline_grid(click_data, turn_grid, fig):
    fig = go.Figure(fig)

    if turn_grid == 0:
        # Clicking back arrow on the year grid
        if click_data and click_data["points"][0]["curveNumber"] == 0:
            # Replotting zigzag figure
            fig = timeline_figure_zigzag(ind_df['ano_de_aquisicao'])
            
            turn_grid = 1
        
        # Clicking an item on the grid
        elif click_data:
            # Finding current page on slider
            for i, trace in enumerate (fig.data):
                try:
                    if 'year_timeline' in trace['name'] and trace['visible']:
                        page = i
                        break
                except:
                    continue
            
            if click_data["points"][0]["curveNumber"] == page:
                # Extracting plotly dash information
                num = click_data["points"][0]['pointIndex']

                # Acessing the dataframe to get the URL we want
                df_row = plot_df.iloc[fig.data[page].customdata[num]]
                url = df_row['url']

                return no_update, no_update, url

    else:
        # Clicking an year
        if click_data and click_data["points"][0]["curveNumber"] == 1:
            # Extracting plotly dash information and plotting grid figure
            pt = click_data["points"][0]
            year = int(pt['text'])
            fig = timeline_figure_grid(plot_df[plot_df['ano_de_aquisicao'] == year], page_size=90)
            
            turn_grid = 0

    return turn_grid, fig, no_update

# Helper client-side callback so we can open the item on a new tab for the timeline grid
app.clientside_callback(
    """
    function(url) {
        if (!url) {
            return "";
        }
        try {
            window.open(url, '_blank');
        } catch(e) {
            console.error("Error opening new tab:", e);
        }
        return "";
    }
    """,
    Output('timeline-dummy', 'children'),
    Input('timeline-url-store', 'data')
)

# Callback for map state-items modal
@app.callback(
    Output("modal-items", "is_open"),
    Output("state-items", "children"),
    Output("state-header", "children"),
    Output("pagination-text", "children"),
    Output("brazil-map", "clickData"),
    Output("current-name", "data"),
    Output("current-page", "data"),
    Output("current-curve", "data"),
    Input("brazil-map", "clickData"),
    Input("prev-page", "n_clicks"),
    Input("next-page", "n_clicks"),
    State("modal-items", "is_open"),
    State("current-name", "data"),
    State("current-page", "data"),
    State("current-curve", "data"),
)
def display_state_items(clickData, prev_page_click, next_page_click, is_open, current_name, current_page, current_curve):
    # Handling context for pagination later
    ctx = dash.callback_context
    potential_button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if potential_button_id == 'prev-page':
        current_page -= 1
        name = current_name
        curve = current_curve
        is_open_update = is_open
    elif potential_button_id == 'next-page':
        current_page += 1
        name = current_name
        curve = current_curve
        is_open_update = is_open
    else:
        name = clickData["points"][0]["hovertext"]
        curve = clickData["points"][0]["curveNumber"]
        is_open_update = not is_open
    
    # Extracting clicked state and filtering items belonging to that state
    if curve == 0:
        symb = brazil_states_dict[name]
        items = plot_df[plot_df['estado_de_origem'].apply(lambda x: symb in x if pd.notna(x) else False)]
        items = items.sort_values(by='nome_do_item')

    # Extracting clicked 'povo' and filtering items belonging to them
    elif curve == 1:
        symb = name.lower()
        items = plot_df[plot_df['povo'].apply(lambda x: symb in x if pd.notna(x) else False)]
        items = items.sort_values(by='nome_do_item')

    # Handling pagination
    items_per_page = 99
    total_pages = len(items)//items_per_page + 1
    if current_page < 0:
        return no_update, no_update, no_update, no_update, no_update, no_update, 0, no_update
    elif current_page > total_pages-1:
        return no_update, no_update, no_update, no_update, no_update, no_update, total_pages-1, no_update

    start_idx = current_page*items_per_page
    end_idx = start_idx+items_per_page
    paginated_items = items.iloc[start_idx:end_idx]

    items_grid = html.Div([
        html.Div([
            html.Img(src=row['thumbnail'], style={'width': '60%', 'margin-bottom': '8px'}),
            html.A([
                row['nome_do_item'].title(), html.Br(),
                f"{row['povo'].title()}, {row['ano_de_aquisicao'] if row['ano_de_aquisicao'] != 0 else '----'}"
            ], href=row['url'], target="_blank", style={'font-weight': 'bold', 'text-decoration': 'none', 'color': '#062a57', 'text-align': 'center', 'font-size': '16px'}),
            dbc.Button(
                html.I(className="bi bi-info-circle-fill"),
                id={'type': 'info-icon', 'index': i},
                color="link",
                style={'font-size': '18px', 'cursor': 'pointer', 'padding': '0px', 'color': '#062a57'}
            ),
            dbc.Popover(
                dbc.PopoverBody(row['descricao'].capitalize()),
                id={'type': 'popover', 'index': i},
                target={'type': 'info-icon', 'index': i},
                trigger="click",
                placement="bottom",
            )
        ], style={
            'display': 'flex', 
            'flex-direction': 'column', 
            'align-items': 'center',
            'justify-content': 'center',
            'margin-bottom': '20px',
            'border': '1px solid #ddd', 
            'padding-top': '10px',
            'border-radius': '8px', 
            'background-color': '#f9f9f9'
        })
        for i, (_, row) in enumerate(paginated_items.iterrows())
    ], style={
        'display': 'grid',
        'grid-template-columns': 'repeat(3, 1fr)',
        'gap': '10px',
        'list-style-type': 'none', 
        'padding': '0'
    })

    pagination_text = f"Página {current_page + 1} / {total_pages}"
    
    if curve == 0:
        header_title = f'{len(items)} Itens do {name}'
    elif curve == 1:
        header_title = f'{len(items)} Itens do povo {name}'

    return is_open_update, items_grid, header_title, pagination_text, None, name, current_page, curve

# Running app
if __name__ == '__main__':
    app.run_server(debug=True)