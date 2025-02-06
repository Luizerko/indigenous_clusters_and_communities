import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
import ast

import pandas as pd
import numpy as np
from scipy.spatial import KDTree

########## VARIABLE INITIALIZATION  ##########
# Brazilian states dataset
brazil_states = pd.DataFrame({
    'state': ['Acre', 'Alagoas', 'Amapá', 'Amazonas', 'Bahia', 'Ceará', 'Distrito Federal', 'Espírito Santo', 'Goiás', 'Maranhão', 'Mato Grosso', 'Mato Grosso do Sul', 'Minas Gerais', 'Pará', 'Paraíba', 'Paraná', 'Pernambuco', 'Piauí', 'Rio de Janeiro', 'Rio Grande do Norte', 'Rio Grande do Sul', 'Rondônia', 'Roraima', 'Santa Catarina', 'São Paulo', 'Sergipe', 'Tocantins'],

    'latitude': [-8.77, -9.62, 1.41, -3.07, -12.96, -5.20, -15.83, -19.19, -15.98, -4.96, -12.64, -20.51, -18.10, -3.79, -7.12, -24.89, -8.28, -6.60, -22.91, -5.81, -30.03, -10.90, 2.82, -27.33, -23.55, -10.57, -10.25],
    'longitude': [-70.55, -36.82, -51.77, -61.66, -38.51, -39.53, -47.86, -40.34, -49.86, -44.30, -55.42, -54.54, -44.38, -52.49, -34.83, -51.55, -34.88, -42.28, -43.20, -36.59, -51.22, -62.80, -60.67, -49.44, -46.63, -37.06, -48.25],
})

brazil_states_dict = {"Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM", "Bahia": "BA", "Ceará": "CE", "Distrito Federal": "DF", "Espírito Santo": "ES", "Goiás": "GO", "Maranhão": "MA", "Mato Grosso": "MT", "Mato Grosso do Sul": "MS", "Minas Gerais": "MG", "Pará": "PA", "Paraíba": "PB", "Paraná": "PR", "Pernambuco": "PE", "Piauí": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN", "Rio Grande do Sul": "RS", "Rondônia": "RO", "Roraima": "RR", "Santa Catarina": "SC", "São Paulo": "SP", "Sergipe": "SE", "Tocantins": "TO"}

# Normalizing factor for visualization range
norm_factor = 12

########## UTIL FUNCTIONS FOR FIGURE ##########
# Updating layout for any given figure
def fig_update_layout(fig, df_len, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor)):
    # Customizing layout
    fig.update_layout(
        # Adjusting style of the graph
        plot_bgcolor="#e6e6e6",
        paper_bgcolor="#e6e6e6",
        
        font=dict(family="Roboto, sans-serif", size=16, color="black"),
        
        title_x=0.5,
        margin=dict(l=0, r=0, t=0, b=0),
        
        showlegend=True,
        legend=dict(yanchor='top', y=0.98, xanchor='right', x=0.99, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor="#062a57", borderwidth=1, orientation='v', itemclick=False, itemdoubleclick=False),

        xaxis_title=None,
        yaxis_title=None,
        coloraxis_showscale=False,

        # Mouse default configuration (panning instead of zooming)
        dragmode='pan',
        hoverdistance = 6,

        # # Animating graphs
        # transition=dict(duration=500, easing='sin-in-out')
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

    # Number of items box
    fig.add_annotation(
        text=f"Número de Itens: {df_len}",
        xref="paper", yref="paper",
        x=0.99, y=0.02,
        showarrow=False,
        font=dict(size=14, color="#062a57"),
        align="center",
        bordercolor="#062a57",
        borderwidth=1,
        borderpad=8,
        bgcolor='rgba(255, 255, 255, 0.8)',
    )

# Create empty figure for initialization
def empty_figure(x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor)):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig_update_layout(fig, 0, x_range, y_range)
    return fig

# Create timeline figure


# Creating the map of Brazil and plotting markers on states
def brazil_figure():
    fig = px.scatter_mapbox(brazil_states, lat='latitude', lon='longitude', hover_name='state', hover_data={'latitude': False, 'longitude': False}, zoom=3.5, center={'lat': -14.2350, 'lon': -51.9253}, width=1350, height=600)

    fig.update_layout(
        mapbox_style="carto-positron",  # options are 'open-street-map', 'stamen-terrain', 'carto-positron', etc.
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Mapa do Brasil e suas Comunidades Indígenas"
    )

    fig.update_traces(marker=dict(size=20, color='#062a57'))

    return fig

# Create scatter plot with markers
def plot_with_markers(df, num_points, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor)):
    # Creating Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        customdata=df.index,
        mode='markers',
        marker=dict(color=df['color']),
        showlegend=False,
        name='marker_trace'
    ))

    # Configuring default hovering
    fig.update_traces(hoverinfo='none', hovertemplate=None, marker=dict(size=20, opacity=0.65), line=dict(width=0, color='rgb(255, 212, 110)'))

    # Updating layout for our standard design
    fig_update_layout(fig, num_points, x_range, y_range)

    # Plotting legend
    unique_clusters = df[['color', 'cluster_names']].drop_duplicates()
    unique_clusters = unique_clusters.sort_values(by=['cluster_names'], ascending=True).sort_values(by='cluster_names', key=lambda x: x.str.len())
    legend_order = unique_clusters['cluster_names'].tolist()

    color_dict = {}
    for _, row in unique_clusters.iterrows():
        color_dict[str(row['cluster_names'])] = row['color']
    dummy_fig = px.scatter(
        df,
        x=[None for i in range(len(df))],
        y=[None for i in range(len(df))],
        color='cluster_names',
        labels={'cluster_names': 'Cluster Names'},
        color_discrete_map=color_dict,
        category_orders={'cluster_names': legend_order}
    )
    for trace in dummy_fig.data:
        trace.showlegend = True
        trace.marker.size = 15
        fig.add_trace(trace)

    return fig

# Create scatter plot with the images themselves
def plot_with_images(df, num_points, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor)):
    df = df.loc[df['image_path_br'] != 'data/placeholder_square.png']
    fig = go.Figure()
    for index, row in df.iterrows():
        fig.add_layout_image(
            dict(source=Image.open(row['image_path_br']), x=row['x'], y=row['y'], xref="x", yref="y", sizex=(x_range[1]-x_range[0])/8, sizey=(y_range[1]-y_range[0])/8, xanchor="center",yanchor="middle")
        )
    
    fig_update_layout(fig, num_points, x_range, y_range)

    # Observation for image option
    fig.add_annotation(
        text="Apenas itens com imagens são exibidos nessa opção",
        xref="paper", yref="paper",
        x=0.01, y=0.02,
        showarrow=False,
        font=dict(size=14, color="#062a57"),
        align="center",
        bordercolor="#062a57",
        borderwidth=1,
        borderpad=8,
        bgcolor='rgba(255, 255, 255, 0.8)',
    )

    return fig

# Generating a (fixed) color palette for cluster IDs to handle lazy plotting
def generate_color_map(clusters):
    ids = np.unique(clusters)
    cmap = plt.cm.get_cmap('tab10', len(ids))
    color_map = {cluster_id: f'rgba({cmap(i)[0]*255}, {cmap(i)[1]*255}, {cmap(i)[2]*255}, 1)' for i, cluster_id in enumerate(ids)}

    return color_map


########## GENERAL PURPOSE UTIL FUNCTIONS ##########
# Normalize points based on zoom level
def normalize_points(points, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor)):
    norm_x = (points[:, 0]-x_range[0]) / (x_range[1]-x_range[0])
    norm_y = (points[:, 1]-y_range[0]) / (y_range[1]-y_range[0])
    
    return np.column_stack((norm_x, norm_y))

# Clustering points according to distance (normalized by zoom level)
def collapse_cluster_points(points, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor), threshold=0.03):

    # Normalizing points and building KDTree for fast (spatial) lookup
    normalized_points = normalize_points(points, x_range, y_range)
    tree = KDTree(normalized_points)

    # Computing clusters by "initializing" centroids at non-visited points and checking neighborhood's distance
    clusters = []
    min_distances = []
    visited = set()
    for i, point_1 in enumerate(normalized_points):
        if i in visited:
            continue

        # Find neighbors within threshold distance and compute clusters
        neighbors = tree.query_ball_point(point_1, r=threshold)
        distances, _ = tree.query([point_1], k=len(neighbors))
        cluster = []
        for j in neighbors:
            if j not in visited:
                cluster.append(j)
                visited.add(j)
                
        if len(cluster) > 1:
            clusters.append(cluster)
            min_distances.append(np.min(distances[0][1:]))
        else:
            clusters.append([i])
            min_distances.append(100)
    
    # Extracting labels for each point from clusters
    labels = {}
    for i, (cluster, min_distance) in enumerate(zip(clusters, min_distances)):
        if len(cluster) < 10 and min_distance > threshold/8:
            for point_index in cluster:
                labels[point_index] = -1
        else:
            for point_index in cluster:
                labels[point_index] = i
    labels = [labels[i] for i in range(len(points))]

    return labels

# Getting all the dropdown options for a given column
def get_dropdown_options(df, column_name):
    unique_values = df[column_name].dropna().unique().tolist()
    
    # Handling special cases
    if column_name == 'estado_de_origem':
        # Items with multiple categories at the same time
        unique_values_aux = set()
        for state_list in unique_values:
            for state in ast.literal_eval(state_list):
                unique_values_aux.add(state)
        unique_values = list(unique_values_aux)

    options = [{'label': 'Sem Filtro', 'value': 'all'}] + [{'label': val[0].upper() + val[1:], 'value': val} for val in sorted(unique_values)]

    return options