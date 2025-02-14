import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from PIL import Image
import ast
import math

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
        
        font=dict(family="Roboto, sans-serif", size=14, color="black"),
        
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
        x=0.01, y=0.98,
        showarrow=False,
        font=dict(size=14, color="#062a57"),
        align="center",
        bordercolor="#062a57",
        borderwidth=1,
        borderpad=8,
        bgcolor='rgba(255, 255, 255, 0.8)',
    )

# Create empty figure for initialization
def empty_figure(x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor), num_points=0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig_update_layout(fig, num_points, x_range, y_range)
    return fig

# Create empty figure with legend to keep legend even when we only have collapses
def empty_figure_legend(color_df, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor), num_points=0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig_update_layout(fig, num_points, x_range, y_range)

    # Plotting legend
    df_copy = color_df.copy()
    df_copy['cluster_names'] = df_copy['cluster_names'].apply(lambda x: x.capitalize() if isinstance(x, str) else x)
    unique_clusters = df_copy[['color', 'cluster_names']].drop_duplicates()
    unique_clusters = unique_clusters.sort_values(by=['cluster_names'], ascending=True).sort_values(by='cluster_names', key=lambda x: x.str.len())
    legend_order = unique_clusters['cluster_names'].tolist()
    
    color_dict = {}
    for _, row in unique_clusters.iterrows():
        color_dict[str(row['cluster_names'])] = row['color']
    dummy_fig = px.scatter(
        df_copy,
        x=[-100 for i in range(len(df_copy))],
        y=[-100 for i in range(len(df_copy))],
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

# Utility function to create a downward half-circle arc between two rows for the timeline
def make_turn_arc(x0, y0, side="right", radius=0.5, steps=20):
    if side == 'right':
        start_angle = math.pi / 2
        end_angle = -math.pi / 2
    else:
        start_angle = math.pi / 2
        end_angle = 3 * math.pi / 2

    cx = x0
    cy = y0 - 0.5
    angles = np.linspace(start_angle, end_angle, steps)
    arc_pts = []
    for angle in angles:
        arc_x = cx + radius * math.cos(angle)
        arc_y = cy + radius * math.sin(angle)
        arc_pts.append((arc_x, arc_y))

    arc_pts[0] = (x0, y0)
    arc_pts[-1] = (x0, y0 - 1)
    return arc_pts

# Create timeline figure with clickable markers for years 
def timeline_figure(years):

    # Computing zigzag coordinates
    years = -np.sort(-years)
    cols = 8
    n_years = len(years)
    n_rows = math.ceil(n_years/cols)
    line_points = []
    year_points = []
    annotations = []
    idx = 0
    direction = -1
    for row in range(n_rows):
        row_years = min(cols, n_years-row*cols)
        col_indices = range(row_years)
        for c in col_indices:
            if direction == 1:
                x = c+0.2
            else:
                x = cols-1-c-0.2
            y = -row
            year = years[idx]
            line_points.append((x, y))
            year_points.append((x, y, year))

            # Add annotation for the year under the marker
            annotations.append(dict(
                x=x,
                y=y+0.2,
                text=str(year),
                showarrow=False,
                font=dict(weight='bold'),
                xanchor='center',
                yanchor='top'
            ))

            idx += 1

        if row < n_rows - 1:
            end_x = (cols-1) if direction == 1 else 0
            end_y = -row
            side = 'right' if direction == 1 else 'left'
            arc_pts = make_turn_arc(end_x, end_y, side, 0.5, 20)

            line_points.extend(arc_pts[1:])

        direction *= -1

    # Separate out x, y from line_points
    if line_points:
        xs_line, ys_line = zip(*line_points)
    else:
        xs_line, ys_line = [], []

    # Separate out x, y, text from year_points
    if year_points:
        xs_year, ys_year, txt_years = zip(*year_points)
    else:
        xs_year, ys_year, txt_years = [], [], []

    # Plot the timeline as a continuous zigzagging line
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs_line,
        y=ys_line,
        mode='lines',
        line=dict(color='#062a57', width=4),
        hoverinfo='none',
        name='timeline'
    ))

    fig.add_trace(
        go.Scatter(
            x=xs_year,
            y=ys_year,
            mode='markers',
            marker=dict(size=30, color='#062a57', line=dict(width=6, color="#f2f2f2")),
            text=[str(y) for y in txt_years],
            hoverinfo='none',
            name='years'
        )
    )


    # Updating figure layout
    fig.update_layout(
        plot_bgcolor="#f2f2f2",
        paper_bgcolor="#f2f2f2",
        
        font=dict(family="Roboto, sans-serif", size=16, color="black"),
        
        title_x=0.5,
        margin=dict(l=0, r=0, t=0, b=0),
        
        showlegend=False,

        xaxis_title=None,
        yaxis_title=None,
        coloraxis_showscale=False,

        dragmode=None,
        hoverdistance = 5,

        annotations=annotations
    )
 
    fig.update_xaxes(
        range=[-1, 8],
        fixedrange=True,
        visible=False
    )

    fig.update_yaxes(
        range=[-9, 1],
        fixedrange=True,
        visible=False,
        autorange='reversed'
    )

    return fig

# Creating the map of Brazil and plotting markers on states
def brazil_figure():
    fig = px.scatter_mapbox(brazil_states, lat='latitude', lon='longitude', hover_name='state', hover_data={'latitude': False, 'longitude': False}, zoom=3.5, center={'lat': -14.2350, 'lon': -51.9253})

    fig.update_layout(
        mapbox_style="carto-positron",  # options are 'open-street-map', 'stamen-terrain', 'carto-positron', etc.
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Mapa do Brasil e suas Comunidades Indígenas"
    )

    fig.update_traces(marker=dict(size=20, color='#062a57'))

    return fig

# Create scatter plot with markers
def plot_with_markers(df, num_points, color_df, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor)):
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
    fig.update_traces(hoverinfo='none', hovertemplate=None, marker=dict(size=15, opacity=0.65), line=dict(width=0, color='rgba(255, 212, 110, 0.5)'))

    # Updating layout for our standard design
    fig_update_layout(fig, num_points, x_range, y_range)

    # Plotting legend
    df_copy = color_df.copy()
    df_copy['cluster_names'] = df_copy['cluster_names'].apply(lambda x: x.capitalize() if isinstance(x, str) else x)
    unique_clusters = df_copy[['color', 'cluster_names']].drop_duplicates()
    unique_clusters = unique_clusters.sort_values(by=['cluster_names'], ascending=True).sort_values(by='cluster_names', key=lambda x: x.str.len())
    legend_order = unique_clusters['cluster_names'].tolist()
    
    color_dict = {}
    for _, row in unique_clusters.iterrows():
        color_dict[str(row['cluster_names'])] = row['color']
    dummy_fig = px.scatter(
        df_copy,
        x=[-100 for i in range(len(df_copy))],
        y=[-100 for i in range(len(df_copy))],
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
            # dict(source='https://c8.alamy.com/comp/CE5JRA/wikipedia-the-online-encyclopedia-screenshot-CE5JRA.jpg', x=row['x'], y=row['y'], xref="x", yref="y", sizex=(x_range[1]-x_range[0])/8, sizey=(y_range[1]-y_range[0])/8, xanchor="center",yanchor="middle")
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
        if len(cluster) < 5 and min_distance > threshold/8:
            for point_index in cluster:
                labels[point_index] = -1
        else:
            for point_index in cluster:
                labels[point_index] = i
    labels = [labels[i] for i in range(len(points))]

    return labels

# Generating a (fixed) color palette for cluster IDs to handle lazy plotting
def generate_color_map(clusters):
    ids = np.unique(clusters)
    sorted_ids = sorted(ids, key=lambda x: (len(str(x)), str(x).lower()))
    
    cmap = LinearSegmentedColormap.from_list('truncated_cmap', plt.cm.hot(np.linspace(0.0, 0.6, len(sorted_ids))))
    
    color_map = {cluster_id: f'rgba({cmap(i/(len(sorted_ids)-1))[0]*255}, {cmap(i/(len(sorted_ids)-1))[1]*255}, {cmap(i/(len(sorted_ids)-1))[2]*255}, 1)' for i, cluster_id in enumerate(sorted_ids)}

    return color_map

# Function to generically update dataframe when we select a clustering option
def update_cluster_selection(plot_df, selected_df):
    plot_df.set_index('ind_index', inplace=True)
        
    indices = selected_df.index
    plot_df.loc[indices, 'visibility'] = True

    plot_df.loc[indices, "x"] = selected_df['x'].values
    plot_df.loc[indices, "y"] = selected_df['y'].values
    plot_df.loc[indices, "cluster"] = selected_df['cluster'].values
    plot_df.loc[indices, 'cluster_names'] = selected_df['cluster_names'].values
    plot_df.fillna({'cluster_names': ''}, inplace=True)

    color_map = generate_color_map(plot_df.loc[indices, 'cluster_names'].values)
    plot_df.loc[indices, 'color'] = [color_map[label] for label in plot_df.loc[indices, 'cluster_names'].values]

    plot_df.reset_index(inplace=True)

# Function to get unique values of columns with multiple categories at the same time
def get_multi_column(unique_values):
    unique_values_aux = set()
    for item_list in unique_values:
        for item in ast.literal_eval(item_list):
            unique_values_aux.add(item)
    unique_values = list(unique_values_aux)

    return unique_values

# Getting all the dropdown options for a given column
def get_dropdown_options(df, column_name):
    unique_values = df[column_name].dropna().unique().tolist()
    
    # Handling special cases
    if column_name == 'estado_de_origem' or column_name == 'tipo_materia_prima':
        unique_values = get_multi_column(unique_values)

    options = [{'label': 'Sem Filtro', 'value': 'all'}] + [{'label': val[0].upper() + val[1:], 'value': val} for val in sorted(unique_values)]

    return options