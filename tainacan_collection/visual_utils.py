import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from PIL import Image
import ast
import math
from datetime import timedelta

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

# Loading and processing geolocation dataframe to increase granularity of map tab
ind_geo = pd.read_csv('data/terras_indigenas_geolocation_filtered.csv', index_col='id')
ind_geo['povo'] = ind_geo['povo'].str.capitalize()

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
        hoverdistance=6,
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
def empty_figure(x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor), num_points=0, only_images=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig_update_layout(fig, num_points, x_range, y_range)

    if only_images:
        # Observation for image option
        fig.add_annotation(
            text="Apenas itens com imagens são exibidos nesta opção",
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

# Create empty figure with legend to keep legend even when we only have collapses
def empty_figure_legend(color_df, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor), num_points=0, only_images=True):
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
        x=[-10000 for i in range(len(df_copy))],
        y=[-10000 for i in range(len(df_copy))],
        color='cluster_names',
        labels={'cluster_names': 'Cluster Names'},
        color_discrete_map=color_dict,
        category_orders={'cluster_names': legend_order}
    )
    for trace in dummy_fig.data:
        trace.showlegend = True
        trace.marker.size = 15
        fig.add_trace(trace)

    if only_images:
        # Observation for image option
        fig.add_annotation(
            text="Apenas itens com imagens são exibidos nesta opção",
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

# Create scatter plot with markers
def plot_with_markers(df, num_points, color_df, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor), only_images=True, no_legend=False):
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
    if not no_legend:
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
            x=[-10000 for i in range(len(df_copy))],
            y=[-10000 for i in range(len(df_copy))],
            color='cluster_names',
            labels={'cluster_names': 'Cluster Names'},
            color_discrete_map=color_dict,
            category_orders={'cluster_names': legend_order}
        )
        for trace in dummy_fig.data:
            trace.showlegend = True
            trace.marker.size = 15
            fig.add_trace(trace)

    if only_images:
        # Observation for image option
        fig.add_annotation(
            text="Apenas itens com imagens são exibidos nesta opção",
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

# Create scatter plot with the images themselves
def plot_with_images(df, num_points, color_df, x_range=(-norm_factor,norm_factor), y_range=(-norm_factor,norm_factor), no_legend=False):
    df = df.loc[df['image_path_br'] != 'data/placeholder_square.png']
    fig = go.Figure()
    for index, row in df.iterrows():
        fig.add_layout_image(
            # dict(source=Image.open(row['image_path_br']), x=row['x'], y=row['y'], xref="x", yref="y", sizex=(x_range[1]-x_range[0])/8, sizey=(y_range[1]-y_range[0])/8, xanchor="center",yanchor="middle")
            dict(source=row['temporary_br_url'], x=row['x'], y=row['y'], xref="x", yref="y", sizex=(x_range[1]-x_range[0])/12, sizey=(y_range[1]-y_range[0])/12, xanchor="center",yanchor="middle")
        )
    
    fig_update_layout(fig, num_points, x_range, y_range)

    # Plotting legend
    if not no_legend:
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
            x=[-10000 for i in range(len(df_copy))],
            y=[-10000 for i in range(len(df_copy))],
            color='cluster_names',
            labels={'cluster_names': 'Cluster Names'},
            color_discrete_map=color_dict,
            category_orders={'cluster_names': legend_order}
        )
        for trace in dummy_fig.data:
            trace.showlegend = True
            trace.marker.size = 15
            fig.add_trace(trace)

    # Observation for image option
    fig.add_annotation(
        text="Apenas itens com imagens são exibidos nesta opção",
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
def timeline_figure_zigzag(df_years):

    # Getting counts to compute proper marker sizes
    counts = df_years.value_counts()
    key_sorted_counts = counts.sort_index(ascending=False)
    year_counts = np.array(key_sorted_counts.tolist())
    marker_sizes = np.maximum(25, 1.3*np.sqrt(year_counts))
    line_sizes = np.maximum(5, 0.2*np.sqrt(year_counts))

    # Extracting unique years
    years = df_years.dropna().unique().astype(np.int16)

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
                x = c-0.2
            else:
                x = cols-1-c+0.2
            y = -row
            year = years[idx]
            line_points.append((x, y))
            year_points.append((x, y, year))

            # Add annotation for the year under the marker
            annotations.append(dict(
                x=x,
                y=y+0.32,
                text=str(year),
                showarrow=False,
                font=dict(size=18, weight='bold'),
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
            marker=dict(size=marker_sizes, color='#062a57', opacity=1, line=dict(width=line_sizes, color="#f2f2f2")),
            text=[str(y) for y in txt_years],
            hoverinfo='none',
            showlegend=False,
            name='years'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xs_year,
            y=ys_year,
            mode='text',
            text=['' for _ in year_counts],
            textfont=dict(size=16, weight='bold', color="#f2f2f2"),
            opacity=1,
            hoverinfo='none',
            showlegend=False,
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
        hoverdistance=5,

        annotations=annotations
    )
 
    fig.update_xaxes(
        range=[-1, 8],
        fixedrange=True,
        visible=False
    )

    fig.update_yaxes(
        range=[-10, 1],
        fixedrange=True,
        visible=False,
        autorange='reversed'
    )

    return fig

# Create timeline figure with clickable markers for years 
def timeline_figure_grid(df, page_size=90):
    # Filtering out wrong data and sorting dataframe by 'data_de_aquisicao'
    grid_df = df.copy()
    grid_df = grid_df.sort_values(by='data_de_aquisicao')
    df_indices = list(grid_df.index)

    # Generating dynamic shaped and marker size grid for aspect ratio 4:3
    num_points = len(grid_df)
    n_pages = math.ceil(num_points/page_size)

    ar_unit = math.sqrt(page_size/7)
    num_rows, num_cols = math.floor(3*ar_unit), math.ceil(4*ar_unit)
    
    x_coords, y_coords = np.linspace(0, 7.5, num_cols), np.linspace(0.5, -7.5, num_rows)
    X, Y = np.meshgrid(x_coords, y_coords)
    X = X.ravel()[:min(num_points, page_size)]
    Y = Y.ravel()[:min(num_points, page_size)]

    marker_size = 35

    # Generating histogram data
    num_no_date = len(grid_df.loc[grid_df['data_de_aquisicao'] == '0001-01-01'])
    hist = {'Sem Data<br>Exata': num_no_date, 'Janeiro': 0, 'Fevereiro': 0, 'Março': 0, 'Abril': 0, 'Maio': 0, 'Junho': 0, 'Julho': 0, 'Agosto': 0, 'Setembro': 0, 'Outubro': 0, 'Novembro': 0, 'Dezembro': 0}
    grid_df.loc[grid_df['data_de_aquisicao'] == '0001-01-01', 'data_de_aquisicao'] = 0

    num_to_month = {0: 'Sem Data<br>Exata', 1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto', 9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}
    months = grid_df.loc[grid_df['data_de_aquisicao'] != 0, 'data_de_aquisicao'] = pd.to_datetime(grid_df.loc[grid_df['data_de_aquisicao'] != 0, 'data_de_aquisicao']).dt.month
    for month, count in dict(sorted(months.value_counts().to_dict().items())).items():
        hist[num_to_month[month]] = count

    # Generating colormaps
    cmap = LinearSegmentedColormap.from_list('truncated_cmap', plt.cm.hot(np.linspace(0.0, 0.6, len(hist))))
    month_colors = {month: f'rgba({int(cmap(i/(len(hist)-1))[0]*255)}, {int(cmap(i/(len(hist)-1))[1]*255)}, {int(cmap(i/(len(hist)-1))[2]*255)}, 1)' for i, month in enumerate(hist.keys())}
    
    colors = [month_colors['Sem Data<br>Exata'] for i in range(len(grid_df.loc[grid_df['data_de_aquisicao'] == 0]))]
    for _, month in months.items():
        colors.append(month_colors[num_to_month[month]])

    square_colors = [color.replace('1)', '0)') for color in colors]
    
    fig = go.Figure()

    # Adding Year annotation
    fig.add_annotation(
        x=-0.64,
        y=0.4,
        text=str(grid_df['ano_de_aquisicao'].unique()[-1]),
        showarrow=False,
        font=dict(size=32, color="#062a57", weight='bold'),
        align="center",
    )

    # Creating back arrow button
    fig.add_trace(go.Scatter(
        x=[-0.8, -0.57],
        y=[-0.5, -0.5],
        mode='markers',
        marker=dict(size=25, color='#062a57', symbol=['arrow-left', 'line-ew'], line=dict(width=7, color='#062a57')),
        hoverinfo='none',
        name='back_button'
    ))
    fig.add_annotation(
        x=-0.62,
        y=-0.85,
        text="Voltar",
        showarrow=False,
        font=dict(size=18, color="#062a57", weight='bold'),
        align="center",
    )

    # Creating pages
    image_pages = []
    for p in range(n_pages):
        # Indexing page and elements in it
        start, end = p*page_size, (p+1)*page_size
        try:
            page_df = grid_df.iloc[start:end]
        except:
            page_df = grid_df.iloc[start:]

        # Creating grid figure with images
        images = []
        for i, (_, row) in enumerate(page_df.iterrows()):
            images.append(dict(source=row['temporary_br_url'], x=X[i], y=Y[i], xref="x", yref="y", sizex=0.6, sizey=0.6, xanchor="center",yanchor="middle"))
            # fig.add_layout_image(
            #     dict(source=row['temporary_br_url'], x=X[i], y=Y[i], xref="x", yref="y", sizex=5/math.sqrt(num_points), sizey=5/math.sqrt(num_points), xanchor="center",yanchor="middle")
            # )
        image_pages.append(images)

        # Dealing with the last page corner case
        if p == n_pages-1:
            X = X[:len(page_df)]
            Y = Y[:len(page_df)]
            colors_aux = colors[start:start+len(page_df)]
            square_colors_aux = square_colors[start:start+len(page_df)]

        # Dealing with colors in general
        else:
            colors_aux = colors[start:end]
            square_colors_aux = square_colors[start:end]

        # Creating color "legend" for images
        fig.add_trace(go.Scatter(
            x=X,
            y=Y-2.9/math.sqrt(num_points/n_pages),
            mode='markers',
            customdata=df_indices[start:end],
            marker=dict(size=marker_size, color=colors_aux, symbol='line-ew', line=dict(width=5, color=colors_aux)),
            hoverinfo='none',
            name=f'year_timeline_{p}',
            visible=(p==0)
        ))

        # Creating invisible squares to better highlight images later
        fig.add_trace(go.Scatter(
            x=X,
            y=Y,
            mode='markers',
            marker=dict(size=marker_size, color=square_colors_aux, symbol='square-open', line=dict(width=4)),
            hoverinfo='none',
            name=f'square_year_timeline_{p}',
            visible=(p==0)
        ))
    
    # Creating visibility mask: making back button and histogram always visible, and legend and square traces visible depending on the slider step (chosen page)
    steps = [] 
    for p in range(n_pages):
        mask = [True]
        for i in range(n_pages):
            mask.append(i==p)
            mask.append(i==p)
        mask.append(True)

        steps.append(dict(
            label=str(p+1),
            method="update",
            args=[
                {"visible": mask},
                {"images": image_pages[p]}
            ]
        ))

    # Creating histogram
    fig.add_trace(go.Bar(
        x=list(hist.keys()),
        y=list(hist.values()),
        marker=dict(color=list(month_colors.values())),
        name='year_histogram',
        xaxis='x2',
        yaxis='y2',
        hoverinfo='y',
        hoverlabel=dict(
            font=dict(
                weight='bold',
                size=22,
                color='#f2f2f2'
            ),
        )
    ))

    # Adding annotation with total number of points on the top‐right of the histogram
    total = sum(hist.values())
    fig.add_annotation(
        text=f"Número total de itens: {total}",
        xref="x2 domain",
        yref="y2 domain",
        x=1, y=1,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        font=dict(size=14, color="#062a57"),
        align="right",
        bordercolor="#062a57",
        borderwidth=1,
        borderpad=8,
        bgcolor='rgba(255, 255, 255, 0.8)',
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

        # X and Y for grid
        xaxis=dict(
            range=[-1, 8],
            fixedrange=True,
            visible=False,
            domain=[0, 1]
        ),
        yaxis=dict(
            range=[-9, 1.1],
            fixedrange=True,
            visible=False,
            domain=[0, 1]
        ),

        # X and Y for histogram
        xaxis2=dict(
            domain=[0.1, 0.95],
            anchor='y2',
            visible=True,
            tickangle=-45
        ),
        yaxis2=dict(
            domain=[0.05, 0.38],
            anchor='x2',
            visible=True
        ),

        # Starting with page 0 for images
        images=image_pages[0],

        # # Adding slider for grid pagination
        sliders=[dict(
            active=0,
            pad={"t": 10},
            currentvalue=dict(visible=False),
            # currentvalue=dict(
            #     visible=True,
            #     prefix="Página ", suffix="",
            #     xanchor="center",
            #     offset=5,
            #     font=dict(
            #         size=20,
            #         color="#062a57",
            #         family="Roboto, sans-serif",
            #         weight="bold"
            #     )
            # ),
            steps=steps,
            x=0.52, xanchor="center", y=0.39, yanchor='bottom', len=0.8,
            bgcolor="#e3e3e3",
            bordercolor="#062a57",
            activebgcolor="#062a57",
            borderwidth=1.5,
            font=dict(color="#f2f2f2", size=14),
            ticklen=0,
        )],
    )

    return fig

# Creating the map of Brazil and plotting markers on states
def brazil_figure():
    fig = px.scatter_mapbox(brazil_states, lat='latitude', lon='longitude', hover_name='state', hover_data={'latitude': False, 'longitude': False}, zoom=3.5, center={'lat': -14.2350, 'lon': -51.9253})

    fig.update_layout(
        mapbox_style="carto-positron",  # options are 'open-street-map', 'stamen-terrain', 'carto-positron', etc.
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Mapa do Brasil e suas Comunidades Indígenas",
        hoverdistance=8
    )

    fig.update_traces(marker=dict(size=22, color='#062a57', opacity=0.8), hoverlabel=dict(font=dict(size=22)))

    fig.add_trace(go.Scattermapbox(lat=ind_geo['x'], lon=ind_geo['y'], mode='markers', marker=dict(size=16, color='#9b3636', opacity=0.8), hovertext=ind_geo['povo'], hoverinfo='text', showlegend=False, hoverlabel=dict(font=dict(size=16, weight='bold'))))

    return fig

########## GENERAL PURPOSE UTIL FUNCTIONS ##########
# Function to access image on he bucket and create a temporary signed URL
def generate_signed_url(storage_client, bucket_name, blob_name, expiration_minutes=1):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version='v4',
        expiration=timedelta(minutes=expiration_minutes),
        method='GET'
    )
    return url

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
        if len(cluster) < 5 and min_distance > threshold/2:
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
def update_cluster_selection(plot_df, selected_df, no_clusters=False):
    plot_df.set_index('ind_index', inplace=True)
        
    indices = selected_df.index
    plot_df.loc[indices, 'visibility'] = True

    plot_df.loc[indices, "x"] = selected_df['x'].values
    plot_df.loc[indices, "y"] = selected_df['y'].values
    
    if no_clusters:
        plot_df.loc[indices, "cluster"] = 1
        plot_df.loc[indices, 'cluster_names'] = 'no cluster'
        plot_df.fillna({'cluster_names': ''}, inplace=True)
        plot_df.loc[indices, 'color'] = 'rgba(255, 113, 0, 1)'

    else:
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

    options = [{'label': val[0].upper() + val[1:], 'value': val} for val in sorted(unique_values)]

    return options