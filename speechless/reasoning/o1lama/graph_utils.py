import networkx as nx
import plotly.graph_objects as go
import numpy as np

def bezier_curve(start, end, num_points=20):
    control = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.1)
    t = np.linspace(0, 1, num_points)
    x = (1-t)**2 * start[0] + 2*(1-t)*t * control[0] + t**2 * end[0]
    y = (1-t)**2 * start[1] + 2*(1-t)*t * control[1] + t**2 * end[1]
    return x, y

def plot_graph(G, strongest_edges=None, layout_type='circular'):
    G = G.copy()
    final_answer_node = next((node for node, data in G.nodes(data=True) if "Final Answer" in data['label']), None)
    if final_answer_node:
        G.remove_node(final_answer_node)

    if G.number_of_nodes() == 0:
        # Return an empty figure if there are no nodes
        return go.Figure()

    if layout_type == 'force':
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)

    edge_traces = []
    edge_label_traces = []
    
    # Get min and max weights for scaling
    weights = [edge[2].get('weight', 0) for edge in G.edges(data=True)]
    if weights:
        min_weight, max_weight = min(weights), max(weights)
    else:
        min_weight, max_weight = 0, 1  # Default values if there are no edges

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2].get('weight', 1)
        
        # Scale edge width (minimum 1 pixel, maximum 8 pixels)
        if min_weight != max_weight:
            scaled_width = 1 + 7 * (weight - min_weight) / (max_weight - min_weight)
        else:
            scaled_width = 4  # Default width if all weights are the same
        
        x, y = bezier_curve((x0, y0), (x1, y1))
        
        # Check if this edge is part of the strongest path
        is_strongest = strongest_edges and (edge[0], edge[1]) in strongest_edges
        
        edge_trace = go.Scatter(
            x=x, y=y,
            line=dict(
                width=scaled_width * 4, 
                color='red' if is_strongest else 'rgba(150,150,150,0.5)',
                dash='solid'  # Change this line to always use solid lines
            ),
            hoverinfo='text',
            text=f"Weight: {weight:.2f}",
            mode='lines'
        )
        edge_traces.append(edge_trace)

        # Scale font size between 10px and 18px
        if min_weight != max_weight:
            font_size = 10 + 8 * (weight - min_weight) / (max_weight - min_weight)
        else:
            font_size = 14  # Default font size if all weights are the same
        
        edge_label = go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2 + 0.03],
            text=[f"{weight:.2f}"],
            mode='text',
            textposition='middle center',
            textfont=dict(size=font_size, color='black'),
            hoverinfo='none'
        )
        edge_label_traces.append(edge_label)

    node_sizes = [20 + 5 * G.degree(node) for node in G.nodes()]
    node_colors = ['#000000' if strongest_edges and node in set(sum(strongest_edges, ())) else '#66B2FF' for node in G.nodes()]

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='white'),
            opacity=1,
        ),
        text=[G.nodes[node]['label'] for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=14, color='black', family='Arial', weight='bold')
    )

    data = edge_traces + edge_label_traces + [node_trace]

    fig = go.Figure(data=data,
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600,
                        width=800,
                        plot_bgcolor='rgba(240,240,240,0.5)'
                    ))
    
    fig.update_layout(
        dragmode='pan',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    
    return fig
