
import torch
import asyncio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_model, get_model_predictions


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The tensor to convert.

    Returns:
        np.ndarray: The converted NumPy array.
    """
    return tensor.cpu().detach().numpy()

def make_readable_name(name: str) -> str:
    """
    Converts a cryptic graph name to a more readable format.

    Args:
        name (str): The cryptic name.

    Returns:
        str: The readable name.
    """
    name = name.replace("blocks.", "Block ")
    name = name.replace(".attn.hook_q", " - Attention Q Vector")
    name = name.replace(".attn.hook_k", " - Attention K Vector")
    name = name.replace(".attn.hook_v", " - Attention V Vector")
    name = name.replace(".attn.hook_z", " - Attention Output")
    name = name.replace(".attn.hook_attn_scores", " - Attention Scores")
    name = name.replace(".attn.hook_attn", " - Attention Weights")
    name = name.replace(".mlp.hook_pre", " - MLP Pre-Activation")
    name = name.replace(".mlp.hook_post", " - MLP Post-Activation")
    name = name.replace(".hook_mlp_out", " - MLP Output")
    name = name.replace(".hook_resid_pre", " - Residual Stream Pre")
    name = name.replace(".hook_resid_mid", " - Residual Stream Mid")
    name = name.replace(".hook_resid_post", " - Residual Stream Post")
    name = name.replace(".ln1.hook_scale", " - LayerNorm1 Scale")
    name = name.replace(".ln1.hook_normalized", " - LayerNorm1 Normalized")
    name = name.replace(".ln2.hook_scale", " - LayerNorm2 Scale")
    name = name.replace(".ln2.hook_normalized", " - LayerNorm2 Normalized")
    name = name.replace("hook_embed", "Token Embeddings")
    name = name.replace("hook_pos_embed", "Position Embeddings")
    return name



def visualize_model(query: str, block_number: int, temperature: float, seed: int, top_p: float) -> dict:
    model = load_model()
    tokens = model.to_tokens(query)
    token_strings = model.to_str_tokens(tokens)
    _, cache = model.run_with_cache(tokens)

    # Get model predictions
    predictions = get_model_predictions(model, tokens, temperature, seed, top_p)

    # Prepare prediction DataFrame
    prediction_df = pd.DataFrame(predictions, columns=["Token", "Probability"])
    prediction_df.index = range(1, len(prediction_df) + 1)  # Start index from 1

    # Prepare prediction plot
    prediction_plot = plot_predictions(predictions)

    # Initialize output dictionary
    output = {
        "prediction_plot": prediction_plot,
        "plots": {
            "Embeddings": [],
            "Residual Streams": [],
            "LayerNorm1": [],
            "Attention": [],
            "LayerNorm2": [],
            "MLP": []
        }
    }

    # Visualize Embeddings
    embed_types = [
        "hook_embed",
        "hook_pos_embed"
    ]
    for embed_type in embed_types:
        if embed_type in cache:
            data = cache[embed_type][0]
            readable_title = make_readable_name(embed_type)
            stem_fig = plot_stem_embeddings(data, readable_title, token_strings)
            output["plots"]["Embeddings"].append(stem_fig)

    # Visualize Residual Streams
    residual_types = [
        f"blocks.{block_number}.hook_resid_pre",
        f"blocks.{block_number}.hook_resid_mid",
        f"blocks.{block_number}.hook_resid_post"
    ]
    for residual_type in residual_types:
        if residual_type in cache:
            data = cache[residual_type][0]
            readable_title = make_readable_name(residual_type)
            heatmap_fig = plot_heatmap(data, readable_title, token_strings)
            output["plots"]["Residual Streams"].append(heatmap_fig)

    # Visualize LayerNorm1 Scales and Normalized Outputs
    ln1_scales = [
        f"blocks.{block_number}.ln1.hook_scale"
    ]
    for ln_scale in ln1_scales:
        if ln_scale in cache:
            data = cache[ln_scale][0]
            readable_title = make_readable_name(ln_scale)
            line_fig = plot_line(data, readable_title, token_strings)
            output["plots"]["LayerNorm1"].append(line_fig)

    ln1_outputs = [
        f"blocks.{block_number}.ln1.hook_normalized"
    ]
    for ln_output in ln1_outputs:
        if ln_output in cache:
            data = cache[ln_output][0]
            readable_title = make_readable_name(ln_output)
            heatmap_fig = plot_heatmap(data, readable_title, token_strings)
            output["plots"]["LayerNorm1"].append(heatmap_fig)

    # Visualize Attention Components
    attn_components = [
        f"blocks.{block_number}.attn.hook_q",
        f"blocks.{block_number}.attn.hook_k",
        f"blocks.{block_number}.attn.hook_v",
        f"blocks.{block_number}.attn.hook_z",
        f"blocks.{block_number}.attn.hook_attn_scores",
        f"blocks.{block_number}.attn.hook_attn"
    ]
    for attn_component in attn_components:
        if attn_component in cache:
            data = cache[attn_component][0]
            readable_title = make_readable_name(attn_component)
            attention_fig = plot_attention(data, readable_title, token_strings)
            output["plots"]["Attention"].append(attention_fig)


    # Visualize Attention Scores and Probabilities
    attn_matrices = [
        f"blocks.{block_number}.attn.hook_attn_scores",
        f"blocks.{block_number}.attn.hook_attn"
    ]
    for attn_matrix in attn_matrices:
        if attn_matrix in cache:
            data = cache[attn_matrix][0]
            readable_title = make_readable_name(attn_matrix)
            attention_fig = plot_attention(data, readable_title, token_strings)
            output["plots"]["Attention"].append(attention_fig)

    # Visualize LayerNorm2 Scales and Normalized Outputs
    ln2_scales = [
        f"blocks.{block_number}.ln2.hook_scale"
    ]
    for ln_scale in ln2_scales:
        if ln_scale in cache:
            data = cache[ln_scale][0]
            readable_title = make_readable_name(ln_scale)
            line_fig = plot_line(data, readable_title, token_strings)
            output["plots"]["LayerNorm2"].append(line_fig)

    ln2_outputs = [
        f"blocks.{block_number}.ln2.hook_normalized"
    ]
    for ln_output in ln2_outputs:
        if ln_output in cache:
            data = cache[ln_output][0]
            readable_title = make_readable_name(ln_output)
            heatmap_fig = plot_heatmap(data, readable_title, token_strings)
            output["plots"]["LayerNorm2"].append(heatmap_fig)

    # Visualize MLP Activations
    mlp_activations = [
        f"blocks.{block_number}.mlp.hook_pre",
        f"blocks.{block_number}.mlp.hook_post",
        f"blocks.{block_number}.hook_mlp_out"
    ]
    for mlp_activation in mlp_activations:
        if mlp_activation in cache:
            data = cache[mlp_activation][0]
            readable_title = make_readable_name(mlp_activation)
            heatmap_fig = plot_heatmap(data, readable_title, token_strings)
            hist_fig = plot_histogram(data, readable_title + " Histogram")
            output["plots"]["MLP"].append(heatmap_fig)
            output["plots"]["MLP"].append(hist_fig)

    return output

async def async_visualize_model(query, block_number, temperature, seed, top_p):
    return await asyncio.to_thread(visualize_model, query, block_number, temperature, seed, top_p)


def plot_heatmap(data: torch.Tensor, title: str, tokens: list = None) -> go.Figure:
    """
    Creates a heatmap using plotly with clear visible lines separating each row (token).
    Uses a more saturated colorscale for better visibility.

    Args:
        data (torch.Tensor): The data to plot.
        title (str): The title of the plot.
        tokens (list, optional): Labels for the y-axis. Defaults to None.

    Returns:
        go.Figure: The plotly figure.
    """
    data_np = to_numpy(data)
    num_tokens, num_dims = data_np.shape

    # Create a custom colorscale with more saturated colors
    colorscale = [
        [0, "rgb(0,0,255)"],      # Saturated blue for negative values
        [0.25, "rgb(100,100,255)"],  # Light blue
        [0.5, "rgb(255,255,255)"],   # White for zero
        [0.75, "rgb(255,100,100)"],  # Light red
        [1, "rgb(255,0,0)"]       # Saturated red for positive values
    ]

    fig = go.Figure(data=go.Heatmap(
        z=data_np,
        x=[f"Dim {i}" for i in range(num_dims)],
        y=tokens if tokens is not None else [f"Token {i}" for i in range(num_tokens)],
        colorscale=colorscale,
        zmin=-np.max(np.abs(data_np)),  # Symmetrical color scale
        zmax=np.max(np.abs(data_np)),
    ))

    # Add horizontal lines between rows
    for i in range(1, num_tokens):
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=i - 0.5,
            x1=num_dims - 0.5,
            y1=i - 0.5,
            line=dict(color="black", width=1),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Dimensions",
        yaxis_title="Tokens" if tokens is not None else "",
        xaxis_nticks=36,
        height=max(300, num_tokens * 30),  # Adjust height based on number of tokens
    )

    return fig

def plot_stem_embeddings(data: torch.Tensor, title: str, tokens: list = None) -> go.Figure:
    """
    Creates individual stem plots for each token's embedding.

    Args:
        data (torch.Tensor): The embedding data to plot.
        title (str): The title of the plot.
        tokens (list, optional): List of token strings. Defaults to None.

    Returns:
        go.Figure: The plotly figure containing subplots of stem plots.
    """
    data_np = to_numpy(data)
    num_tokens, num_dims = data_np.shape
    
    # Create subplots
    fig = make_subplots(rows=num_tokens, cols=1, subplot_titles=[f"Token: {tokens[i] if tokens else i}" for i in range(num_tokens)])
    
    for i in range(num_tokens):
        fig.add_trace(
            go.Scatter(
                x=list(range(num_dims)),
                y=data_np[i],
                mode='markers+lines',
                name=f'Token {i}',
                showlegend=False
            ),
            row=i+1, col=1
        )
        
        # Update y-axis for each subplot
        fig.update_yaxes(title_text="Value", row=i+1, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=300 * num_tokens,  # Adjust height based on number of tokens
        xaxis_title="Embedding Dimension",
        showlegend=False,
    )
    
    # Update x-axis for all subplots
    fig.update_xaxes(title_text="Embedding Dimension")
    
    return fig

def plot_line(data: torch.Tensor, title: str, x_labels: list = None) -> go.Figure:
    data_np = to_numpy(data).squeeze()
    x = x_labels if x_labels else list(range(len(data_np)))
    
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=data_np,
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Token" if x_labels else "Position",
        yaxis_title="Value",
        autosize=True,
        height=400,
        width=None,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly",
    )
    
    fig.update_xaxes(automargin=True, constrain="domain")
    fig.update_yaxes(automargin=True, constrain="domain")
    
    return fig

def plot_histogram(data: torch.Tensor, title: str) -> go.Figure:
    data_np = to_numpy(data).flatten()
    
    fig = go.Figure(data=go.Histogram(
        x=data_np,
        nbinsx=50,
        marker_color='skyblue',
        opacity=0.75
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Activation Value",
        yaxis_title="Frequency",
        autosize=True,
        height=400,
        width=None,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly",
    )
    
    fig.update_xaxes(automargin=True, constrain="domain")
    fig.update_yaxes(automargin=True, constrain="domain")
    
    return fig

def plot_attention(data: torch.Tensor, title: str, tokens: list) -> go.Figure:
    data_np = to_numpy(data)
    
    if len(data_np.shape) == 2:
        fig = go.Figure(data=go.Heatmap(
            z=data_np,
            x=tokens,
            y=tokens,
            colorscale='RdBu',
            reversescale=True
        ))
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            title_y=0.95,
            xaxis_title="Keys",
            yaxis_title="Queries",
            height=600,
            width=800,
            autosize=True,  
            margin=dict(l=50, r=50, t=50, b=50),
            template="plotly",
            uirevision=True
        )
        
        fig.update_xaxes(
            automargin=True,
            scaleanchor="y",
            constrain="domain"
        )
        
        fig.update_yaxes(
            automargin=True,
            scaleanchor="x",
            constrain="domain"
        )
        
        return fig
    
    elif len(data_np.shape) == 3 or len(data_np.shape) == 4:
        if len(data_np.shape) == 4:
            data_np = data_np[0]
        num_heads = data_np.shape[0]
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        # Calculate dynamic spacing based on number of rows
        vertical_spacing = min(0.3, 1.0 / (rows + 1))
        horizontal_spacing = min(0.2, 1.0 / (cols + 1))
        
        # Calculate figure dimensions
        height_per_row = 300
        width_per_col = 250
        total_height = max(600, rows * height_per_row)
        total_width = max(800, cols * width_per_col)
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            subplot_titles=[f"Head {i}" for i in range(num_heads)]
        )
        
        for i in range(num_heads):
            row = i // cols + 1
            col = i % cols + 1
            
            # Calculate appropriate tick frequency
            tick_freq = max(1, len(tokens) // 6)  # Show at most 6 ticks
            
            # Shorten token labels if needed
            max_len = 10
            shortened_tokens = [t[:max_len] + '...' if len(t) > max_len else t 
                              for t in tokens[::tick_freq]]
            tick_positions = list(range(0, len(tokens), tick_freq))
            
            fig.add_trace(
                go.Heatmap(
                    z=data_np[i],
                    x=tokens,
                    y=tokens,
                    colorscale='RdBu',
                    reversescale=True,
                    showscale=(col == cols and row == 1)  # Show colorbar only for first row, last column
                ),
                row=row,
                col=col
            )
            
            fig.update_xaxes(
                title_text="Keys",
                row=row,
                col=col,
                side="bottom",
                ticktext=shortened_tokens,
                tickvals=tick_positions,
                tickangle=45,
                tickfont=dict(size=8),
                title_font=dict(size=10),
                title_standoff=25
            )
            
            fig.update_yaxes(
                title_text="Queries",
                row=row,
                col=col,
                ticktext=shortened_tokens,
                tickvals=tick_positions,
                tickfont=dict(size=8),
                title_font=dict(size=10),
                title_standoff=25
            )
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                y=0.98,
                font=dict(size=14)
            ),
            autosize=True,      # Enable autosizing
            height=total_height,
            width=total_width,
            showlegend=False,
            margin=dict(l=50, r=100, t=100, b=80),  # Increased margins
            template="plotly"
        )
        
        # Adjust colorbar
        if num_heads > 1:
            fig.update_layout(autosize=True,      # Enable autosizing
                coloraxis=dict(colorbar=dict(
                title="Attention Score",
                lenmode="fraction",
                len=0.75,
                yanchor="middle",
                y=0.5,
                xanchor="right",
                x=1.02,
                title_font=dict(size=10),
                tickfont=dict(size=8)
            )))
        
        return fig
    
def plot_predictions(predictions: list) -> go.Figure:
    tokens = [token for token, prob in predictions]
    probs = [prob for token, prob in predictions]
    
    fig = go.Figure(data=go.Bar(
        x=tokens,
        y=probs,
        marker_color='skyblue'
    ))
    
    fig.update_layout(
        title="Top 10 Predictions",
        xaxis_title="Token",
        yaxis_title="Probability",
        autosize=True,
        height=400,
        width=None,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly",
    )
    
    fig.update_xaxes(automargin=True, constrain="domain")
    fig.update_yaxes(automargin=True, constrain="domain")
    
    return fig
