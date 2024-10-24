import gradio as gr
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA
from functools import lru_cache
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from graphviz import Digraph
import os
import base64

# Update the custom_css variable to load from the file
with open(os.path.join('static', 'css', 'style.css'), 'r') as f:
    custom_css = f.read()

@lru_cache(maxsize=1)
def load_model() -> HookedTransformer:
    """
    Loads the distilgpt2 model using the transformer_lens HookedTransformer.

    Returns:
        HookedTransformer: The loaded distilgpt2 model.
    """
    return HookedTransformer.from_pretrained("distilgpt2")


def generate_computation_graph_svg(model):
    dot = Digraph(comment='Styled DistilGPT2 Compute Graph')
    dot.attr(rankdir='TB', nodesep='0.05', ranksep='0.2')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='15')
    dot.attr('edge', fontname='Arial', fontsize='15')

    # Model configuration
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads
    d_head = d_model // n_heads
    d_ff = model.cfg.d_mlp
    vocab_size = model.cfg.d_vocab
    seq_len = 1024  # Assuming max sequence length

    # Enhanced color scheme
    colors = {
        'input': '#E6F3FF',
        'embed': '#CCEBFF',
        'pos_encoding': '#FFD700',
        'ln': '#FFA07A',  # Light Salmon for Layer Norm
        'proj': '#98FB98',
        'split': '#87CEFA',
        'attn': '#E6FFCC',
        'scale': '#F0E68C',  # Khaki for Scale operation
        'ffn': '#FFD9CC',
        'gelu': '#DDA0DD',
        'output': '#F0E6FF',
        'add': '#F0E6FF',
        'matmul': '#FFFACD',
        'softmax': '#FFB6C1',
        'subgraph_input': '#F0F8FF',
        'subgraph_attn': '#F0FFF0',
        'subgraph_ffn': '#FFF0F5',
        'subgraph_output': '#F5F5F5'
    }

    def node_label(name, size):
        return f'<<B><I>{name}</I></B><BR/><FONT FACE="Courier" POINT-SIZE="10">{size}</FONT>>'

    # Input and Embedding
    with dot.subgraph(name='cluster_input') as c:
        c.attr(rank='same', style='filled,rounded', fillcolor=colors['subgraph_input'], color='none')
        c.node('input', node_label('Input Tokens', f'(seq_len)'), fillcolor=colors['input'])
        c.node('token_embed', node_label('Token Embedding', f'(seq_len, {d_model})'), fillcolor=colors['embed'])
        c.node('pos_encoding', node_label('Positional Encoding', f'(seq_len, {d_model})'), fillcolor=colors['pos_encoding'])
        c.node('embed_add', '+', shape='circle', fillcolor=colors['add'])
        c.node('embed', node_label('Token + Position Embedding', f'(seq_len, {d_model})'), fillcolor=colors['embed'])

    dot.edge('input', 'token_embed')
    dot.edge('token_embed', 'embed_add')
    dot.edge('pos_encoding', 'embed_add')
    dot.edge('embed_add', 'embed')

    last_node = 'embed'

    # Main loop for transformer blocks
    for i in range(model.cfg.n_layers):
        with dot.subgraph(name=f'cluster_block_{i}') as c:
            c.attr(style='filled,rounded', fillcolor=colors['subgraph_attn'], color='none')
            
            c.node(f'ln1_{i}', node_label('Layer Norm 1', f'(seq_len, {d_model})'), fillcolor=colors['ln'])
            
            with c.subgraph(name=f'cluster_qkv_{i}') as qkv:
                qkv.attr(rank='same')
                qkv.node(f'q_proj_{i}', node_label('Q Projection', f'(seq_len, {d_model})'), fillcolor=colors['proj'])
                qkv.node(f'k_proj_{i}', node_label('K Projection', f'(seq_len, {d_model})'), fillcolor=colors['proj'])
                qkv.node(f'v_proj_{i}', node_label('V Projection', f'(seq_len, {d_model})'), fillcolor=colors['proj'])

            with c.subgraph(name=f'cluster_split_{i}') as split:
                split.attr(rank='same')
                split.node(f'q_split_{i}', node_label('Split Q', f'({n_heads}, seq_len, {d_head})'), fillcolor=colors['split'])
                split.node(f'k_split_{i}', node_label('Split K', f'({n_heads}, seq_len, {d_head})'), fillcolor=colors['split'])
                split.node(f'v_split_{i}', node_label('Split V', f'({n_heads}, seq_len, {d_head})'), fillcolor=colors['split'])

            c.node(f'qk_matmul_{i}', node_label('Q·K^T', f'({n_heads}, seq_len, seq_len)'), fillcolor=colors['matmul'])
            c.node(f'qk_scale_{i}', node_label('Scale', f'({n_heads}, seq_len, seq_len)'), fillcolor=colors['scale'])
            c.node(f'attn_softmax_{i}', node_label('Softmax', f'({n_heads}, seq_len, seq_len)'), fillcolor=colors['softmax'])
            c.node(f'attn_v_matmul_{i}', node_label('·V', f'({n_heads}, seq_len, {d_head})'), fillcolor=colors['matmul'])
            c.node(f'attn_concat_{i}', node_label('Concat Heads', f'(seq_len, {d_model})'), fillcolor=colors['attn'])
            c.node(f'attn_out_{i}', node_label('Attention Output', f'(seq_len, {d_model})'), fillcolor=colors['attn'])

            c.node(f'add1_{i}', '+', shape='circle', fillcolor=colors['add'])
            c.node(f'ln2_{i}', node_label('Layer Norm 2', f'(seq_len, {d_model})'), fillcolor=colors['ln'])

        with dot.subgraph(name=f'cluster_ffn_{i}') as ffn:
            ffn.attr(style='filled,rounded', fillcolor=colors['subgraph_ffn'], color='none')
            ffn.node(f'ff1_{i}', node_label('Linear 1', f'(seq_len, {d_ff})'), fillcolor=colors['ffn'])
            ffn.node(f'gelu_{i}', node_label('GELU', f'(seq_len, {d_ff})'), fillcolor=colors['gelu'])
            ffn.node(f'ff2_{i}', node_label('Linear 2', f'(seq_len, {d_model})'), fillcolor=colors['ffn'])
            ffn.node(f'add2_{i}', '+', shape='circle', fillcolor=colors['add'])

        # Connect nodes within the block
        dot.edge(last_node, f'ln1_{i}')
        dot.edge(f'ln1_{i}', f'q_proj_{i}')
        dot.edge(f'ln1_{i}', f'k_proj_{i}')
        dot.edge(f'ln1_{i}', f'v_proj_{i}')
        dot.edge(f'q_proj_{i}', f'q_split_{i}')
        dot.edge(f'k_proj_{i}', f'k_split_{i}')
        dot.edge(f'v_proj_{i}', f'v_split_{i}')
        dot.edge(f'q_split_{i}', f'qk_matmul_{i}')
        dot.edge(f'k_split_{i}', f'qk_matmul_{i}')
        dot.edge(f'qk_matmul_{i}', f'qk_scale_{i}')
        dot.edge(f'qk_scale_{i}', f'attn_softmax_{i}')
        dot.edge(f'attn_softmax_{i}', f'attn_v_matmul_{i}')
        dot.edge(f'v_split_{i}', f'attn_v_matmul_{i}')
        dot.edge(f'attn_v_matmul_{i}', f'attn_concat_{i}')
        dot.edge(f'attn_concat_{i}', f'attn_out_{i}')
        dot.edge(f'attn_out_{i}', f'add1_{i}')
        dot.edge(last_node, f'add1_{i}', color='red', style='dashed')
        dot.edge(f'add1_{i}', f'ln2_{i}')
        dot.edge(f'ln2_{i}', f'ff1_{i}')
        dot.edge(f'ff1_{i}', f'gelu_{i}')
        dot.edge(f'gelu_{i}', f'ff2_{i}')
        dot.edge(f'ff2_{i}', f'add2_{i}')
        dot.edge(f'ln2_{i}', f'add2_{i}', color='red', style='dashed')

        # Add invisible edge between softmax and Linear 1
        dot.edge(f'attn_softmax_{i}', f'ff1_{i}', style='invis')
        
        # Add invisible edge between Layer Norm 1 and Linear 2
        dot.edge(f'ln1_{i}', f'ff2_{i}', style='invis')

        last_node = f'add2_{i}'

    # Final Layer Norm and Output
    with dot.subgraph(name='cluster_output') as c:
        c.attr(rank='same', style='filled,rounded', fillcolor=colors['subgraph_output'], color='none')
        c.node('final_ln', node_label('Final Layer Norm', f'(seq_len, {d_model})'), fillcolor=colors['ln'])
        c.node('lm_head', node_label('LM Head', f'(seq_len, {vocab_size})'), fillcolor=colors['output'])
        c.node('output', node_label('Output Logits', f'(seq_len, {vocab_size})'), fillcolor=colors['output'])

    dot.edge(last_node, 'final_ln')
    dot.edge('final_ln', 'lm_head')
    dot.edge('lm_head', 'output')


    svg_data = dot.pipe(format='svg')
    return svg_data.decode('utf-8')


def get_model_predictions(model: HookedTransformer, tokens: torch.Tensor, temperature: float, seed: int, top_p: float, top_k: int = 10) -> list:
    """
    Generates top_k model predictions for the next token based on the input tokens using nucleus sampling.

    Args:
        model (HookedTransformer): The language model.
        tokens (torch.Tensor): Input tokens tensor.
        temperature (float): Sampling temperature.
        seed (int): Random seed for sampling.
        top_p (float): Cumulative probability threshold for nucleus sampling.
        top_k (int, optional): Number of top predictions to return. Defaults to 10.

    Returns:
        list: A list of tuples containing predicted tokens and their probabilities.
    """
    with torch.no_grad():
        logits = model(tokens)
        logits = logits[0, -1, :]  # The logits for the last token

        # Adjust logits by temperature
        if temperature == 0:
            # Take the argmax
            topk_logits, top_indices = torch.topk(logits, k=top_k)
            top_probs = torch.zeros_like(topk_logits)
            top_probs[0] = 1.0  # Set probability of the top token to 1
            predictions = [(model.to_string(idx.item()), prob.item()) for idx, prob in zip(top_indices, top_probs)]
        else:
            adjusted_logits = logits / temperature
            probs = torch.softmax(adjusted_logits, dim=-1)

            if top_p <= 0:
                # Include only the top token
                top_probs, top_indices = torch.topk(probs, k=1)
                predictions = [(model.to_string(top_indices[0].item()), 1.0)]
            elif top_p >= 1.0:
                # Include all tokens
                top_probs, top_indices = torch.topk(probs, k=top_k)
                predictions = [(model.to_string(idx.item()), prob.item()) for idx, prob in zip(top_indices, top_probs)]
            else:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                # Remove tokens with cumulative probability above top_p
                cumulative_mask = cumulative_probs <= top_p
                # Ensure at least one token is kept
                cumulative_mask = cumulative_mask | (torch.arange(len(cumulative_mask), device=cumulative_mask.device) == 0)
                filtered_probs = sorted_probs[cumulative_mask]
                filtered_indices = sorted_indices[cumulative_mask]

                # Normalize the filtered probabilities
                filtered_probs = filtered_probs / filtered_probs.sum()

                # Get top_k tokens
                top_probs, top_indices = filtered_probs[:top_k], filtered_indices[:top_k]

                predictions = [(model.to_string(idx.item()), prob.item()) for idx, prob in zip(top_indices, top_probs)]

    return predictions

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
            height=total_height,
            width=total_width,
            showlegend=False,
            margin=dict(l=50, r=100, t=100, b=80),  # Increased margins
            template="plotly"
        )
        
        # Adjust colorbar
        if num_heads > 1:
            fig.update_layout(coloraxis=dict(colorbar=dict(
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


def read_svg_file(file_path):
    """
    Read SVG file content and return it as a string.
    
    Args:
        file_path (str): Path to the SVG file
        
    Returns:
        str: Content of the SVG file
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading SVG file: {e}")
        return None



custom_css = """
#tabgroup { flex-grow: 1; }
.gradio-container .footer {display: none !important;}
footer {display: none !important;}
img.logo {
    margin-right: 20px;
    height: 50px;
    width: auto;
}
/* Force plots to use full width */
.plot-container {
    width: 100% !important;
}
.plotly-graph-div {
    height: auto !important;
}
.sidebar .gr-accordion {
    margin-bottom: 10px;
}
/* Adjust the width of the sidebar */
.gradio-container .sidebar {
    width: 300px;
}
"""

# Read the SVG file
svg_path = os.path.join('static', 'logo.svg')
logo_svg = read_svg_file(svg_path)

if logo_svg is None:
    print("Warning: Could not read logo.svg file. Please ensure it exists in the static directory.")
    logo_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" fill="#4a90e2"/>
        <text x="50" y="65" font-family="Arial" font-size="50" fill="white" text-anchor="middle">D</text>
    </svg>
    """

# Create data URL for the logo
logo_data_url = f"data:image/svg+xml;base64,{base64.b64encode(logo_svg.encode('utf-8')).decode('utf-8')}"

logo_and_favicon_html = f"""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <img src="{logo_data_url}" class="logo" alt="Logo"/>
    <h1 style="margin: 0;">DistilGPT2.guts</h1>
</div>
"""

# Calculate the total number of possible plots
num_embeddings_plots = len(["Token Embeddings", "Position Embeddings"])
num_residuals_plots = len(["Residual Stream Pre", "Residual Stream Mid", "Residual Stream Post"])
num_ln1_plots = len(["LayerNorm1 Scale", "LayerNorm1 Normalized"])
num_attention_plots = len([
    "Attention Q Vector",
    "Attention K Vector",
    "Attention V Vector",
    "Attention Output",
    "Attention Scores",
    "Attention Weights"
])
num_ln2_plots = len(["LayerNorm2 Scale", "LayerNorm2 Normalized"])
num_mlp_plots = len([
    "MLP Pre-Activation Heatmap",
    "MLP Pre-Activation Histogram",
    "MLP Post-Activation Heatmap",
    "MLP Post-Activation Histogram",
    "MLP Output Heatmap",
    "MLP Output Histogram"
])
num_prediction_plots = len(["Top 10 Predictions"])
num_computation_graph_plots = len(["Computational Graph"])

# Total plots
total_plots = (
    num_prediction_plots +
    num_embeddings_plots +
    num_residuals_plots +
    num_ln1_plots +
    num_attention_plots +
    num_ln2_plots +
    num_mlp_plots +
    num_computation_graph_plots
)

# Predefine the output components
plot_outputs = [gr.Plot(visible=False) for _ in range(total_plots - num_computation_graph_plots)]
# For the computation graph HTML
plot_outputs.extend([gr.HTML(visible=False) for _ in range(num_computation_graph_plots)])

# Build the interface
with gr.Blocks(
    css=custom_css,
    title="DistilGPT2.guts",
    analytics_enabled=False,
    theme=gr.themes.Default(),
    head=f"""
        <meta property="og:image" content="{logo_data_url}" />
        <meta property="og:title" content="DistilGPT2 Visualization" />
        <link rel="icon" type="image/svg+xml" href="{logo_data_url}" />
        <link rel="stylesheet" href="file/static/css/style.css" />
    """
) as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300) as sidebar:
            # Sidebar with collapsible sections
            gr.HTML(logo_and_favicon_html)
            # Input components
            query_input = gr.Textbox(label="Enter your query:", value="Hello, world!")
            block_number_input = gr.Dropdown(
                choices=list(range(6)),
                label="Select a block to visualize:",
                value=0
            )
            temperature_slider = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.7,
                label="Model Temperature"
            )
            seed_input = gr.Number(label="Seed for the model", value=42)
            top_p_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=1.0,
                label="Top P (nucleus sampling)"
            )
            submit_button = gr.Button("Submit")

            # Collapsible tabs with checkboxes
            with gr.Accordion("Embeddings", open=False):
                embeddings_graph_names = ["Token Embeddings", "Position Embeddings"]
                embeddings_checkboxes = [
                    gr.Checkbox(label=name, value=False) for name in embeddings_graph_names
                ]
            with gr.Accordion("Residual Streams", open=False):
                residuals_graph_names = ["Residual Stream Pre", "Residual Stream Mid", "Residual Stream Post"]
                residuals_checkboxes = [
                    gr.Checkbox(label=name, value=False) for name in residuals_graph_names
                ]
            with gr.Accordion("LayerNorm1", open=False):
                ln1_graph_names = ["LayerNorm1 Scale", "LayerNorm1 Normalized"]
                ln1_checkboxes = [
                    gr.Checkbox(label=name, value=False) for name in ln1_graph_names
                ]
            with gr.Accordion("Attention", open=False):
                attention_graph_names = [
                    "Attention Q Vector",
                    "Attention K Vector",
                    "Attention V Vector",
                    "Attention Output",
                    "Attention Scores",
                    "Attention Weights"
                ]
                attention_checkboxes = [
                    gr.Checkbox(label=name, value=False) for name in attention_graph_names
                ]
            with gr.Accordion("LayerNorm2", open=False):
                ln2_graph_names = ["LayerNorm2 Scale", "LayerNorm2 Normalized"]
                ln2_checkboxes = [
                    gr.Checkbox(label=name, value=False) for name in ln2_graph_names
                ]
            with gr.Accordion("MLP", open=False):
                mlp_graph_names = [
                    "MLP Pre-Activation Heatmap",
                    "MLP Pre-Activation Histogram",
                    "MLP Post-Activation Heatmap",
                    "MLP Post-Activation Histogram",
                    "MLP Output Heatmap",
                    "MLP Output Histogram"
                ]
                mlp_checkboxes = [
                    gr.Checkbox(label=name, value=False) for name in mlp_graph_names
                ]
            with gr.Accordion("Top 10 Predictions", open=False):
                top_predictions_graph_names = ["Top 10 Predictions"]
                top_predictions_checkboxes = [
                    gr.Checkbox(label=name, value=False) for name in top_predictions_graph_names
                ]
            with gr.Accordion("Computational Graph", open=False):
                computational_graph_names = ["Computational Graph"]
                computational_graph_checkboxes = [
                    gr.Checkbox(label=name, value=False) for name in computational_graph_names
                ]
        with gr.Column(scale=3) as main_area:
            # Predefine output components
            outputs_list = plot_outputs
            # For layout purposes, group them in a Column
            with gr.Column() as output_column:
                for plot_output in outputs_list:
                    plot_output.render()

    def on_submit(
        query,
        block_number,
        temperature,
        seed,
        top_p,
        *checkbox_values  # Accept all checkbox values as variable arguments
    ):
        # Now, checkbox_values is a flat tuple of all the checkbox values.
        # We need to split them back into their respective lists.
        
        # Determine the number of checkboxes in each category
        num_embeddings = len(embeddings_checkboxes)
        num_residuals = len(residuals_checkboxes)
        num_ln1 = len(ln1_checkboxes)
        num_attention = len(attention_checkboxes)
        num_ln2 = len(ln2_checkboxes)
        num_mlp = len(mlp_checkboxes)
        num_prediction_plots = len(top_predictions_checkboxes)
        num_computation_graph_plots = len(computational_graph_checkboxes)
        
        # Split the checkbox_values tuple into the respective lists
        idx = 0
        embeddings_checks = list(checkbox_values[idx: idx + num_embeddings])
        idx += num_embeddings
        residuals_checks = list(checkbox_values[idx: idx + num_residuals])
        idx += num_residuals
        ln1_checks = list(checkbox_values[idx: idx + num_ln1])
        idx += num_ln1
        attention_checks = list(checkbox_values[idx: idx + num_attention])
        idx += num_attention
        ln2_checks = list(checkbox_values[idx: idx + num_ln2])
        idx += num_ln2
        mlp_checks = list(checkbox_values[idx: idx + num_mlp])
        idx += num_mlp
        top_predictions_checks = list(checkbox_values[idx: idx + num_prediction_plots])
        idx += num_prediction_plots
        computational_graph_checks = list(checkbox_values[idx: idx + num_computation_graph_plots])

        output = visualize_model(query, int(block_number), temperature, int(seed), top_p)

        # Initialize list to hold outputs
        outputs_to_return = []

        # Top 10 Predictions Plot
        if top_predictions_checks[0]:
            outputs_to_return.append(
                gr.update(value=output["prediction_plot"], visible=True)
            )
        else:
            outputs_to_return.append(gr.update(visible=False))

        # Embeddings
        embeddings_plots = output["plots"]["Embeddings"]
        for idx2, checked in enumerate(embeddings_checks):
            if checked and idx2 < len(embeddings_plots):
                outputs_to_return.append(
                    gr.update(value=embeddings_plots[idx2], visible=True)
                )
            else:
                outputs_to_return.append(gr.update(visible=False))

        # Residual Streams
        residuals_plots = output["plots"]["Residual Streams"]
        for idx2, checked in enumerate(residuals_checks):
            if checked and idx2 < len(residuals_plots):
                outputs_to_return.append(
                    gr.update(value=residuals_plots[idx2], visible=True)
                )
            else:
                outputs_to_return.append(gr.update(visible=False))

        # LayerNorm1
        ln1_plots = output["plots"]["LayerNorm1"]
        for idx2, checked in enumerate(ln1_checks):
            if checked and idx2 < len(ln1_plots):
                outputs_to_return.append(
                    gr.update(value=ln1_plots[idx2], visible=True)
                )
            else:
                outputs_to_return.append(gr.update(visible=False))

        # Attention
        attention_plots_output = output["plots"]["Attention"]
        for idx2, checked in enumerate(attention_checks):
            if checked and idx2 < len(attention_plots_output):
                outputs_to_return.append(
                    gr.update(value=attention_plots_output[idx2], visible=True)
                )
            else:
                outputs_to_return.append(gr.update(visible=False))

        # LayerNorm2
        ln2_plots = output["plots"]["LayerNorm2"]
        for idx2, checked in enumerate(ln2_checks):
            if checked and idx2 < len(ln2_plots):
                outputs_to_return.append(
                    gr.update(value=ln2_plots[idx2], visible=True)
                )
            else:
                outputs_to_return.append(gr.update(visible=False))

        # MLP
        mlp_plots = output["plots"]["MLP"]
        for idx2, checked in enumerate(mlp_checks):
            if checked and idx2 < len(mlp_plots):
                outputs_to_return.append(
                    gr.update(value=mlp_plots[idx2], visible=True)
                )
            else:
                outputs_to_return.append(gr.update(visible=False))

        # Computational Graph
        if computational_graph_checks[0]:
            # Generate the computation graph SVG data
            model = load_model()
            svg_data = generate_computation_graph_svg(model)
            outputs_to_return.append(
                gr.update(value=svg_data, visible=True)
            )
        else:
            outputs_to_return.append(gr.update(visible=False))

        return outputs_to_return

    submit_button.click(
        fn=on_submit,
        inputs=[
            query_input,
            block_number_input,
            temperature_slider,
            seed_input,
            top_p_slider,
            *embeddings_checkboxes,  # Unpack the lists of checkboxes
            *residuals_checkboxes,
            *ln1_checkboxes,
            *attention_checkboxes,
            *ln2_checkboxes,
            *mlp_checkboxes,
            *top_predictions_checkboxes,
            *computational_graph_checkboxes
        ],
        outputs=plot_outputs  # Use the predefined outputs
    )

if __name__ == "__main__":
    demo.launch(
        show_api=False,
        favicon_path=svg_path,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
