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
            title_x=0.5,  # Center the title horizontally (0 to 1)
            title_y=0.95,  # Position from bottom (0) to top (1)
            xaxis_title="Keys",
            yaxis_title="Queries",
            autosize=True,
            height=500,
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
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        # Calculate appropriate vertical spacing based on number of rows
        # The vertical spacing must be less than 1/(rows-1)
        vertical_spacing = min(0.2, 0.9/(rows)) if rows > 1 else 0.2
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            horizontal_spacing=0.15,
            vertical_spacing=vertical_spacing
        )
        
        for i in range(num_heads):
            row = i // cols + 1
            col = i % cols + 1
            fig.add_trace(
                go.Heatmap(
                    z=data_np[i],
                    x=tokens,
                    y=tokens,
                    colorscale='RdBu',
                    reversescale=True,
                    showscale=False
                ),
                row=row,
                col=col
            )
            
            fig.update_xaxes(
                title_text="Keys", 
                row=row, 
                col=col, 
                side="top", 
                title_standoff=25,
                automargin=True,
                scaleanchor=f"y{i+1}",
                constrain="domain"
            )
            
            fig.update_yaxes(
                title_text="Queries", 
                row=row, 
                col=col, 
                title_standoff=25,
                automargin=True,
                scaleanchor=f"x{i+1}",
                constrain="domain"
            )
            
            fig.add_annotation(
                text=f"Head {i}",
                xref=f"x{i+1}",
                yref=f"y{i+1}",
                x=0.5,
                y=-0.25,
                showarrow=False,
                font=dict(size=12),
                row=row,
                col=col
            )
        
        fig.update_layout(
            title=title,
            title_x=0.5,  # Center the title horizontally (0 to 1)
            title_y=0.97,  # Position from bottom (0) to top (1)
            height=300*rows,
            autosize=True,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=100),
            template="plotly",
            uirevision=True
        )
        
        fig.update_layout(coloraxis=dict(colorbar=dict(title="Attention Score", y=0.5)))
        
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
.tab-content {
    min-height: 500px;
    width: 100%;
}
/* Ensure plots maintain size in hidden tabs */
.tabs > div[role="tabpanel"] {
    min-width: 100%;
    width: 100% !important;
}
.plot-container.plotly {
    min-width: 100%;
    width: 100% !important;
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

custom_css = """
#tabgroup { flex-grow: 1; }
.gradio-container .footer {display: none !important;}
footer {display: none !important;}
img.logo {
    margin-right: 20px;
    height: 50px;
    width: auto;
}
"""


with gr.Blocks(
    css=custom_css,
    title="DistilGPT2.guts",
    analytics_enabled=False,
    head=f"""
        <meta property="og:image" content="{logo_data_url}" />
        <meta property="og:title" content="DistilGPT2 Visualization" />
        <link rel="icon" type="image/svg+xml" href="{logo_data_url}" />
        <link rel="stylesheet" href="file/static/css/style.css" />
    """
) as demo:
    gr.HTML(logo_and_favicon_html)
    with gr.Row():
        with gr.Column(scale=1):
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
        with gr.Column(scale=1):
            prediction_plot = gr.Plot(label="Top 10 Predictions")

    with gr.Tabs(elem_id="tabgroup") as tabs:
        with gr.Tab("Embeddings") as embeddings_tab:
            embeddings_plot1 = gr.Plot(label="Token Embeddings")
            embeddings_plot2 = gr.Plot(label="Position Embeddings")
        with gr.Tab("Residual Streams") as residuals_tab:
            residuals_plot1 = gr.Plot(label="Residual Stream Pre")
            residuals_plot2 = gr.Plot(label="Residual Stream Mid")
            residuals_plot3 = gr.Plot(label="Residual Stream Post")
        with gr.Tab("LayerNorm1") as ln1_tab:
            ln1_plot1 = gr.Plot(label="LayerNorm1 Scale")
            ln1_plot2 = gr.Plot(label="LayerNorm1 Normalized")
        with gr.Tab("Attention") as attention_tab:
            attention_plots = [gr.Plot(label=f"Attention Component {i+1}") for i in range(6)]
        with gr.Tab("LayerNorm2") as ln2_tab:
            ln2_plot1 = gr.Plot(label="LayerNorm2 Scale")
            ln2_plot2 = gr.Plot(label="LayerNorm2 Normalized")
        with gr.Tab("MLP") as mlp_tab:
            mlp_plot1 = gr.Plot(label="MLP Pre-Activation Heatmap")
            mlp_plot2 = gr.Plot(label="MLP Pre-Activation Histogram")
            mlp_plot3 = gr.Plot(label="MLP Post-Activation Heatmap")
            mlp_plot4 = gr.Plot(label="MLP Post-Activation Histogram")
            mlp_plot5 = gr.Plot(label="MLP Output Heatmap")
            mlp_plot6 = gr.Plot(label="MLP Output Histogram")
        with gr.Tab("Computation Graph") as computation_graph_tab:
            computation_graph_html = gr.HTML(label="Computation Graph")

    def on_submit(query, block_number, temperature, seed, top_p):
        output = visualize_model(query, int(block_number), temperature, int(seed), top_p)
        # Extract the plots
        embeddings_plots = output["plots"]["Embeddings"]
        residuals_plots = output["plots"]["Residual Streams"]
        ln1_plots = output["plots"]["LayerNorm1"]
        attention_plots_output = output["plots"]["Attention"]
        ln2_plots = output["plots"]["LayerNorm2"]
        mlp_plots = output["plots"]["MLP"]

        # Prepare the return values
        return_values = [
            output["prediction_plot"]
        ]

        # Embeddings plots
        embeddings_plot1_val = embeddings_plots[0] if len(embeddings_plots) > 0 else None
        embeddings_plot2_val = embeddings_plots[1] if len(embeddings_plots) > 1 else None
        return_values.extend([embeddings_plot1_val, embeddings_plot2_val])

        # Residuals plots
        residuals_plot1_val = residuals_plots[0] if len(residuals_plots) > 0 else None
        residuals_plot2_val = residuals_plots[1] if len(residuals_plots) > 1 else None
        residuals_plot3_val = residuals_plots[2] if len(residuals_plots) > 2 else None
        return_values.extend([residuals_plot1_val, residuals_plot2_val, residuals_plot3_val])

        # LayerNorm1 plots
        ln1_plot1_val = ln1_plots[0] if len(ln1_plots) > 0 else None
        ln1_plot2_val = ln1_plots[1] if len(ln1_plots) > 1 else None
        return_values.extend([ln1_plot1_val, ln1_plot2_val])

        # Attention plots
        attention_plots_vals = attention_plots_output
        return_values.extend(attention_plots_vals + [None] * (6 - len(attention_plots_vals)))  # Pad with None if less than 6 plots

        # LayerNorm2 plots
        ln2_plot1_val = ln2_plots[0] if len(ln2_plots) > 0 else None
        ln2_plot2_val = ln2_plots[1] if len(ln2_plots) > 1 else None
        return_values.extend([ln2_plot1_val, ln2_plot2_val])

        # MLP plots
        mlp_plot1_val = mlp_plots[0] if len(mlp_plots) > 0 else None
        mlp_plot2_val = mlp_plots[1] if len(mlp_plots) > 1 else None
        mlp_plot3_val = mlp_plots[2] if len(mlp_plots) > 2 else None
        mlp_plot4_val = mlp_plots[3] if len(mlp_plots) > 3 else None
        mlp_plot5_val = mlp_plots[4] if len(mlp_plots) > 4 else None
        mlp_plot6_val = mlp_plots[5] if len(mlp_plots) > 5 else None
        return_values.extend([mlp_plot1_val, mlp_plot2_val, mlp_plot3_val, mlp_plot4_val, mlp_plot5_val, mlp_plot6_val])

        # Generate the computation graph SVG data
        model = load_model()
        svg_data = generate_computation_graph_svg(model)
        return_values.append(svg_data)

        return tuple(return_values)

    submit_button.click(
        fn=on_submit,
        inputs=[query_input, block_number_input, temperature_slider, seed_input, top_p_slider],
        outputs=[
            prediction_plot,
            embeddings_plot1,
            embeddings_plot2,
            residuals_plot1,
            residuals_plot2,
            residuals_plot3,
            ln1_plot1,
            ln1_plot2,
            *attention_plots,  # Unpack the attention plots
            ln2_plot1,
            ln2_plot2,
            mlp_plot1,
            mlp_plot2,
            mlp_plot3,
            mlp_plot4,
            mlp_plot5,
            mlp_plot6,
            computation_graph_html  # Added the new output here
        ]
    )

if __name__ == "__main__":
    demo.launch(
        show_api=False,
        favicon_path=svg_path  # Set the favicon path directly in launch
    )