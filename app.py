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
import graphviz

@lru_cache(maxsize=1)
def load_model() -> HookedTransformer:
    """
    Loads the distilgpt2 model using the transformer_lens HookedTransformer.

    Returns:
        HookedTransformer: The loaded distilgpt2 model.
    """
    return HookedTransformer.from_pretrained("distilgpt2")


def create_hierarchical_graph(model):
    dot = graphviz.Digraph(comment='DistilGPT2 Hierarchical Computational Graph')
    dot.attr(rankdir='TB', size='12,12', ratio='fill')
    
    # Input
    dot.node('input', 'Input Tokens', shape='ellipse')
    
    # Embedding
    with dot.subgraph(name='cluster_embedding') as c:
        c.attr(label='Embedding', style='filled', color='lightblue')
        c.node('token_emb', 'Token Embedding')
        c.node('pos_emb', 'Positional Embedding')
        c.edge('token_emb', 'pos_emb')
    
    dot.edge('input', 'token_emb')
    
    # Transformer Blocks (similar to above code)
    for i in range(6):
        with dot.subgraph(name=f'cluster_block_{i}') as c:
            c.attr(label=f'Block {i}', style='filled', color='lightgrey')
            
            with c.subgraph(name=f'cluster_ln1_{i}') as ln1:
                ln1.attr(label='Layer Norm 1', style='filled', color='lightyellow')
                ln1.node(f'ln1_mean_{i}', 'Mean')
                ln1.node(f'ln1_var_{i}', 'Variance')
                ln1.node(f'ln1_norm_{i}', 'Normalize')
                ln1.edge(f'ln1_mean_{i}', f'ln1_norm_{i}')
                ln1.edge(f'ln1_var_{i}', f'ln1_norm_{i}')
            
            with c.subgraph(name=f'cluster_attn_{i}') as attn:
                attn.attr(label='Self-Attention', style='filled', color='lightgreen')
                attn.node(f'q_{i}', 'Query')
                attn.node(f'k_{i}', 'Key')
                attn.node(f'v_{i}', 'Value')
                attn.node(f'attn_scores_{i}', 'Attention Scores')
                attn.node(f'attn_output_{i}', 'Attention Output')
                attn.edge(f'q_{i}', f'attn_scores_{i}')
                attn.edge(f'k_{i}', f'attn_scores_{i}')
                attn.edge(f'attn_scores_{i}', f'attn_output_{i}')
                attn.edge(f'v_{i}', f'attn_output_{i}')
            
            with c.subgraph(name=f'cluster_ln2_{i}') as ln2:
                ln2.attr(label='Layer Norm 2', style='filled', color='lightyellow')
                ln2.node(f'ln2_mean_{i}', 'Mean')
                ln2.node(f'ln2_var_{i}', 'Variance')
                ln2.node(f'ln2_norm_{i}', 'Normalize')
                ln2.edge(f'ln2_mean_{i}', f'ln2_norm_{i}')
                ln2.edge(f'ln2_var_{i}', f'ln2_norm_{i}')
            
            with c.subgraph(name=f'cluster_mlp_{i}') as mlp:
                mlp.attr(label='MLP', style='filled', color='lightpink')
                mlp.node(f'fc1_{i}', 'FC1')
                mlp.node(f'gelu_{i}', 'GELU')
                mlp.node(f'fc2_{i}', 'FC2')
                mlp.edge(f'fc1_{i}', f'gelu_{i}')
                mlp.edge(f'gelu_{i}', f'fc2_{i}')
            
            c.edge(f'ln1_norm_{i}', f'q_{i}')
            c.edge(f'ln1_norm_{i}', f'k_{i}')
            c.edge(f'ln1_norm_{i}', f'v_{i}')
            c.edge(f'attn_output_{i}', f'ln2_mean_{i}')
            c.edge(f'attn_output_{i}', f'ln2_var_{i}')
            c.edge(f'ln2_norm_{i}', f'fc1_{i}')
        
        if i == 0:
            dot.edge('pos_emb', f'ln1_mean_0')
            dot.edge('pos_emb', f'ln1_var_0')
        else:
            dot.edge(f'fc2_{i-1}', f'ln1_mean_{i}')
            dot.edge(f'fc2_{i-1}', f'ln1_var_{i}')
    
    # Final Layer Norm and Output (similar to above code)
    dot.edge('fc2_5', 'final_ln_mean')
    dot.edge('fc2_5', 'final_ln_var')
    dot.node('output', 'Output Logits', shape='ellipse')
    dot.edge('final_ln_norm', 'output')
    
    return dot

def get_model_predictions(model: HookedTransformer, tokens: torch.Tensor, top_k: int = 10) -> list:
    """
    Generates top_k model predictions for the next token based on the input tokens.

    Args:
        model (HookedTransformer): The language model.
        tokens (torch.Tensor): Input tokens tensor.
        top_k (int, optional): Number of top predictions to return. Defaults to 10.

    Returns:
        list: A list of tuples containing predicted tokens and their probabilities.
    """
    with torch.no_grad():
        logits = model(tokens)
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
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
    """
    Creates a line plot using plotly.

    Args:
        data (torch.Tensor): The data to plot.
        title (str): The title of the plot.
        x_labels (list, optional): Labels for the x-axis. Defaults to None.

    Returns:
        go.Figure: The plotly figure.
    """
    data_np = to_numpy(data).squeeze()
    if x_labels:
        x = x_labels
    else:
        x = list(range(len(data_np)))
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=data_np,
        mode='lines+markers'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Token" if x_labels else "Position",
        yaxis_title="Value"
    )
    return fig

def plot_histogram(data: torch.Tensor, title: str) -> go.Figure:
    """
    Creates a histogram using plotly.

    Args:
        data (torch.Tensor): The data to plot.
        title (str): The title of the plot.

    Returns:
        go.Figure: The plotly figure.
    """
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
        yaxis_title="Frequency"
    )
    return fig

def plot_attention(data: torch.Tensor, title: str, tokens: list) -> go.Figure:
    """
    Creates attention heatmaps using plotly, combined into a subplot figure with correctly placed head labels.

    Args:
        data (torch.Tensor): The attention data to plot.
        title (str): The base title of the plot.
        tokens (list): The tokens for axis labels.

    Returns:
        go.Figure: The plotly figure containing subplots.
    """
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
            xaxis_title="Keys",
            yaxis_title="Queries"
        )
        return fig
    elif len(data_np.shape) == 3 or len(data_np.shape) == 4:
        if len(data_np.shape) == 4:
            data_np = data_np[0]  # Remove batch dimension
        num_heads = data_np.shape[0]
        cols = 4
        rows = (num_heads + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols,
                            horizontal_spacing=0.15,
                            vertical_spacing=0.3)  # Maximum allowed vertical spacing
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
            fig.update_xaxes(title_text="Keys", row=row, col=col, side="top", title_standoff=25)
            fig.update_yaxes(title_text="Queries", row=row, col=col, title_standoff=25)
            
            # Add head label below each matrix
            fig.add_annotation(
                text=f"Head {i}",
                xref=f"x{i+1}",
                yref=f"y{i+1}",
                x=0.5,
                y=-0.2,  # Adjusted to be below the matrix but within constraints
                showarrow=False,
                font=dict(size=12),
                yanchor="top"
            )
        
        # Adjust layout for better readability
        fig.update_layout(
            title=title,
            height=300*rows,  # Adjusted height
            width=300*cols,
            showlegend=False,
            margin=dict(t=100, l=100, r=100, b=50)
        )
        
        # Rotate x-axis labels and adjust their position
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
        
        # Adjust font sizes
        fig.update_layout(
            font=dict(size=10),
            title_font=dict(size=16)
        )
        
        # Add colorbar
        fig.update_layout(coloraxis=dict(colorbar=dict(title="Attention Score", y=0.5)))
        
        return fig
    

def plot_predictions(predictions: list) -> go.Figure:
    """
    Plots a bar chart of the top 10 prediction probabilities.

    Args:
        predictions (list): A list of tuples containing predicted tokens and their probabilities.

    Returns:
        go.Figure: The plotly figure.
    """
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
        yaxis_title="Probability"
    )
    return fig

def visualize_model(query: str, block_number: int) -> dict:
    model = load_model()
    tokens = model.to_tokens(query)
    token_strings = model.to_str_tokens(tokens)
    _, cache = model.run_with_cache(tokens)

    # Get model predictions
    predictions = get_model_predictions(model, tokens)

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

with gr.Blocks(css="#tabgroup { flex-grow: 1; }") as demo:
    gr.Markdown("# DistilGPT2 Visualization")
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="Enter your query:", value="Hello, world!")
            block_number_input = gr.Dropdown(
                choices=list(range(6)),
                label="Select a block to visualize:",
                value=0
            )
            submit_button = gr.Button("Submit")
        with gr.Column(scale=1):
            prediction_plot = gr.Plot(label="Top 10 Predictions")

    with gr.Tabs(elem_id="tabgroup") as tabs:
        with gr.Tab("Embeddings"):
            embeddings_plot1 = gr.Plot(label="Token Embeddings")
            embeddings_plot2 = gr.Plot(label="Position Embeddings")
        with gr.Tab("Residual Streams"):
            residuals_plot1 = gr.Plot(label="Residual Stream Pre")
            residuals_plot2 = gr.Plot(label="Residual Stream Mid")
            residuals_plot3 = gr.Plot(label="Residual Stream Post")
        with gr.Tab("LayerNorm1"):
            ln1_plot1 = gr.Plot(label="LayerNorm1 Scale")
            ln1_plot2 = gr.Plot(label="LayerNorm1 Normalized")
        with gr.Tab("Attention"):
            attention_plots = [gr.Plot(label=f"Attention Component {i+1}") for i in range(6)]
        with gr.Tab("LayerNorm2"):
            ln2_plot1 = gr.Plot(label="LayerNorm2 Scale")
            ln2_plot2 = gr.Plot(label="LayerNorm2 Normalized")
        with gr.Tab("MLP"):
            mlp_plot1 = gr.Plot(label="MLP Pre-Activation Heatmap")
            mlp_plot2 = gr.Plot(label="MLP Pre-Activation Histogram")
            mlp_plot3 = gr.Plot(label="MLP Post-Activation Heatmap")
            mlp_plot4 = gr.Plot(label="MLP Post-Activation Histogram")
            mlp_plot5 = gr.Plot(label="MLP Output Heatmap")
            mlp_plot6 = gr.Plot(label="MLP Output Histogram")

    def on_submit(query, block_number):
        output = visualize_model(query, int(block_number))
        # Extract the plots
        embeddings_plots = output["plots"]["Embeddings"]
        residuals_plots = output["plots"]["Residual Streams"]
        ln1_plots = output["plots"]["LayerNorm1"]
        attention_plots = output["plots"]["Attention"]
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

        # Attention plot (combined)
        attention_plots_val = output["plots"]["Attention"]
        return_values.extend(attention_plots_val + [None] * (6 - len(attention_plots_val)))  # Pad with None if less than 6 plots


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

        return tuple(return_values)

    submit_button.click(
        fn=on_submit,
        inputs=[query_input, block_number_input],
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
            mlp_plot6
        ]
    )


if __name__ == "__main__":
    demo.launch()
