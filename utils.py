import torch
from graphviz import Digraph
from functools import lru_cache
from transformer_lens import HookedTransformer


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

@lru_cache(maxsize=1)
def load_model() -> HookedTransformer:
    """
    Loads the distilgpt2 model using the transformer_lens HookedTransformer.

    Returns:
        HookedTransformer: The loaded distilgpt2 model.
    """
    return HookedTransformer.from_pretrained("distilgpt2")


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
