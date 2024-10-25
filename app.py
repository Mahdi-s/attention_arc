import os
import base64
import asyncio
import gradio as gr
from plotting import async_visualize_model
from utils import load_model, generate_computation_graph_svg, read_svg_file


# Update the custom_css variable to load from the file
with open(os.path.join('static', 'css', 'style.css'), 'r') as f:
    custom_css = f.read()

theme_toggle_html = """
<div class="header-container">
    <div class="logo-title">
        <img src="{{logo_data_url}}" class="logo" alt="Logo"/>
        <h1>DistilGPT2.guts</h1>
    </div>
</div>
"""

light_theme = gr.themes.Soft()
dark_theme = gr.themes.Base()

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

# Update the header HTML to use the theme toggle
header_html = theme_toggle_html.replace("{{logo_data_url}}", logo_data_url)

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

def toggle_theme(dark_mode):
    """Simple theme toggle function that returns the appropriate theme object"""
    return dark_theme if dark_mode else light_theme

with gr.Blocks(
    css=custom_css,
    title="DistilGPT2.guts",
    analytics_enabled=False,
    theme=dark_theme
) as demo:
    # Add the header with logo and title
    gr.HTML(header_html)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300) as sidebar:
            # Add the hidden theme toggle checkbox
            theme_toggle = gr.Checkbox(
                label="Dark Mode",
                value=True,
                visible=False,
                elem_id="theme-hidden"
            )
            
            # Define the theme toggle function
            def toggle_theme(checked):
                return dark_theme if checked else light_theme
            
            # Connect the theme toggle to the theme update function
            theme_toggle.change(
                fn=toggle_theme,
                inputs=theme_toggle,
                outputs=demo
            )
            
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
            error_message = gr.Markdown(visible=False)
            
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
            
            # Predefine the output components inside the Blocks context
            plot_outputs = [gr.Plot(visible=False) for _ in range(total_plots - num_computation_graph_plots)]
            # For the computation graph HTML
            plot_outputs.extend([gr.HTML(visible=False) for _ in range(num_computation_graph_plots)])
            
            outputs_list = [error_message] + plot_outputs
            
            # For layout purposes, group them in a Column
            with gr.Column() as output_column:
                pass  # Components are already added to the Blocks context

    def on_submit(*args):
        # First yield: Indicate processing
        yield [gr.update(value="Processing...", visible=True)] + [gr.update(visible=False)] * len(plot_outputs)
        
        try:
            result = asyncio.run(async_visualize_model(*args[:5]))
            
            # Extract arguments
            query = args[0]
            block_number = args[1]
            temperature = args[2]
            seed = args[3]
            top_p = args[4]
            checkbox_values = args[5:]
            
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

            # Build outputs_to_return, starting with gr.update(visible=False) for error_message
            outputs_to_return = [gr.update(visible=False)]

            # Top 10 Predictions Plot
            if top_predictions_checks[0]:
                outputs_to_return.append(
                    gr.update(value=result["prediction_plot"], visible=True)
                )
            else:
                outputs_to_return.append(gr.update(visible=False))

            # Embeddings
            embeddings_plots = result["plots"]["Embeddings"]
            for idx2, checked in enumerate(embeddings_checks):
                if checked and idx2 < len(embeddings_plots):
                    outputs_to_return.append(
                        gr.update(value=embeddings_plots[idx2], visible=True)
                    )
                else:
                    outputs_to_return.append(gr.update(visible=False))

            # Residual Streams
            residuals_plots = result["plots"]["Residual Streams"]
            for idx2, checked in enumerate(residuals_checks):
                if checked and idx2 < len(residuals_plots):
                    outputs_to_return.append(
                        gr.update(value=residuals_plots[idx2], visible=True)
                    )
                else:
                    outputs_to_return.append(gr.update(visible=False))

            # LayerNorm1
            ln1_plots = result["plots"]["LayerNorm1"]
            for idx2, checked in enumerate(ln1_checks):
                if checked and idx2 < len(ln1_plots):
                    outputs_to_return.append(
                        gr.update(value=ln1_plots[idx2], visible=True)
                    )
                else:
                    outputs_to_return.append(gr.update(visible=False))

            # Attention
            attention_plots_output = result["plots"]["Attention"]
            for idx2, checked in enumerate(attention_checks):
                if checked and idx2 < len(attention_plots_output):
                    outputs_to_return.append(
                        gr.update(value=attention_plots_output[idx2], visible=True)
                    )
                else:
                    outputs_to_return.append(gr.update(visible=False))

            # LayerNorm2
            ln2_plots = result["plots"]["LayerNorm2"]
            for idx2, checked in enumerate(ln2_checks):
                if checked and idx2 < len(ln2_plots):
                    outputs_to_return.append(
                        gr.update(value=ln2_plots[idx2], visible=True)
                    )
                else:
                    outputs_to_return.append(gr.update(visible=False))

            # MLP
            mlp_plots = result["plots"]["MLP"]
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

            yield outputs_to_return
        except Exception as e:
            yield [gr.update(value=f"An error occurred: {str(e)}", visible=True)] + [gr.update(visible=False)] * len(plot_outputs)

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
        outputs=outputs_list
    )

if __name__ == "__main__":
    demo.launch(
        show_api=False,
        favicon_path=svg_path,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )

