# 🔬 DistilGPT2 Visualization 🧠

## 📚 Educational Tool for Exploring Language Model Internals

Welcome to the DistilGPT2 Visualization app! 🎉 This interactive tool allows you to peek inside the "brain" of a language model and understand its inner workings.

![image](https://github.com/user-attachments/assets/90c5157f-81ad-451e-b9a8-95cd225b7695)

### 🌟 Features

- 🔍 Visualize various components of the DistilGPT2 model
- 💡 Explore embeddings, attention mechanisms, and more
- 📊 Interactive plots and heatmaps
- 🎓 Perfect for students, researchers, and AI enthusiasts

### 🚀 Getting Started

1. 📝 Enter your query in the text box
2. 🔢 Select a transformer block to visualize
3. 🖱️ Click "Submit" to generate visualizations

### 📊 Visualization Tabs

- 📌 Embeddings: See how words are represented numerically
- 🔁 Residual Streams: Observe information flow through the model
- 📏 LayerNorm1 & LayerNorm2: Understand normalization techniques
- 👀 Attention: Visualize how the model focuses on different parts of the input
- 🧮 MLP: Explore the feed-forward neural network components

### 🎯 Educational Goals

- 🧠 Demystify the inner workings of transformer-based language models
- 🔬 Provide hands-on experience with model internals
- 📈 Enhance understanding of NLP concepts through visual representations

### 🛠️ Technical Details

- Built with Python, Gradio, and PyTorch
- Utilizes the `transformer_lens` library for model introspection
- Generates interactive plots using Plotly

### 🏃‍♂️ Running Locally

To run this project on your local machine, follow these steps:

1. 📥 Clone the repository:
   ```
   git clone https://github.com/your-username/distilgpt2-visualization.git
   cd distilgpt2-visualization
   ```

2. 🐍 Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. 📦 Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. 🚀 Run the application:
   ```
   python app.py
   ```

5. 🌐 Open your web browser and navigate to `http://localhost:7860` to access the application.

Note: Ensure you have Python 3.7+ installed on your system before running the application locally.

### 🐳 Running with Docker

Alternatively, you can run the application using Docker:

1. 🏗️ Build the Docker image:
   ```
   docker build -t distilgpt2-visualization .
   ```

2. 🐳 Run the Docker container:
   ```
   docker run -p 7860:7860 distilgpt2-visualization
   ```

3. 🌐 Access the application at `http://localhost:7860` in your web browser.

Dive in and start exploring the fascinating world of language models! 🌊🤖
