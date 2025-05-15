# BERT Pretraining Visualization

This application provides an interactive visualization of the BERT model pretraining process using Streamlit. It allows you to experiment with different pretraining configurations and observe the training process in real-time.

## Features

- Choose between Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) pretraining tasks
- Visualize training loss curves in real-time
- See examples of masked tokens and predictions
- Adjust model parameters (hidden size, feed-forward size)
- Configure training parameters (epochs, learning rate)
- Customize the training corpus
- View final evaluation metrics

## How to Run

1. Install the required dependencies:

```bash
pip install -r ../requirements.txt
```

2. Run the Streamlit application:

```bash
cd python
streamlit run streamlit_app.py
```

3. The application will open in your default web browser.

## Understanding the Interface

### Configuration (Sidebar)

- **Training Corpus**: Enter sentences for training (one per line)
- **Model Parameters**: Adjust hidden size and feed-forward layer size
- **Training Parameters**: Set number of epochs and learning rate
- **Pretraining Tasks**: Select MLM, NSP, or both

### Main Interface

- **Model Architecture**: Visual representation of the BERT model
- **Pretraining Tasks**: Explanation of MLM and NSP
- **Training Process**: Real-time visualization of training metrics
- **Evaluation Results**: Final accuracy metrics
- **Training Examples**: Examples of masked sentences and predictions

## About BERT Pretraining

BERT (Bidirectional Encoder Representations from Transformers) uses two pretraining tasks:

1. **Masked Language Modeling (MLM)**: The model learns to predict masked tokens in a sentence, helping it understand context and semantics.

2. **Next Sentence Prediction (NSP)**: The model learns to predict if two sentences follow each other in the original text, helping it understand relationships between sentences.

This implementation provides a simplified version of BERT with a single transformer layer for educational purposes.