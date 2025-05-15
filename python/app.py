import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from model import create_bert_model
from pretrain import data_for_pretraining, tokenizer, create_nsp_example, apply_mlm, token_ids, compute_loss, compute_gradients

# Set page configuration
st.set_page_config(page_title="BERT Pretraining Visualization", layout="wide")

# Title and introduction
st.title("BERT Model Pretraining Visualization")
st.markdown("""
This application allows you to visualize the pretraining process of a simplified BERT model.

BERT (Bidirectional Encoder Representations from Transformers) uses two pretraining tasks:
- **Masked Language Modeling (MLM)**: Predicts masked tokens in a sentence
- **Next Sentence Prediction (NSP)**: Predicts if two sentences follow each other in the original text

You can choose to train with either or both objectives and visualize the training process.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Sample corpus
default_corpus = """
the cat sat on the mat
the dog played in the yard
he is playing football now
she was reading a book
they are going to the park
the weather is nice today
we should go for a walk
the children are playing outside
the birds are singing in the trees
the sun is shining brightly
"""

# Corpus input
st.sidebar.subheader("Training Corpus")
corpus_text = st.sidebar.text_area("Enter sentences for training (one per line):", default_corpus, height=200)
corpus_list = [line.strip() for line in corpus_text.split('\n') if line.strip()]

# Model parameters
st.sidebar.subheader("Model Parameters")
d_model = st.sidebar.slider("Hidden Size (d_model)", min_value=16, max_value=128, value=32, step=16)
d_ff = st.sidebar.slider("Feed-Forward Size (d_ff)", min_value=64, max_value=512, value=128, step=64)

# Training parameters
st.sidebar.subheader("Training Parameters")
n_epoch = st.sidebar.slider("Number of Epochs", min_value=100, max_value=1000, value=300, step=100)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1], value=0.01)

# Pretraining tasks selection
st.sidebar.subheader("Pretraining Tasks")
use_mlm = st.sidebar.checkbox("Masked Language Modeling (MLM)", value=True)
use_nsp = st.sidebar.checkbox("Next Sentence Prediction (NSP)", value=True)

if not use_mlm and not use_nsp:
    st.warning("Please select at least one pretraining task (MLM or NSP).")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Model Architecture")
    st.image("https://miro.medium.com/v2/resize:fit:1162/1*vG_xN7a9HuLCU05U5IznPQ.png", caption="BERT Model Architecture", use_container_width=True)
    
    st.markdown("""
    ### Simplified BERT Implementation
    
    This implementation includes:
    - Token, position, and segment embeddings
    - Self-attention mechanism
    - Feed-forward neural network
    - Layer normalization
    - MLM and NSP prediction heads
    
    Note: This is a simplified version with a single transformer layer.
    """)

with col2:
    st.header("Pretraining Tasks")
    
    st.subheader("Masked Language Modeling (MLM)")
    st.markdown("""
    In MLM, 15% of tokens are randomly masked, and the model learns to predict them:
    - 80% replaced with [MASK]
    - 10% replaced with random word
    - 10% left unchanged
    
    Example: "The cat [MASK] on the mat" → predict "sat"
    """)
    
    st.subheader("Next Sentence Prediction (NSP)")
    st.markdown("""
    In NSP, the model predicts if sentence B follows sentence A in the original text:
    - 50% are consecutive sentences (IsNext)
    - 50% are random sentences (NotNext)
    
    Example: 
    - "The cat sat on the mat. The dog barked loudly." → NotNext
    - "The cat sat on the mat. It was sleeping." → IsNext
    """)

# Training section
st.header("Training Process")

# Initialize placeholders for metrics
if 'losses' not in st.session_state:
    st.session_state.losses = {"total": [], "mlm": [], "nsp": []}
    st.session_state.examples = []
    st.session_state.trained = False

# Training button
start_training = st.button("Start Training")

if start_training and len(corpus_list) >= 2:
    # Reset metrics
    st.session_state.losses = {"total": [], "mlm": [], "nsp": []}
    st.session_state.examples = []
    st.session_state.trained = False
    
    # Process corpus
    vocab = data_for_pretraining(corpus_list)
    vocab_size = len(vocab)
    token2idx, idx2token = tokenizer(vocab)
    
    # Initialize model
    model = create_bert_model(
        vocab_size=vocab_size,
        max_seq_length=512,
        d_model=d_model,
        num_heads=1,  # Simplified for this implementation
        d_ff=d_ff,
        num_layers=1,  # Simplified for this implementation
        for_pretraining=True
    )
    
    # Create progress bar and metrics
    progress_bar = st.progress(0)
    metrics_container = st.container()
    examples_container = st.container()
    
    # Create placeholders for charts
    loss_chart = st.empty()
    
    # Training loop
    for epoch in range(n_epoch):
        # Create training example
        is_next = random.choice([True, False])
        tokens, segment_ids, nsp_label = create_nsp_example(corpus_list, is_next)
        
        # Apply masking for MLM
        masked_tokens, mlm_labels = apply_mlm(tokens, token2idx)
        
        # Convert to token IDs
        token_ids_input, seq_len = token_ids(masked_tokens, token2idx)
        
        # Forward pass
        model_outputs = model.forward(token_ids_input, segment_ids, seq_len)
        
        # Calculate loss based on selected tasks
        if use_mlm and use_nsp:
            total_loss, mlm_loss, nsp_loss = compute_loss(
                model_outputs, mlm_labels, nsp_label, token2idx
            )
        elif use_mlm:
            mlm_loss, _, _ = compute_loss(
                model_outputs, mlm_labels, nsp_label, token2idx
            )
            total_loss = mlm_loss
            nsp_loss = 0
        elif use_nsp:
            _, _, nsp_loss = compute_loss(
                model_outputs, mlm_labels, nsp_label, token2idx
            )
            total_loss = nsp_loss
            mlm_loss = 0
        
        st.session_state.losses["total"].append(total_loss)
        st.session_state.losses["mlm"].append(mlm_loss if use_mlm else 0)
        st.session_state.losses["nsp"].append(nsp_loss if use_nsp else 0)
        
        # Compute gradients and update parameters
        gradients = compute_gradients(model_outputs, mlm_labels, nsp_label, token2idx, model)
        model.update_parameters(gradients, learning_rate)
        
        # Update progress
        progress_bar.progress((epoch + 1) / n_epoch)
        
        # Store example for visualization
        if (epoch + 1) % 50 == 0 or epoch == n_epoch - 1:
            example = {
                "epoch": epoch + 1,
                "tokens": tokens,
                "masked_tokens": masked_tokens,
                "mlm_labels": mlm_labels,
                "nsp_label": nsp_label,
                "predictions": {}
            }
            
            # MLM predictions
            if use_mlm:
                mask_indices = [i for i, label in enumerate(mlm_labels) if label != -1]
                if mask_indices:
                    for mask_idx in mask_indices:
                        predicted_token_id = np.argmax(model_outputs['mlm_probs'][mask_idx])
                        example["predictions"][mask_idx] = {
                            "predicted": idx2token[predicted_token_id],
                            "actual": mlm_labels[mask_idx],
                            "correct": idx2token[predicted_token_id] == mlm_labels[mask_idx]
                        }
            
            # NSP prediction
            if use_nsp:
                predicted_nsp = np.argmax(model_outputs['nsp_probs'])
                example["nsp_prediction"] = {
                    "predicted": "IsNext" if predicted_nsp == 1 else "NotNext",
                    "actual": "IsNext" if nsp_label == 1 else "NotNext",
                    "correct": predicted_nsp == nsp_label
                }
            
            st.session_state.examples.append(example)
        
        # Update loss chart every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == n_epoch - 1:
            with loss_chart.container():
                fig, ax = plt.subplots(figsize=(10, 6))
                epochs = list(range(1, len(st.session_state.losses["total"]) + 1))
                
                if use_mlm and use_nsp:
                    ax.plot(epochs, st.session_state.losses["total"], label="Total Loss")
                    ax.plot(epochs, st.session_state.losses["mlm"], label="MLM Loss")
                    ax.plot(epochs, st.session_state.losses["nsp"], label="NSP Loss")
                elif use_mlm:
                    ax.plot(epochs, st.session_state.losses["mlm"], label="MLM Loss")
                elif use_nsp:
                    ax.plot(epochs, st.session_state.losses["nsp"], label="NSP Loss")
                
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
        
        # Small delay to allow UI updates
        time.sleep(0.01)
    
    st.session_state.trained = True
    
    # Final evaluation
    st.header("Evaluation Results")
    
    # NSP evaluation
    if use_nsp:
        correct_nsp = 0
        total_tests = 50
        
        for _ in range(total_tests):
            is_next = random.choice([True, False])
            tokens, segment_ids, nsp_label = create_nsp_example(corpus_list, is_next)
            masked_tokens, mlm_labels = apply_mlm(tokens, token2idx)
            token_ids_input, seq_len = token_ids(masked_tokens, token2idx)
            
            model_outputs = model.forward(token_ids_input, segment_ids, seq_len)
            
            predicted_nsp = np.argmax(model_outputs['nsp_probs'])
            if predicted_nsp == nsp_label:
                correct_nsp += 1
        
        nsp_accuracy = correct_nsp / total_tests
        st.metric("NSP Accuracy", f"{nsp_accuracy:.2%}")
    
    # MLM evaluation
    if use_mlm:
        correct_mlm = 0
        total_mlm_tests = 0
        
        for _ in range(25):  # Test on 25 examples
            tokens, segment_ids, _ = create_nsp_example(corpus_list)
            masked_tokens, mlm_labels = apply_mlm(tokens, token2idx)
            token_ids_input, seq_len = token_ids(masked_tokens, token2idx)
            
            model_outputs = model.forward(token_ids_input, segment_ids, seq_len)
            
            for i, label in enumerate(mlm_labels):
                if label != -1:  # If this position was masked
                    label_id = token2idx.get(label, token2idx.get('[UNK]', 0))
                    predicted_token_id = np.argmax(model_outputs['mlm_probs'][i])
                    if predicted_token_id == label_id:
                        correct_mlm += 1
                    total_mlm_tests += 1
        
        if total_mlm_tests > 0:
            mlm_accuracy = correct_mlm / total_mlm_tests
            st.metric("MLM Accuracy", f"{mlm_accuracy:.2%}")
    
    # Display examples
    st.header("Training Examples")
    
    for i, example in enumerate(st.session_state.examples):
        with st.expander(f"Example at Epoch {example['epoch']}"):
            # Original sentence
            original_tokens = [token for token in example['tokens'] if token not in ['[CLS]', '[SEP]']]
            st.markdown(f"**Original**: {' '.join(original_tokens)}")
            
            # Masked sentence
            masked_text = []
            for i, token in enumerate(example['masked_tokens']):
                if token == '[MASK]':
                    masked_text.append(f"**[MASK]**")
                elif token not in ['[CLS]', '[SEP]']:
                    masked_text.append(token)
            
            st.markdown(f"**Masked**: {' '.join(masked_text)}")
            
            # MLM predictions
            if use_mlm and example["predictions"]:
                st.markdown("**MLM Predictions**:")
                for idx, pred_info in example["predictions"].items():
                    color = "green" if pred_info["correct"] else "red"
                    st.markdown(f"- Position {idx}: Predicted '{pred_info['predicted']}' for '{pred_info['actual']}' - <span style='color:{color}'>{'✓' if pred_info['correct'] else '✗'}</span>", unsafe_allow_html=True)
            
            # NSP prediction
            if use_nsp and "nsp_prediction" in example:
                nsp_info = example["nsp_prediction"]
                color = "green" if nsp_info["correct"] else "red"
                st.markdown(f"**NSP Prediction**: Predicted '{nsp_info['predicted']}' (Actual: '{nsp_info['actual']}') - <span style='color:{color}'>{'✓' if nsp_info['correct'] else '✗'}</span>", unsafe_allow_html=True)

elif start_training and len(corpus_list) < 2:
    st.error("Please provide at least 2 sentences in the corpus for training.")

# Instructions for running the app
st.sidebar.markdown("---")
st.sidebar.subheader("How to Run")
st.sidebar.markdown("""
1. Adjust the corpus and parameters as needed
2. Select the pretraining tasks (MLM, NSP, or both)
3. Click 'Start Training' to begin
4. Watch the training progress and results
""")

# Footer
st.markdown("---")
st.markdown("BERT Model Pretraining Visualization | Simplified Implementation")