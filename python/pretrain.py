import numpy as np
import random
from model import create_bert_model

def data_for_pretraining(corpus_list):
    """
    Process list of sentences for pretraining tasks
    Args:
        corpus_list: List of sentences to process
    Returns:
        vocab: List of unique tokens in the corpus
    """
    # Split each sentence into tokens and flatten
    all_tokens = []
    for sentence in corpus_list:
        all_tokens.extend(sentence.split())
    
    # Create unique vocabulary
    vocab = list(set(all_tokens))
    # Add special tokens
    vocab.extend(['[CLS]', '[SEP]', '[MASK]', '[UNK]'])
    return vocab

def tokenizer(vocab):
    """
    Create mapping between tokens and indices
    """
    token2idx = {}
    idx2token = {}
    for idx, token in enumerate(vocab):
        token2idx[token] = idx
        idx2token[idx] = token
    return token2idx, idx2token

def create_nsp_example(corpus_list, is_next=True):
    """
    Create an example for the NSP task
    Args:
        corpus_list: List of sentences to sample from
        is_next: If True, sentences are consecutive; if False, random pair
    Returns:
        tokens: List of tokens for the NSP example
        segment_ids: Segment IDs for each token (0 for first sentence, 1 for second)
        nsp_label: 1 if sentences are consecutive, 0 if not
    """
    if len(corpus_list) < 2:
        raise ValueError("Need at least 2 sentences for NSP task")
    
    # Select random sentence as the first sentence
    idx = random.randint(0, len(corpus_list) - 2)
    sentence_a = corpus_list[idx].split()
    
    if is_next:
        # If is_next is True, take the next sentence
        sentence_b = corpus_list[idx + 1].split()
        nsp_label = 1
    else:
        # If is_next is False, randomly select a sentence that's not the next one
        random_idx = random.randint(0, len(corpus_list) - 1)
        while random_idx == idx + 1 or random_idx == idx:
            random_idx = random.randint(0, len(corpus_list) - 1)
        sentence_b = corpus_list[random_idx].split()
        nsp_label = 0
    
    # Construct the complete sequence with special tokens
    tokens = ['[CLS]'] + sentence_a + ['[SEP]'] + sentence_b + ['[SEP]']
    
    # Create segment IDs (0 for first sentence, 1 for second sentence)
    segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
    
    return tokens, segment_ids, nsp_label

def apply_mlm(tokens, token2idx, mask_prob=0.15):
    """
    Apply masking for the MLM task
    Args:
        tokens: List of tokens
        token2idx: Token to index mapping
        mask_prob: Probability of masking a token
    Returns:
        masked_tokens: Tokens with some replaced by [MASK]
        mlm_labels: Original tokens for masked positions, -1 for unmasked
    """
    masked_tokens = tokens.copy()
    mlm_labels = [-1] * len(tokens)
    
    # Don't mask special tokens [CLS] and [SEP]
    maskable_indices = [i for i, token in enumerate(tokens) 
                        if token not in ['[CLS]', '[SEP]']]
    
    # Number of tokens to mask
    n_to_mask = max(1, int(mask_prob * len(maskable_indices)))
    
    # Randomly select indices to mask
    mask_indices = random.sample(maskable_indices, n_to_mask)
    
    for idx in mask_indices:
        # Store the original token as the label
        mlm_labels[idx] = tokens[idx]
        
        rand = random.random()
        if rand < 0.8:
            # 80% of the time, replace with [MASK]
            masked_tokens[idx] = '[MASK]'
        elif rand < 0.9:
            # 10% of the time, replace with random word
            masked_tokens[idx] = random.choice(list(token2idx.keys()))
        # 10% of the time, keep the original word
    
    return masked_tokens, mlm_labels

def token_ids(tokens, token2idx):
    """
    Convert tokens to token IDs
    """
    token_ids = []
    for token in tokens:
        token_ids.append(token2idx.get(token, token2idx.get('[UNK]', 0)))
    return token_ids, len(token_ids)

def compute_loss(model_outputs, mlm_labels, nsp_label, token2idx):
    """
    Compute combined loss for MLM and NSP tasks
    Args:
        model_outputs: Outputs from the BERT model forward pass
        mlm_labels: Original tokens for masked positions, -1 for unmasked
        nsp_label: True label for NSP (0 or 1)
        token2idx: Token to index mapping
    """
    mlm_probs = model_outputs['mlm_probs']
    nsp_probs = model_outputs['nsp_probs']
    
    # MLM loss: only for masked positions
    mlm_loss = 0
    mask_count = 0
    
    for i, label in enumerate(mlm_labels):
        if label != -1:  # If this position was masked
            label_id = token2idx.get(label, token2idx.get('[UNK]', 0))
            mlm_loss -= np.log(mlm_probs[i, label_id] + 1e-10)
            mask_count += 1
    
    if mask_count > 0:
        mlm_loss /= mask_count
    
    # NSP loss
    nsp_loss = -np.log(nsp_probs[nsp_label] + 1e-10)
    
    # Combined loss
    total_loss = mlm_loss + nsp_loss
    
    return total_loss, mlm_loss, nsp_loss

def compute_gradients(model_outputs, mlm_labels, nsp_label, token2idx):
    """
    Compute gradients for model parameters
    """
    mlm_probs = model_outputs['mlm_probs']
    nsp_probs = model_outputs['nsp_probs']
    sequence_output = model_outputs['sequence_output']
    
    # Gradients for MLM output layer
    d_W_mlm_output = np.zeros_like(model.W_mlm_output)
    d_b_mlm_output = np.zeros_like(model.b_mlm_output)
    
    # Compute gradients for masked positions
    for i, label in enumerate(mlm_labels):
        if label != -1:  # If this position was masked
            label_id = token2idx.get(label, token2idx.get('[UNK]', 0))
            d_probs = np.zeros_like(mlm_probs[i])
            d_probs[label_id] = -1.0 / (mlm_probs[i, label_id] + 1e-10)
            d_logits = mlm_probs[i] * d_probs
            
            d_W_mlm_output += np.outer(sequence_output[i], d_logits)
            d_b_mlm_output += d_logits
    
    # Gradients for NSP output layer
    d_W_nsp_output = np.zeros_like(model.W_nsp_output)
    d_b_nsp_output = np.zeros_like(model.b_nsp_output)
    
    d_nsp_probs = np.zeros_like(nsp_probs)
    d_nsp_probs[nsp_label] = -1.0 / (nsp_probs[nsp_label] + 1e-10)
    d_nsp_logits = nsp_probs * d_nsp_probs
    
    d_W_nsp_output += np.outer(sequence_output[0], d_nsp_logits)  # [CLS] token
    d_b_nsp_output += d_nsp_logits
    
    # Return gradients
    return {
        'W_mlm_output': d_W_mlm_output,
        'b_mlm_output': d_b_mlm_output,
        'W_nsp_output': d_W_nsp_output,
        'b_nsp_output': d_b_nsp_output
    }

def train_bert(corpus_list, n_epoch=1000, d_model=16, d_ff=64, learning_rate=0.01):
    """
    Train BERT with both MLM and NSP objectives
    Args:
        corpus_list: List of sentences
        n_epoch: Number of training epochs
        d_model: Hidden size of the model
        d_ff: Size of feed-forward layer
        learning_rate: Learning rate for optimization
    """
    # Process corpus
    vocab = data_for_pretraining(corpus_list)
    vocab_size = len(vocab)
    token2idx, idx2token = tokenizer(vocab)
    
    # Initialize model
    global model  # Make model accessible to compute_gradients
    model = create_bert_model(
        vocab_size=vocab_size,
        max_seq_length=512,
        d_model=d_model,
        num_heads=1,  # Simplified for this implementation
        d_ff=d_ff,
        num_layers=1,  # Simplified for this implementation
        for_pretraining=True
    )
    
    # Training loop
    losses = {"total": [], "mlm": [], "nsp": []}
    
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
        
        # Calculate loss
        total_loss, mlm_loss, nsp_loss = compute_loss(
            model_outputs, mlm_labels, nsp_label, token2idx
        )
        
        losses["total"].append(total_loss)
        losses["mlm"].append(mlm_loss)
        losses["nsp"].append(nsp_loss)
        
        # Compute gradients and update parameters
        gradients = compute_gradients(model_outputs, mlm_labels, nsp_label, token2idx)
        model.update_parameters(gradients, learning_rate)
        
        # Logging
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epoch}")
            print(f"Total Loss: {total_loss:.4f}, MLM Loss: {mlm_loss:.4f}, NSP Loss: {nsp_loss:.4f}")
            
            # MLM accuracy check: predict random masked token
            mask_indices = [i for i, label in enumerate(mlm_labels) if label != -1]
            if mask_indices:
                random_mask_idx = random.choice(mask_indices)
                predicted_token_id = np.argmax(model_outputs['mlm_probs'][random_mask_idx])
                print(f"MLM sample - Predicted: {idx2token[predicted_token_id]}, Target: {mlm_labels[random_mask_idx]}")
            
            # NSP accuracy check
            predicted_nsp = np.argmax(model_outputs['nsp_probs'])
            print(f"NSP - Predicted: {predicted_nsp}, Target: {nsp_label}")
            print()
        
    # Final evaluation
    correct_nsp = 0
    total_tests = 100
    
    print("\nFinal Evaluation:")
    for _ in range(total_tests):
        is_next = random.choice([True, False])
        tokens, segment_ids, nsp_label = create_nsp_example(corpus_list, is_next)
        masked_tokens, mlm_labels = apply_mlm(tokens, token2idx)
        token_ids_input, seq_len = token_ids(masked_tokens, token2idx)
        
        model_outputs = model.forward(token_ids_input, segment_ids, seq_len)
        
        predicted_nsp = np.argmax(model_outputs['nsp_probs'])
        if predicted_nsp == nsp_label:
            correct_nsp += 1
    
    print(f"NSP Accuracy: {correct_nsp / total_tests:.4f}")
    
    # MLM evaluation
    correct_mlm = 0
    total_mlm_tests = 0
    
    for _ in range(50):  # Test on 50 examples
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
        print(f"MLM Accuracy: {correct_mlm / total_mlm_tests:.4f}")
    
    return model, losses

# Example usage
if __name__ == "__main__":
    # Sample corpus for testing
    corpus = [
        "the cat sat on the mat",
        "the dog played in the yard",
        "he is playing football now",
        "she was reading a book",
        "they are going to the park"
    ]
    
    model, losses = train_bert(corpus, n_epoch=500)
    print("Training complete!")