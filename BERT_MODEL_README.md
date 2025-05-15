# BERT Model Implementation

A comprehensive NumPy implementation of the BERT (Bidirectional Encoder Representations from Transformers) model for natural language processing tasks. This project provides a from-scratch implementation focused on educational understanding of the BERT architecture and pretraining process.

## Project Structure

```
├── python/
│   ├── model.py       # Core BERT model implementation
│   └── pretrain.py    # Pretraining utilities (MLM and NSP)
├── jupyter/
│   ├── Bert Model.ipynb              # Model architecture exploration
│   ├── Bert Fundamental.ipynb        # BERT concepts and basics
│   ├── Pretrain Bert Model MLM.ipynb # Masked Language Model pretraining
│   └── Pretrain Bert Model NSP.ipynb # Next Sentence Prediction pretraining
└── BERT_MODEL_README.md             # This documentation file
```

## Model Architecture

Our implementation includes the following components:

- **BERTModel**: The main model class with configurable parameters
- **Embedding Layer**: Combines token, segment, and positional embeddings
- **Self-Attention Mechanism**: Multi-head attention implementation
- **Feed-Forward Networks**: Position-wise feed-forward layers with GELU activation
- **Pretraining Heads**:
  - **Masked Language Modeling (MLM)**: Predicts masked tokens in a sequence
  - **Next Sentence Prediction (NSP)**: Determines if two sentences follow each other

### Key Parameters

- `vocab_size`: Size of the vocabulary (default: 30000)
- `max_seq_length`: Maximum sequence length (default: 512)
- `d_model`: Hidden size of the model (default: 768)
- `num_heads`: Number of attention heads (default: 12)
- `d_ff`: Size of feed-forward layer (default: 3072)
- `num_layers`: Number of transformer layers (default: 12)
- `for_pretraining`: Whether to include MLM and NSP heads

## Pretraining Process

The pretraining process involves two main tasks:

### 1. Masked Language Modeling (MLM)

In MLM, approximately 15% of input tokens are masked, and the model is trained to predict these masked tokens based on the context. The masking process follows these rules:

- 80% of selected tokens are replaced with [MASK]
- 10% are replaced with a random token
- 10% remain unchanged

This approach helps the model learn bidirectional context and prevents it from simply memorizing the training data.

### 2. Next Sentence Prediction (NSP)

In NSP, the model is given pairs of sentences and trained to predict whether the second sentence follows the first in the original text. This helps the model understand relationships between sentences, which is useful for tasks like question answering and natural language inference.

## Usage

### Importing the Model

```python
# Import the BERT model and helper functions
from python.model import create_bert_model, BERTModel
```

### Creating a Model for Pretraining

```python
# Create a BERT model with pretraining heads (MLM and NSP)
model = create_bert_model(
    vocab_size=30000,
    max_seq_length=512,
    d_model=768,       # Hidden size
    num_heads=12,      # Number of attention heads
    d_ff=3072,         # Feed-forward layer size
    num_layers=12,     # Number of transformer layers
    for_pretraining=True
)

# Forward pass with input token IDs
outputs = model.forward(token_ids, segment_ids, seq_len)

# Access different outputs
mlm_logits = outputs['mlm_logits']         # For masked token prediction
nsp_logits = outputs['nsp_logits']         # For next sentence prediction
sequence_output = outputs['sequence_output'] # Full sequence representations
pooled_output = outputs['pooled_output']    # [CLS] token representation
```

### Creating a Base Model (without pretraining heads)

```python
# Create just the base BERT model without MLM and NSP heads
base_model = create_bert_model(
    vocab_size=30000,
    max_seq_length=512,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    num_layers=12,
    for_pretraining=False
)

# Forward pass returns sequence output directly
sequence_output = base_model.forward(token_ids, segment_ids, seq_len)
```

### Pretraining Example

```python
# Import pretraining utilities
from python.pretrain import data_for_pretraining, tokenizer, create_nsp_example, apply_mlm

# Prepare corpus for pretraining
corpus_list = ["This is the first sentence.", "This is the second sentence.", "This is a random sentence."]

# Create vocabulary and tokenizer
vocab = data_for_pretraining(corpus_list)
token2idx, idx2token = tokenizer(vocab)

# Create NSP example
tokens, segment_ids, nsp_label = create_nsp_example(corpus_list, is_next=True)

# Apply MLM masking
masked_tokens, mlm_labels = apply_mlm(tokens, token2idx, mask_prob=0.15)

# Convert tokens to IDs
token_ids = [token2idx.get(token, token2idx['[UNK]']) for token in masked_tokens]

# Forward pass through the model
outputs = model.forward(token_ids, segment_ids, len(token_ids))
```

## Jupyter Notebooks

The project includes several Jupyter notebooks that demonstrate different aspects of the BERT model:

- **Bert Model.ipynb**: Explores the BERT architecture and implementation details
- **Bert Fundamental.ipynb**: Covers the fundamental concepts behind BERT
- **Pretrain Bert Model MLM.ipynb**: Demonstrates the Masked Language Modeling pretraining task
- **Pretrain Bert Model NSP.ipynb**: Demonstrates the Next Sentence Prediction pretraining task

## Implementation Notes

- This implementation uses NumPy for all operations, making it easy to understand the underlying mathematics
- The model is designed to be educational and follows the original BERT paper's architecture
- While not optimized for production use, this implementation provides a clear view of how BERT works
- The code includes detailed comments to explain each component's purpose and functionality

## Future Work

- Fine-tuning examples for specific NLP tasks (classification, NER, etc.)
- Performance optimizations for larger datasets
- Support for more advanced BERT variants (RoBERTa, ALBERT, etc.)

## References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/)