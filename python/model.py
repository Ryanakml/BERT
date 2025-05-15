import numpy as np

class BERTModel:
    def __init__(self, vocab_size, max_seq_length=512, d_model=768, num_heads=12, d_ff=3072, num_layers=1, for_pretraining=True):
        """
        Initialize BERT model with configurable parameters
        
        Args:
            vocab_size: Size of the vocabulary
            max_seq_length: Maximum sequence length
            d_model: Hidden size of the model
            num_heads: Number of attention heads (not implemented in this simplified version)
            d_ff: Size of feed-forward layer
            num_layers: Number of transformer layers (simplified to 1 in this implementation)
            for_pretraining: Whether to include MLM and NSP heads
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.d_ff = d_ff
        self.for_pretraining = for_pretraining
        self.num_segments = 2  # For segment embeddings (sentence A and B)
        
        # Initialize model parameters
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """Initialize all model parameters"""
        # Embedding matrices
        self.token_embed_matrix = self._create_token_embedding(0.01, self.d_model, self.vocab_size)
        self.segment_embed_matrix = self._create_segment_embedding(0.01, self.d_model, self.num_segments)
        self.position_embed_matrix = self._create_position_embedding(self.max_seq_length, self.d_model)
        
        # Attention weights
        self.Wq = np.random.randn(self.d_model, self.d_model) * 0.01
        self.Wk = np.random.randn(self.d_model, self.d_model) * 0.01
        self.Wv = np.random.randn(self.d_model, self.d_model) * 0.01
        
        # Feed-forward weights
        self.W1 = np.random.randn(self.d_model, self.d_ff) * 0.01
        self.W2 = np.random.randn(self.d_ff, self.d_model) * 0.01
        
        # Output layers
        if self.for_pretraining:
            # MLM output layer
            self.W_mlm_output = np.random.randn(self.d_model, self.vocab_size) * 0.01
            self.b_mlm_output = np.zeros(self.vocab_size)
            
            # NSP output layer
            self.W_nsp_output = np.random.randn(self.d_model, 2) * 0.01  # 2 classes for NSP
            self.b_nsp_output = np.zeros(2)
    
    def _create_token_embedding(self, scale, d_model, vocab_size):
        """Create token embedding matrix"""
        np.random.seed(42)
        return np.random.rand(vocab_size, d_model) * scale
    
    def _create_segment_embedding(self, scale, d_model, n_segments=2):
        """Create segment embedding matrix"""
        np.random.seed(43)
        return np.random.rand(n_segments, d_model) * scale
    
    def _create_position_embedding(self, max_seq_len, d_model):
        """Create position embedding matrix using sine and cosine functions"""
        pos = np.arange(max_seq_len).reshape(max_seq_len, 1)
        i = np.arange(d_model)
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return pe
    
    def _embedding_output(self, token_ids, segment_ids, seq_len):
        """Combine token, segment, and position embeddings"""
        # One-hot encode the tokens
        input_matrix = np.zeros((seq_len, self.vocab_size))
        for i, token_id in enumerate(token_ids[:seq_len]):
            input_matrix[i, token_id] = 1
        
        # Get embeddings
        token_embeds = input_matrix @ self.token_embed_matrix
        segment_embeds = np.array([self.segment_embed_matrix[segment_id] for segment_id in segment_ids[:seq_len]])
        position_embeds = self.position_embed_matrix[:seq_len]
        
        return token_embeds + segment_embeds + position_embeds
    
    def _attention_output(self, x):
        """Compute self-attention"""
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        d_k = Q.shape[-1]
        scaled = np.matmul(Q, K.transpose()) / np.sqrt(d_k)
        softmax = np.exp(scaled) / np.sum(np.exp(scaled), axis=-1, keepdims=True)
        return np.matmul(softmax, V)
    
    def _add_and_norm(self, x, attention_output):
        """Add and normalize"""
        eps = 1e-6
        avg = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        norm = (x - avg) / (std + eps)
        return attention_output + norm
    
    def _ffn_output(self, norm_output):
        """Feed-forward network computation"""
        ffn_output = np.matmul(norm_output, self.W1)
        relu = np.maximum(0, ffn_output)
        return np.matmul(relu, self.W2)
    
    def forward(self, token_ids, segment_ids=None, seq_len=None):
        """Forward pass through the BERT model"""
        if seq_len is None:
            seq_len = len(token_ids)
            
        if segment_ids is None:
            # Default: all tokens belong to first segment
            segment_ids = np.zeros(seq_len, dtype=int)
        
        # Embedding layer
        embed_output = self._embedding_output(token_ids, segment_ids, seq_len)
        
        # Self-attention block
        attn_output = self._attention_output(embed_output)
        norm_output = self._add_and_norm(embed_output, attn_output)
        
        # Feed-forward block
        ffn_out = self._ffn_output(norm_output)
        sequence_output = self._add_and_norm(norm_output, ffn_out)
        
        # Pooled output (for NSP and other tasks)
        pooled_output = sequence_output[0]  # [CLS] token representation
        
        if not self.for_pretraining:
            return sequence_output
        
        # MLM head: predict token for each position
        mlm_logits = sequence_output @ self.W_mlm_output + self.b_mlm_output
        mlm_probs = np.exp(mlm_logits) / np.sum(np.exp(mlm_logits), axis=-1, keepdims=True)
        
        # NSP head: use [CLS] token representation for next sentence prediction
        nsp_logits = pooled_output @ self.W_nsp_output + self.b_nsp_output
        nsp_probs = np.exp(nsp_logits) / np.sum(np.exp(nsp_logits))
        
        return {
            'sequence_output': sequence_output,
            'pooled_output': pooled_output,
            'mlm_logits': mlm_logits,
            'mlm_probs': mlm_probs,
            'nsp_logits': nsp_logits,
            'nsp_probs': nsp_probs
        }
    
    def get_parameters(self):
        """Get all model parameters as a dictionary"""
        params = {
            'token_embed_matrix': self.token_embed_matrix,
            'segment_embed_matrix': self.segment_embed_matrix,
            'Wq': self.Wq, 'Wk': self.Wk, 'Wv': self.Wv,
            'W1': self.W1, 'W2': self.W2,
        }
        
        if self.for_pretraining:
            params.update({
                'W_mlm_output': self.W_mlm_output,
                'b_mlm_output': self.b_mlm_output,
                'W_nsp_output': self.W_nsp_output,
                'b_nsp_output': self.b_nsp_output
            })
            
        return params
    
    def update_parameters(self, gradients, learning_rate=0.01):
        """Update model parameters using gradients"""
        for param_name, grad in gradients.items():
            if hasattr(self, param_name):
                setattr(self, param_name, getattr(self, param_name) - learning_rate * grad)


def create_bert_model(vocab_size, max_seq_length=512, d_model=768, num_heads=12, d_ff=3072, num_layers=12, for_pretraining=True):
    """Helper function to create a BERT model with standard parameters"""
    return BERTModel(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        for_pretraining=for_pretraining
    )