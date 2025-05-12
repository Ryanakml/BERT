def data(corpus):
    vocab = corpus.split()
    vocab.insert(0, '[CLS]')
    vocab.extend(['[SEP]', '[MASK]'])
    return vocab

def tokennizer(vocab):
    token2idx = {}
    idx2token = {}
    for idx, token in enumerate(vocab):
        token2idx[token] = idx
        idx2token[idx] = token
    return token2idx, idx2token

def masking(corpus):
    tokens = corpus.split()
    tokens.insert(0, '[CLS]')
    tokens.extend(['[SEP]', '[MASK]'])
    labels = 'playing'
    tokens[tokens.index(labels)] = '[MASK]'
    return tokens

def token_ids(tokens, token2idx):
    token_ids = []
    for token in tokens:
        token_ids.append(token2idx[token])
    return token_ids, len(token_ids)

def embedding(scale, d_model, vocab_size):
    np.random.seed(42)
    embed_matrix = np.random.rand(vocab_size, d_model) * scale
    return embed_matrix

def position_embedding(seq_len, d_model):
    pos = np.arange(seq_len).reshape(seq_len,1)
    i = np.arange(d_model)
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return pe

def embedding_output(embed_matrix, position_embedding):
    return embed_matrix + position_embedding


def attention_weights(d_model):
    Wq = np.random.randn(d_model, d_model) * 0.01
    Wk = np.random.randn(d_model, d_model) * 0.01
    Wv = np.random.randn(d_model, d_model) * 0.01

    return Wq, Wk, Wv

def attention_output(x, Wq, Wk, Wv):
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    d_k = Q.shape[-1]
    scaled = np.matmul(Q, K.transpose()) / np.sqrt(d_k)
    softmax = np.exp(scaled) / np.sum(np.exp(scaled), axis=-1, keepdims=True)

    return np.matmul(softmax, V)

def add_and_norm(x, attention_output):
    eps = 1e1
    avg = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    norm = (x - avg) / (std + eps)
    norm_output = attention_output + norm
    return norm_output
    
def ffn_weights(d_model, d_ff):
    W1 = np.random.randn(d_model, d_ff) * 0.01
    W2 = np.random.randn(d_ff, d_model) * 0.01
    return W1, W2

def ffn_output(norm_output, W1, W2):
    ffn_output = np.matmul(norm_output, W1)
    relu = np.maximum(0, ffn_output)
    ffn_output = np.matmul(relu, W2)
    return ffn_output

def initialize_model(d_model, d_ff, vocab_size):
    Wq, Wk, Wv = attention_weights(d_model)
    W1, W2 = ffn_weights(d_model, d_ff)
    embed_matrix = embedding(0.01, d_model, vocab_size)
    W_output = np.random.randn(d_model, vocab_size) * 0.01
    b_output = np.zeros(vocab_size)
    return {
        'Wq': Wq, 'Wk': Wk, 'Wv': Wv,
        'W1': W1, 'W2': W2,
        'embed_matrix': embed_matrix,
        'W_output': W_output,
        'b_output': b_output
    }

def forward_pass(token_ids_input, seq_len, model_params, d_model, vocab_size):
    input_matrix = np.zeros((seq_len, len(model_params['embed_matrix'])))
    for i, token_id in enumerate(token_ids_input):
        input_matrix[i,token_id] = 1
    
    token_embeds = input_matrix @ model_params['embed_matrix']
    position_embeds = position_embedding(seq_len, d_model)
    embed_output = token_embeds + position_embeds

    attn_output = attention_output(embed_output, model_params['Wq'], model_params['Wk'], model_params['Wv'])
    norm_output = add_and_norm(embed_output, attn_output)
    ffn_out = ffn_output(norm_output, model_params['W1'], model_params['W2'])
    ffn_norm_output = add_and_norm(norm_output, ffn_out)

    logits = ffn_norm_output @ model_params['W_output'] + model_params['b_output']

    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    return probs, ffn_norm_output

def loss(probs, target_token_id, mask_id):
    mask_probs = probs[mask_id]
    loss = -np.log(mask_probs[target_token_id] + 1e-10)
    return loss

def backward_propagation(loss, probs, target_token_id, ffn_norm_output, model_params, token_ids, mask_id, lr = 0.01):
    d_W_output = np.zeros_like(model_params['W_output'])
    d_b_output = np.zeros_like(model_params['b_output'])
    
    d_probs = np.zeros_like(probs[mask_id])
    d_probs[target_token_id] = -1.0 / (probs[mask_id, target_token_id] + 1e-10)
    
    d_logits = probs[mask_id] * d_probs
    
    d_W_output += np.outer(ffn_norm_output[mask_id], d_logits)
    d_b_output += d_logits

    model_params['W_output'] -= lr * d_W_output
    model_params['b_output'] -= lr * d_b_output

    return model_params

def train(corpus, masked_token, n_epoch=1000):
    d_model = 16
    d_ff = 64
    vocab = data(corpus)
    vocab_size = len(vocab)
    token2idx, idx2token = tokennizer(vocab)
    tokens = masking(corpus)
    token_ids_input, seq_len = token_ids(tokens, token2idx)
    mask_position = tokens.index('[MASK]')
    target_idx = token2idx[masked_token]
    model_params = initialize_model(d_model, d_ff, vocab_size)

    losses = []
    for epoch in range(n_epoch):
        probs, ffn_norm_output = forward_pass(token_ids_input, seq_len, model_params, d_model, vocab_size)
        loss_value = loss(probs, target_idx, mask_position)
        losses.append(loss_value)

        model_params = backward_propagation(loss_value, probs, target_idx, ffn_norm_output, model_params, token_ids_input, mask_position, lr=0.01)
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epoch}, Loss: {loss_value}")
            
            predicted_idx = np.argmax(probs[mask_position])
            print(f"Current prediction: {idx2token[predicted_idx]}, Target: {idx2token[target_idx]}")
        
    final_probs, _ = forward_pass(token_ids_input, seq_len, model_params, d_model, vocab_size)
    predicted_idx = np.argmax(final_probs[mask_position])
    print(f"\nFinal prediction: {idx2token[predicted_idx]}")
    print(f"Target token: {idx2token[target_idx]}")

    return model_params, losses

