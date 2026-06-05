# In-Context Operator Networks (ICON)

**Keywords**: ["ODE", "PDE", "forward-problem", "inverse-problem", "MLP", "Transformer", "meta-learning", "transfer-learning", "few-shot", "adam", "mse", "pytorch", "jax"]

**Problem:** Traditional neural network methods for solving differential equations require retraining or fine-tuning when the equation parameters or problem type changes. Even minor changes to the differential equation (e.g., modifying source terms or adding new terms) require the neural network to be retrained to approximate the new operator. This limitation prevents seamless switching between multiple operators and requires relatively large datasets for fine-tuning each new task. ICON addresses this by training a single neural network as an "operator learner" rather than a solution or operator approximator. The key innovation is the ability to learn operators from prompted examples during inference without any weight updates, inspired by in-context learning in natural language processing (GPT-style models).

**Issues addressed:** This method addresses several critical issues in operator learning for differential equations: (1) eliminates the need for retraining or fine-tuning when switching to new differential equation problems, (2) dramatically reduces data requirements for learning new operators (achieves few-shot learning with 1-5 examples), (3) enables generalization to operators beyond the training distribution, including equations of new forms not seen during training, (4) avoids potential overfitting during fine-tuning which can lead to poor out-of-distribution generalization, (5) provides the ability to seamlessly switch between or mix together multiple operator learning skills. The method leverages commonalities shared across different solution operators, enabling the network to quickly adapt to new operators with minimal examples.

## Key Method

ICON employs a transformer encoder-decoder architecture that learns to learn operators from prompted data. The key architectural innovations are:

**1. Data Representation as Key-Value Pairs:**
Continuous functions (conditions and quantities of interest) are represented as sets of key-value pairs, where keys are function inputs and values are function outputs. This flexible representation allows:
- Variable number of examples in prompts
- Variable number and choice of key-value pairs for each function
- Permutation invariance to the order of key-value pairs

**2. Prompt Construction:**
A "prompt" consists of multiple examples (condition-QoI pairs) plus a question condition, concatenated into a matrix where each column represents a key-value pair. An index column vector distinguishes between different examples and the question condition. The first row indicates different function terms, second row denotes temporal coordinates, third row for spatial coordinates, etc.

**3. Transformer Encoder-Decoder Architecture:**
- **Encoder**: Self-attention transformer that processes the entire prompt (examples + question condition) to generate an embedding representing the operator and question context
- **Decoder**: Cross-attention transformer (without self-attention layers) that uses the encoder's embedding as key/value and takes queries (keys of question QoI) to predict the corresponding QoI values
- Each query output is independently determined by its corresponding query input, enabling parallel prediction at arbitrary evaluation points

**4. Training Strategy:**
The network is trained on diverse differential equation problems (19 types including ODEs, PDEs, mean-field control problems, both forward and inverse). During training, prompts are randomly constructed with 1-5 examples and 41-50 key-value pairs per function. The training teaches the network to recognize operator patterns from examples and apply them to new conditions.

## Implementation

### Core Model Architecture

```python
import haiku as hk
from transformer import SelfAttnTransformer, CrossAttnTransformer

class SolverModel(hk.Module):
    """
    ICON model architecture consisting of:
    - kv_projection: linear layer to project prompt (examples + question condition) to embedding space
    - q_projection: linear layer to project queries (keys of question QoI) to query embedding space
    - encoder: self-attention transformer to create operator + question embedding from prompt
    - decoder: cross-attention transformer to predict QoI values from queries and embedding
    - out_projection: linear layer to project decoder output to final QoI value dimension
    """
    def __init__(self, q_size: int,          # dim for query after preprocessing
                      kv_size: int,          # dim for key-value after preprocessing
                      qoi_v_size: int,       # output dim for QoI values
                      QK_size: int,          # dim for Q and K in attention
                      V_size: int,           # dim for V in attention
                      num_heads: int,        # number of attention heads
                      num_layers: int,       # number of transformer layers
                      initializer: str = 'glorot_uniform',
                      widening_factor: int = 4):
        super(SolverModel, self).__init__()

        # Preprocessing layers
        self.kv_projection = hk.Linear(kv_size)
        self.q_projection = hk.Linear(q_size)

        # Encoder: processes prompt to learn operator
        self.encoder = SelfAttnTransformer(
            num_heads=num_heads,
            num_layers=num_layers,
            model_size=kv_size,
            QK_size=QK_size,
            V_size=V_size,
            initializer=initializer,
            widening_factor=widening_factor
        )

        # Decoder: applies learned operator to queries
        self.decoder = CrossAttnTransformer(
            num_heads=num_heads,
            num_layers=num_layers,
            model_size=q_size,
            QK_size=QK_size,
            V_size=V_size,
            initializer=initializer,
            widening_factor=widening_factor
        )

        # Postprocessing layer
        self.out_projection = hk.Linear(qoi_v_size)

    def __call__(self, prompt, mask, query):
        """
        Forward pass performing in-context operator learning

        Args:
            prompt: 2D array [prompt_len, prompt_dim], including examples and question condition
            mask: 1D array [prompt_len], masking for zero-padding
            query: 2D array [query_len, query_dim], keys where we want to evaluate QoI

        Returns:
            qoi_v: 2D array [query_len, qoi_v_size], predicted QoI values at query points
        """
        # Step 1: Project prompt to embedding space
        kv_embedding = self.kv_projection(prompt)

        # Step 2: Encode the operator from examples and question condition
        # This is where the network "learns" the operator from the prompted examples
        sys_embedding = self.encoder(kv_embedding, mask)

        # Step 3: Project queries to query embedding space
        q_embedding = self.q_projection(query)

        # Step 4: Apply learned operator to queries via cross-attention
        # Decoder uses sys_embedding as both key and value, queries as query
        out_embedding = self.decoder(
            query=q_embedding,
            key=sys_embedding,
            value=sys_embedding,
            mask=mask
        )

        # Step 5: Project to final QoI value dimension
        qoi_v = self.out_projection(out_embedding)

        return qoi_v
```

### Self-Attention Transformer (Encoder)

```python
import jax
import jax.numpy as jnp

class SelfAttnTransformer(hk.Module):
    """
    Self-attention transformer encoder that processes the prompt to extract
    operator information from examples and question condition.
    """
    def __call__(self, embeddings: jnp.ndarray, mask=None):
        """
        Args:
            embeddings: [..., T, model_size] - prompt embeddings
            mask: [..., 1, T, T] - attention mask for padding
        Returns:
            h: [..., T, model_size] - encoded operator representation
        """
        # Initialize with layer normalization
        h = layer_norm(embeddings)

        # Stack of transformer layers
        for _ in range(self.num_layers):
            # Multi-head self-attention block
            attn_block = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.QK_size,
                value_size=self.V_size,
                model_size=self.model_size
            )
            h_attn = attn_block(h, h, h, mask=mask)
            h = h + h_attn  # Residual connection
            h = layer_norm(h)

            # Feed-forward block with GELU activation
            dense_block = hk.Sequential([
                hk.Linear(self.widening_factor * self.model_size),
                jax.nn.gelu,
                hk.Linear(self.model_size),
            ])
            h_dense = dense_block(h)
            h = h + h_dense  # Residual connection
            h = layer_norm(h)

        return h
```

### Cross-Attention Transformer (Decoder)

```python
class CrossAttnTransformer(hk.Module):
    """
    Cross-attention transformer decoder that applies the learned operator
    to query points. Note: self-attention layers are removed, so each query
    output is independently determined by the operator embedding.
    """
    def __call__(self, query: jnp.ndarray, key: jnp.ndarray,
                 value: jnp.ndarray, mask=None):
        """
        Args:
            query: [t, model_size] - query embeddings (keys of question QoI)
            key: [T, key_size] - operator embedding from encoder
            value: [T, value_size] - operator embedding from encoder
            mask: [1, t, T] - attention mask
        Returns:
            query_norm: [t, model_size] - predicted QoI value embeddings
        """
        # Normalize inputs
        query_norm = layer_norm(query)
        key_norm = layer_norm(key)
        value_norm = layer_norm(value)

        # Stack of decoder layers
        for i in range(self.num_layers):
            # Cross-attention block (no self-attention!)
            # This allows parallel, independent predictions for each query
            attn_block = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.QK_size,
                value_size=self.V_size,
                model_size=self.model_size
            )

            # Cross-attend to operator embedding
            this_query = attn_block(
                query=query_norm,
                key=key_norm,
                value=value_norm,
                mask=mask
            )
            query_norm = layer_norm(this_query + query_norm)

            # Feed-forward block
            dense_block = hk.Sequential([
                hk.Linear(self.widening_factor * self.model_size),
                jax.nn.gelu,
                hk.Linear(self.model_size),
            ])
            this_query = dense_block(query_norm)
            query_norm = layer_norm(this_query + query_norm)

        return query_norm
```

### Training Loop

```python
def loss_fn(params, rng_key, prompt, mask, query, query_mask, ground_truth):
    """
    Mean squared error loss between predictions and ground truth.

    Args:
        params: model parameters
        prompt: [prompt_len, prompt_dim] - examples and question condition
        mask: [prompt_len] - padding mask for prompt
        query: [query_len, query_dim] - evaluation points for QoI
        query_mask: [query_len] - mask for query padding
        ground_truth: [query_len, qoi_v_dim] - true QoI values at query points
    """
    out = predict_fn(params, rng_key, prompt, mask, query)
    # MSE loss, only computed where query_mask is True
    loss = jnp.mean((out - ground_truth)**2, where=query_mask[..., None])
    return loss

# Training configuration from run.py
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),           # Gradient clipping
    optax.adamw(learning_rate=0.0001,         # Peak learning rate
                weight_decay=0.0001)          # Weight decay
)

# Training hyperparameters:
# - Batch size: 32
# - Number of examples in prompt: randomly 1-5
# - Key-value pairs per function: randomly 41-50
# - Model dimension: 128 (for smaller models) or 256 (for larger models)
# - Number of attention heads: 8
# - Number of transformer layers: 4-6
# - Training epochs: 20-100
```

### Prompt Construction

```python
def pad_and_concat(demos, quest_cond, k_dim, v_dim, cond_len, qoi_len, demo_num):
    """
    Construct prompt matrix from examples and question condition.

    Args:
        demos: list of (demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v) tuples
        quest_cond: (quest_cond_k, quest_cond_v) - question condition
        k_dim: max dimension for keys
        v_dim: max dimension for values
        cond_len: max length for each demo condition
        qoi_len: max length for each demo QoI
        demo_num: max number of demos

    Returns:
        prompt: [total_len, k_dim + v_dim + demo_num + 1] - concatenated prompt
        mask: [total_len] - mask indicating non-padded entries
    """
    # Total length: demo_num * (cond_len + qoi_len) + cond_len
    total_len = demo_num * (cond_len + qoi_len) + cond_len
    prompt = np.zeros((total_len, k_dim + v_dim + demo_num + 1))
    mask = np.zeros((total_len,))

    # Fill in examples
    for i, (demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v) in enumerate(demos):
        # Condition part
        start_idx = i * (cond_len + qoi_len)
        cond_slice = prompt[start_idx:start_idx + len(demo_cond_k)]
        cond_slice[:, :demo_cond_k.shape[1]] = demo_cond_k
        cond_slice[:, k_dim:k_dim + demo_cond_v.shape[1]] = demo_cond_v
        cond_slice[:, k_dim + v_dim + i] = 1.0  # Index for this demo's condition
        mask[start_idx:start_idx + len(demo_cond_k)] = 1.0

        # QoI part
        qoi_start = start_idx + cond_len
        qoi_slice = prompt[qoi_start:qoi_start + len(demo_qoi_k)]
        qoi_slice[:, :demo_qoi_k.shape[1]] = demo_qoi_k
        qoi_slice[:, k_dim:k_dim + demo_qoi_v.shape[1]] = demo_qoi_v
        qoi_slice[:, k_dim + v_dim + i] = -1.0  # Index for this demo's QoI
        mask[qoi_start:qoi_start + len(demo_qoi_k)] = 1.0

    # Fill in question condition
    quest_cond_k, quest_cond_v = quest_cond
    quest_start = demo_num * (cond_len + qoi_len)
    quest_slice = prompt[quest_start:quest_start + len(quest_cond_k)]
    quest_slice[:, :quest_cond_k.shape[1]] = quest_cond_k
    quest_slice[:, k_dim:k_dim + quest_cond_v.shape[1]] = quest_cond_v
    quest_slice[:, k_dim + v_dim + demo_num] = 1.0  # Index for question
    mask[quest_start:quest_start + len(quest_cond_k)] = 1.0

    return prompt, mask
```

## Critical Parameters

**Architecture Parameters:**
- `hidden_dim`: 128-256 (embedding dimension for encoder/decoder)
- `num_heads`: 8 (number of attention heads)
- `num_layers`: 4-6 (number of transformer layers in encoder and decoder)
- `widening_factor`: 4 (expansion factor for feed-forward networks)
- `initializer`: 'glorot_uniform' (weight initialization strategy)

**Training Parameters:**
- `train_peak_lr`: 0.0001 (peak learning rate)
- `train_weight_decay`: 0.0001 (AdamW weight decay)
- `train_gnorm_clip`: 1.0 (gradient norm clipping threshold)
- `batch_size`: 32 (batch size per device)
- `epochs`: 20-100 (total training epochs)

**Data Configuration:**
- `demo_num`: 1-5 (number of examples in each prompt, randomly selected during training)
- `cond_len`, `qoi_len`: 50 (max length for condition and QoI representations)
- `cond_len_in_use`: 41-51 (actual number of key-value pairs used, randomly selected)
- `qoi_len_in_use`: 41-51 (actual number of key-value pairs for QoI)
- `k_dim`: 2-3 (dimension of keys, depends on problem - 1D vs 2D spatial)
- `v_dim`: 1 (dimension of values for single scalar field)
- `qoi_v_dim`: 1 (output dimension)

**Key Design Choices:**
- Remove self-attention in decoder to enable parallel, independent predictions
- Use index vectors to distinguish examples and question condition in prompt
- Zero-padding with masks to handle variable-length inputs
- Layer normalization after each sublayer with residual connections
- GELU activation in feed-forward networks
