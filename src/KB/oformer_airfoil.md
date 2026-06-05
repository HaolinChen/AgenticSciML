# Operator Transformer for Compressible Flow around Airfoils

**Keywords**: ["PDE", "hyperbolic", "nonlinear", "forward-problem", "euler", "2D", "irregular", "Transformer", "finite_volume", "self-adaptive", "strong-form", "mse", "adam", "pytorch"]

**Problem:** Learning solution operators for time-dependent compressible flow (Euler equations) around irregular airfoil geometries using attention-based neural networks. The method addresses the challenge of handling irregular, unstructured grids where traditional CNN-based approaches fail, and enables discretization-invariant predictions without retraining.

**Issues addressed:**
- Handling irregular/unstructured mesh geometries where grid-based methods (CNNs, spectral methods) cannot be directly applied
- Resolution-invariance and discretization-flexibility: the model can handle varying numbers of input points and query at arbitrary locations without retraining
- Long-range spatial dependencies in flow fields using global attention rather than local convolutions
- Mesh-agnostic operator learning that generalizes across different geometries

## Key Method

Operator Transformer (OFormer) is an attention-based architecture for learning PDE solution operators. The core innovation is using **linear attention mechanisms** (Galerkin or Fourier type) combined with **Rotary Position Embeddings (RoPE)** to enable flexible, discretization-invariant operator learning.

**Architecture:**
1. **Input Encoder**: Processes input function values and coordinates using self-attention blocks with RoPE for relative positional encoding
2. **Linear Attention (Galerkin type)**: Computes attention without softmax as Z = Q(K^T V)/n with instance normalization on K and V matrices
3. **Rotary Position Embedding**: Encodes relative positions using rotation matrices, enabling the model to handle arbitrary discretizations
4. **Latent Time-Marching**: For time-dependent PDEs, propagates dynamics in latent space using recurrent MLPs with residual connections
5. **Decoder**: Maps latent representations back to physical space predictions

**Key Innovation:**  The use of RoPE allows the attention mechanism to encode relative spatial relationships without being tied to specific grid structures, making the model applicable to irregular geometries.

## Implementation

```python
# Core Linear Attention Module (Galerkin Type)
class LinearAttention(nn.Module):
    """Galerkin type attention with instance normalization on Key and Value"""
    def __init__(self,
                 dim,
                 attn_type,  # 'galerkin' or 'fourier'
                 heads=8,
                 dim_head=64,
                 relative_emb=True,  # Use Rotary Position Embedding
                 scale=1.,
                 relative_emb_dim=2,  # 2D spatial coordinates
                 min_freq=1/64,
                ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.attn_type = attn_type

        # Linear projection to Q, K, V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Instance normalization for Galerkin type
        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)  # Normalize keys
            self.v_norm = nn.InstanceNorm1d(dim_head)  # Normalize values
        else:  # fourier type
            self.q_norm = nn.InstanceNorm1d(dim_head)  # Normalize queries
            self.k_norm = nn.InstanceNorm1d(dim_head)  # Normalize keys

        # Rotary position embedding module
        if relative_emb:
            self.emb_module = RotaryEmbedding(
                dim_head // relative_emb_dim,
                min_freq=min_freq,
                scale=scale
            )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos=None):
        # x: [batch, num_points, dim]
        # pos: [batch, num_points, 2] - spatial coordinates

        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            qkv
        )

        # Apply instance normalization based on attention type
        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)  # Normalize K
            v = self.norm_wrt_domain(v, self.v_norm)  # Normalize V
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)  # Normalize Q
            k = self.norm_wrt_domain(k, self.k_norm)  # Normalize K

        # Apply Rotary Position Embedding for 2D coordinates
        if self.relative_emb:
            freqs_x = self.emb_module.forward(pos[..., 0], x.device)  # x-coordinate
            freqs_y = self.emb_module.forward(pos[..., 1], x.device)  # y-coordinate
            freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=q.shape[1])
            freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=q.shape[1])

            # Apply 2D rotary embedding: split features between x and y
            q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
            k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)

        # Linear attention: Z = Q(K^T V) / n (softmax-free)
        dots = torch.matmul(k.transpose(-1, -2), v)  # [b, h, d, d]
        out = torch.matmul(q, dots) * (1./q.shape[2])  # [b, h, n, d], normalize by n

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```

```python
# Rotary Position Embedding (RoPE)
class RotaryEmbedding(nn.Module):
    """Encodes relative positions using sinusoidal rotation matrices"""
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        # Frequency for each dimension
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.min_freq = min_freq
        self.scale = scale

    def forward(self, coordinates, device):
        # coordinates: [batch, num_points] - 1D coordinate values
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)  # Scale coordinates

        # Compute frequencies: outer product of coords and inv_freq
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]

def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    """Apply rotary embedding for 2D coordinates"""
    # Split features between x and y dimensions
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    # Apply rotation: t * cos(freq) + rotate_half(t) * sin(freq)
    return torch.cat((
        apply_rotary_pos_emb(t_x, freqs_x),
        apply_rotary_pos_emb(t_y, freqs_y)
    ), dim=-1)

def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

def rotate_half(x):
    """Rotate half the hidden dims"""
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)
```

```python
# Encoder for Irregular Spatio-Temporal Data (Airfoil)
class IrregSTEncoder2D(nn.Module):
    """Encoder for irregular grids with time dimension"""
    def __init__(self,
                 input_channels=6,     # vx, vy, pressure, density, pos_x, pos_y
                 time_window=4,        # Number of input time steps
                 in_emb_dim=128,       # Embedding dimension
                 out_chanels=128,      # Output channels
                 max_node_type=3,      # Number of node types (boundary, interior, etc.)
                 heads=1,              # Number of attention heads
                 depth=4,              # Number of transformer layers
                 res=200,              # Characteristic resolution for RoPE
                 use_ln=True,          # Use layer normalization
                ):
        super().__init__()
        self.tw = time_window

        # Temporal convolution to aggregate time window
        self.to_embedding = nn.Sequential(
            Rearrange('b t n c -> b c t n'),
            nn.Conv2d(input_channels, in_emb_dim,
                     kernel_size=(self.tw, 1), stride=(self.tw, 1), bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, in_emb_dim, kernel_size=(1, 1), bias=False),
            Rearrange('b c 1 n -> b n c'),
        )

        # Node type embedding (e.g., boundary vs interior nodes)
        self.node_embedding = nn.Embedding(max_node_type, in_emb_dim)
        self.combine_embedding = nn.Linear(in_emb_dim*2, in_emb_dim, bias=False)

        # Stack of Galerkin attention layers with varying scales
        if depth > 4:
            scales = [32, 16, 8, 8] + [1] * (depth - 4)
        else:
            scales = [32] + [16] * (depth - 2) + [1]

        self.s_transformer = TransformerCatNoCls(
            in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
            'galerkin', use_ln,
            scale=scales,  # Different scales for different layers
            min_freq=1/res,
            attention_init='orthogonal'
        )

        self.ln = nn.LayerNorm(in_emb_dim)

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, out_chanels, bias=False),
        )

    def forward(self, x, node_type, input_pos):
        # x: [batch, time, num_points, channels]
        # node_type: [batch, num_points, 1]
        # input_pos: [batch, num_points, 2]

        # Aggregate temporal information
        x = self.to_embedding(x)  # [b, n, emb_dim]

        # Add node type embedding
        x_node = self.node_embedding(node_type.squeeze(-1))
        x = self.combine_embedding(torch.cat([x, x_node], dim=-1))

        x_skip = x
        x = self.dropout(x)

        # Apply transformer with RoPE
        x = self.s_transformer.forward(x, input_pos)

        x = self.ln(x + x_skip)  # Residual connection with layer norm
        x = self.to_out(x)

        return x
```

```python
# Training configuration
def train_oformer_airfoil():
    # Build encoder-decoder model
    encoder = IrregSTEncoder2D(
        input_channels=6,     # vx, vy, prs, dns, pos_x, pos_y
        time_window=4,        # Look back 4 time steps
        in_emb_dim=128,
        out_chanels=128,
        max_node_type=3,      # Boundary, interior, airfoil nodes
        heads=1,
        depth=4,              # 4 transformer layers
        res=200,              # Characteristic resolution
        use_ln=True,
    )

    decoder = IrregSTDecoder2D(
        max_node_type=3,
        latent_channels=128,
        out_channels=4,       # vx, vy, pressure, density
        res=200,
        scale=2,
        dropout=0.1
    )

    # Adam optimizer with weight decay
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=3e-4,
        weight_decay=1e-4
    )

    # One-cycle learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=100000,
        div_factor=1e4,
        pct_start=0.3,
        final_div_factor=1e4,
    )

    # Curriculum learning: start with shorter sequences
    for iter in range(100000):
        # Gradually increase prediction horizon
        if iter < 20000:  # 20% of training
            progress = (iter * 2) / 20000
            curriculum_steps = 8 + int(max(0, progress - 1.) * (full_steps - 8) / 2.) * 2
        else:
            curriculum_steps = full_steps

        # Forward pass
        z = encoder(x, node_type, input_pos)
        pred = decoder(z, prop_pos, node_type, curriculum_steps, input_pos)

        # Loss: pointwise + region-of-interest (ROI) loss
        all_loss = pointwise_rel_loss(pred, y, p=2)
        roi_loss = roi_rel_loss(pred[..., :2]*pred[..., 2:3],
                                y[..., :2]*y[..., 2:3], pos, p=2)
        loss = all_loss + roi_loss * 2.0

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
```

## Critical Parameters

- **in_emb_dim**: 128 (embedding dimension for latent representations)
- **depth**: 4 (number of transformer layers)
- **heads**: 1 (number of attention heads)
- **time_window**: 4 (number of input time steps to condition on)
- **scales**: [32, 16, 8, 1] for RoPE across layers (controls frequency of position encoding)
- **min_freq**: 1/200 (minimum frequency for rotary embeddings, matched to grid resolution)
- **attention_init**: 'orthogonal' (initialization method for attention weights)
- **learning_rate**: 3e-4 with OneCycleLR scheduler
- **weight_decay**: 1e-4 (L2 regularization)
- **curriculum_steps**: Start with 8 rollout steps, gradually increase to full horizon
- **gradient_clipping**: 2.0 (prevents gradient explosion)
- **loss_weights**: ROI loss weighted 2x higher than pointwise loss
