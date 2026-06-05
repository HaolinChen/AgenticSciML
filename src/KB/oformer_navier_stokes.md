# Operator Transformer for 2D Navier-Stokes Equations

**Keywords**: ["PDE", "parabolic", "nonlinear", "forward-problem", "navier-stokes", "2D", "regular", "periodic", "Transformer", "spectral-method", "strong-form", "relative-l2", "adam", "pytorch"]

**Problem:** Learning solution operators for 2D time-dependent Navier-Stokes equations, mapping initial conditions to future time steps on regular grids. The method predicts long-term vorticity dynamics on periodic domains, enabling multi-step forecasting through latent time-marching with discretization-invariant predictions.

**Issues addressed:**
- Long-term temporal prediction stability through latent space propagation
- Curriculum learning to handle long rollout horizons without training instabilities
- Gradient-based regularization for capturing fine-scale turbulent structures
- Regular grid handling with periodic boundary conditions using linear attention

## Key Method

Operator Transformer with spatio-temporal attention for Navier-Stokes equations. Uses separate spatial and temporal Galerkin attention blocks with multi-scale RoPE for spatial coordinates. The decoder employs latent time-marching through recurrent MLPs to propagate dynamics forward in time.

**Architecture:**
1. **Spatio-Temporal Encoder**: Alternating spatial and temporal attention with multi-scale RoPE
2. **Cross-Attention Decoder**: Maps latent representation to query positions with Gaussian Fourier Features
3. **Latent Propagator**: Recurrent MLP with residual connections for time-marching in latent space
4. **Multi-step Rollout**: Generates sequences by iteratively propagating latent states

## Implementation

```python
# Spatio-Temporal Encoder for Navier-Stokes
class SpatialTemporalEncoder2D(nn.Module):
    """Encoder with alternating spatial and temporal attention"""
    def __init__(self,
                 input_channels,      # Vorticity + coordinates
                 in_emb_dim,          # 128
                 out_seq_emb_dim,     # 128
                 heads,               # 4
                 depth,               # 2
                ):
        super().__init__()

        # Linear embedding without nonlinearity
        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        # Spatio-temporal transformer with multi-scale RoPE
        if depth > 4:
            scales = [32, 16, 8, 8] + [1] * (depth - 4)
        else:
            scales = [32] + [16]*(depth-2) + [1]

        # Alternating spatial and temporal attention
        self.s_transformer = STTransformerCatNoCls(
            in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
            attn_type='galerkin',    # Galerkin-type attention
            use_ln=True,             # Layer normalization
            scale=scales,            # Multi-scale for spatial attention
            dropout=0.,
            relative_emb_dim=2,      # 2D spatial coordinates
            min_freq=1/64,
            attention_init='orthogonal'
        )

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False)
        )

    def forward(self, x, input_pos):
        # x: [b, t, n, c] - time series of vorticity + coordinates
        # input_pos: [b, n, 2] - spatial coordinates

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x  # [b, n, t, c]
```

```python
# Spatio-Temporal Transformer with Alternating Attention
class STTransformerCatNoCls(nn.Module):
    """Transformer with separate spatial and temporal attention blocks"""
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type='galerkin',
                 use_ln=False,
                 scale=32,              # Multi-scale for spatial
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal'):
        super().__init__()

        if isinstance(scale, int):
            scale = [scale] * depth

        self.layers = nn.ModuleList([])

        for d in range(depth):
            # Spatial attention with 2D RoPE
            spatial_attn = LinearAttention(
                dim, attn_type,
                heads=heads, dim_head=dim_head, dropout=dropout,
                relative_emb=True, scale=scale[d],
                relative_emb_dim=2,      # 2D coordinates
                min_freq=min_freq,
                init_method=attention_init
            )

            # Temporal attention with 1D RoPE
            temporal_attn = LinearAttention(
                dim, attn_type,
                heads=heads, dim_head=dim_head, dropout=dropout,
                relative_emb=True, scale=1,
                relative_emb_dim=1,      # 1D time
                min_freq=1,
                init_method=attention_init
            )

            if use_ln:
                self.layers.append(nn.ModuleList([
                    # Spatial block
                    nn.ModuleList([
                        nn.LayerNorm(dim),
                        spatial_attn,
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]),
                    # Temporal block
                    nn.ModuleList([
                        nn.LayerNorm(dim),
                        temporal_attn,
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]),
                ]))

    def forward(self, x, pos_embedding):
        # x: [b, t, n, c]
        # pos_embedding: [b, n, 2] - spatial coordinates
        b, t, n, c = x.shape

        # Broadcast position embeddings for all time steps
        pos_embedding = repeat(pos_embedding, 'b n c -> (b repeat) n c', repeat=t)

        # Time embedding: simple integer indices
        temp_embedding = torch.arange(t).float().to(x.device).view(1, t, 1)
        temp_embedding = repeat(temp_embedding, '() t c -> b t c', b=b*n)

        for layer_no, (spa_attn, temp_attn) in enumerate(self.layers):
            # Reshape for spatial attention: [b*t, n, c]
            if layer_no == 0:
                x = rearrange(x, 'b t n c -> (b t) n c')
            else:
                x = rearrange(x, '(b n) t c -> (b t) n c', n=n)

            # Spatial attention
            [ln, attn, ffn] = spa_attn
            x = ln(x)
            x = attn(x, pos_embedding) + x  # Residual connection
            x = ffn(x) + x

            # Reshape for temporal attention: [b*n, t, c]
            x = rearrange(x, '(b t) n c -> (b n) t c', t=t)

            # Temporal attention
            [ln, attn, ffn] = temp_attn
            x = ln(x)
            x = attn(x, temp_embedding, not_assoc=True) + x  # Residual
            x = ffn(x) + x

            # Final reshape
            if layer_no == len(self.layers) - 1:
                x = rearrange(x, '(b n) t c -> b n t c', n=n)
        return x
```

```python
# Decoder with Latent Time-Marching
class PointWiseDecoder2D(nn.Module):
    """Decoder with latent space propagation for multi-step rollout"""
    def __init__(self,
                 latent_channels,     # 128
                 out_channels,        # 1 (vorticity)
                 out_steps,           # 10 steps per propagation
                 propagator_depth,    # 2
                 scale=8,             # Fourier feature scale
                 dropout=0.,
                ):
        super().__init__()
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        # Coordinate projection with Gaussian Fourier Features
        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, latent_channels//2, scale=scale),
            nn.Linear(latent_channels, latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(latent_channels, latent_channels//2, bias=False),
        )

        # Cross-attention decoder with RoPE
        self.decoding_transformer = CrossFormer(
            latent_channels//2, 'galerkin', 4,
            latent_channels//2, latent_channels//2,
            relative_emb=True,
            scale=16.,
            relative_emb_dim=2,
            min_freq=1/64
        )

        self.expand_feat = nn.Linear(latent_channels//2, latent_channels)

        # Latent propagator: recurrent MLP with residual connections
        self.propagator = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(latent_channels),
                nn.Sequential(
                    nn.Linear(latent_channels + 2, latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(latent_channels, latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(latent_channels, latent_channels, bias=False)
                )
            ])
            for _ in range(propagator_depth)
        ])

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(latent_channels),
            nn.Linear(latent_channels, latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(latent_channels // 2, latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_channels//2, out_channels * out_steps, bias=True)
        )

    def propagate(self, z, pos):
        """Propagate latent state forward in time"""
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z  # Residual
        return z

    def decode(self, z):
        """Decode latent state to solution"""
        return self.to_out(z)

    def rollout(self, z, propagate_pos, forward_steps, input_pos):
        """Multi-step rollout through latent time-marching"""
        # z: [b, n, c]
        # propagate_pos: [b, n, 2]
        # forward_steps: total number of time steps to predict
        # input_pos: [b, n, 2] - input coordinates for cross-attention

        history = []

        # Cross-attention to decode latent representation
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)

        # Forward dynamics in latent space
        for step in range(forward_steps // self.out_steps):
            z = self.propagate(z, propagate_pos)  # Propagate latent state
            u = self.decode(z)                    # Decode to solution
            history.append(rearrange(u, 'b n (t c) -> b (t c) n',
                                    c=self.out_channels, t=self.out_steps))

        # Concatenate all predictions
        history = torch.cat(history, dim=-2)  # [b, length_of_history*c, n]
        return history
```

```python
# Training with Curriculum Learning and Gradient Regularization
def train_navier_stokes():
    encoder = SpatialTemporalEncoder2D(
        input_channels=3,        # Vorticity + x, y coordinates
        in_emb_dim=128,
        out_seq_emb_dim=128,
        heads=4,
        depth=2
    )

    decoder = PointWiseDecoder2D(
        latent_channels=128,
        out_channels=1,          # Vorticity
        out_steps=10,            # Predict 10 steps at a time
        propagator_depth=2,
        scale=8,
        dropout=0.0
    )

    # AdamW optimizer with weight decay
    enc_optim = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=1e-4)
    dec_optim = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-4)

    # OneCycleLR scheduler
    enc_scheduler = OneCycleLR(enc_optim, max_lr=1e-4, total_steps=5000,
                               div_factor=1e4, final_div_factor=1e4)
    dec_scheduler = OneCycleLR(dec_optim, max_lr=1e-4, total_steps=5000,
                               div_factor=1e4, final_div_factor=1e4)

    for iter in range(5000):
        # Curriculum learning: gradually increase rollout length
        if iter < int(0.2 * 5000):  # First 20% of training
            progress = (iter * 2) / (5000 * 0.2)
            curriculum_steps = 8 + int(max(0, progress - 1.) * (40 - 8) / 2.) * 2
            gt = gt[:, :curriculum_steps, :]
        else:
            curriculum_steps = 40  # Full sequence length

        # Forward pass
        z = encoder.forward(in_seq, input_pos)
        pred = decoder.rollout(z, prop_pos, curriculum_steps, input_pos)

        # Relative L2 loss
        pred_loss = rel_l2norm_loss(pred, gt)

        # Optional: Gradient-based regularization for turbulence
        if use_grad:
            gt_grad_x, gt_grad_y = central_diff(gt)
            pred_grad_x, pred_grad_y = central_diff(pred)
            grad_loss = rel_l2norm_loss(pred_grad_x, gt_grad_x) + \
                       rel_l2norm_loss(pred_grad_y, gt_grad_y)
            loss = pred_loss + 5e-2 * grad_loss
        else:
            loss = pred_loss

        # Backward and optimize
        enc_optim.zero_grad()
        dec_optim.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0)

        enc_optim.step()
        dec_optim.step()
        enc_scheduler.step()
        dec_scheduler.step()
```

```python
# Relative L2 Norm Loss
def rel_l2norm_loss(pred, target):
    """Relative L2 norm loss for scale-invariant prediction"""
    diff_norms = torch.norm(pred.reshape(pred.shape[0], -1) -
                           target.reshape(target.shape[0], -1), p=2, dim=1)
    target_norms = torch.norm(target.reshape(target.shape[0], -1), p=2, dim=1)
    return torch.mean(diff_norms / (target_norms + 1e-8))
```

```python
# Central Difference for Gradient Computation
def central_diff(x):
    """Compute spatial gradients using central difference (periodic BC)"""
    # x: [batch, seq_len, n] where n = h*w
    x = rearrange(x, 'b t (h w) -> b t h w', h=64, w=64)
    h_step = 1./64.

    # Circular padding for periodic boundaries
    x = F.pad(x, (1, 1, 1, 1), mode='circular')

    # Central difference: (f(x+h) - f(x-h)) / 2h
    grad_x = (x[..., 1:-1, 2:] - x[..., 1:-1, :-2]) / (2*h_step)
    grad_y = (x[..., 2:, 1:-1] - x[..., :-2, 1:-1]) / (2*h_step)

    return grad_x, grad_y
```

## Critical Parameters

- **in_emb_dim**: 128 (embedding dimension)
- **out_seq_emb_dim**: 128 (latent representation dimension)
- **heads**: 4 (multi-head attention)
- **depth**: 2 (shallow transformer for Navier-Stokes)
- **attn_type**: 'galerkin' (instance norm on K and V)
- **spatial_scales**: [32, 16] (multi-scale RoPE for 2 layers)
- **temporal_scale**: 1 (single scale for time)
- **use_ln**: True (layer normalization for stability)
- **out_steps**: 10 (number of steps predicted per propagation)
- **propagator_depth**: 2 (depth of latent propagator MLP)
- **fourier_frequency**: 8 (scale for Gaussian Fourier Features)
- **learning_rate**: 1e-4 with AdamW
- **weight_decay**: 1e-4 (L2 regularization)
- **scheduler**: OneCycleLR (div_factor=1e4, final_div_factor=1e4)
- **gradient_clipping**: 2.0 (prevents gradient explosion)
- **curriculum_steps**: Start with 8 steps, gradually increase to 40
- **curriculum_ratio**: 0.2 (20% of training uses curriculum)
- **aug_ratio**: Optional random cropping augmentation
- **use_grad**: Optional gradient-based regularization (weight=5e-2)
- **batch_size**: 16
- **training_iterations**: 5,000
- **resolution**: 64×64 grid with periodic boundaries
- **in_seq_len**: 10 (input time steps)
- **out_seq_len**: 40 (output time steps)
