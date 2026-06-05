# Operator Transformer for Darcy Flow

**Keywords**: ["PDE", "elliptic", "linear", "forward-problem", "darcy", "2D", "regular", "dirichlet", "Transformer", "finite_difference", "strong-form", "relative-l2", "adam", "pytorch"]

**Problem:** Learning solution operators for 2D steady-state Darcy flow equations, mapping diffusion coefficient fields a(x) to pressure/flux solutions u(x) on regular grids. Darcy flow models fluid flow through porous media and is fundamental to subsurface hydrology, oil reservoir simulation, and groundwater modeling.

**Issues addressed:**
- Resolution-invariance: trained on one resolution, can query at different resolutions without retraining
- Handling spatially-varying coefficients with complex patterns (permeability fields from Gaussian Random Fields)
- Learning operators without explicit PDE knowledge, purely data-driven
- Computational efficiency through softmax-free linear attention (O(n) vs O(n²))

## Key Method

Operator Transformer uses Galerkin-type linear attention with 2D Rotary Position Embeddings. For the steady-state Darcy problem, the model directly learns the mapping from coefficient function to solution without time-marching.

**Architecture:**
1. **2D Spatial Encoder**: Processes coefficient field a(x,y) and coordinates using multi-head Galerkin attention
2. **Multi-scale RoPE**: Different scales [res, res/4, 1, ...] across layers to capture multi-scale features
3. **Point-wise Decoder**: Maps latent representation to solution u(x,y)

## Implementation

```python
# 2D Spatial Encoder for Darcy Flow
class SpatialEncoder2D(nn.Module):
    def __init__(self,
                 input_channels=3,      # a(x,y) + coordinates (x,y)
                 in_emb_dim=96,
                 out_seq_emb_dim=256,   # Latent dimension
                 heads=4,               # Multi-head attention
                 depth=4,               # Transformer layers
                 res=64,                # Grid resolution (64x64)
                 use_ln=True,           # Layer normalization
                ):
        super().__init__()

        # Input embedding
        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(0.05)

        # Galerkin attention with multi-scale RoPE
        self.s_transformer = TransformerCatNoCls(
            in_emb_dim, depth,
            heads=heads,
            dim_head=in_emb_dim,
            mlp_dim=in_emb_dim,
            attn_type='galerkin',    # Normalize K and V
            use_relu=False,          # Use GELU activation
            use_ln=use_ln,
            scale=[res, res//4] + [1]*(depth-2),  # Multi-scale encoding
            relative_emb_dim=2,      # 2D coordinates
            min_freq=1/res,          # Minimum frequency
            dropout=0.03,
            attention_init='orthogonal'
        )

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False)
        )

    def forward(self, x, input_pos):
        # x: [batch, num_points, 1] - coefficient values a(x,y)
        # input_pos: [batch, num_points, 2] - spatial coordinates (x,y)

        # Concatenate input with coordinates
        x = torch.cat((x, input_pos), dim=-1)  # [b, n, 3]
        x = self.to_embedding(x)
        x = self.dropout(x)

        # Apply Galerkin attention with 2D RoPE
        x = self.s_transformer.forward(x, input_pos)
        x = self.to_out(x)

        return x
```

```python
# Point-wise Decoder
class SimplePointwiseDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=1, hidden_dim=128):
        super().__init__()

        # Simple MLP decoder shared across all points
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, out_channels)
        )

    def forward(self, z):
        # z: [batch, num_points, latent_dim]
        return self.decoder(z)  # [batch, num_points, out_channels]
```

```python
# Training
def train_darcy():
    encoder = SpatialEncoder2D(
        input_channels=3,      # a(x,y), x, y
        in_emb_dim=96,
        out_seq_emb_dim=256,
        heads=4,
        depth=4,
        res=141,               # Training resolution
        use_ln=True
    )

    decoder = SimplePointwiseDecoder(
        latent_dim=256,
        out_channels=1,        # Pressure u(x,y)
        hidden_dim=128
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10000,
        gamma=0.5
    )

    for iter in range(32000):
        # a: coefficient field [b, n, 1]
        # u: solution field [b, n, 1]
        # pos: coordinates [b, n, 2]

        z = encoder(a, pos)
        pred = decoder(z)

        # Relative L2 loss
        loss = torch.mean(
            torch.norm(pred - u, p=2, dim=-1) /
            (torch.norm(u, p=2, dim=-1) + 1e-8)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
```

## Critical Parameters

- **in_emb_dim**: 96 (transformer embedding dimension)
- **out_seq_emb_dim**: 256 (latent representation dimension)
- **heads**: 4 (multi-head attention)
- **depth**: 4 (transformer layers)
- **attn_type**: 'galerkin' (instance norm on K and V)
- **scales**: [141, 35, 1, 1] (multi-scale RoPE for resolution 141)
- **min_freq**: 1/141 (minimum frequency for RoPE)
- **use_ln**: True (layer normalization between blocks)
- **dropout**: 0.03 (attention dropout), 0.05 (embedding dropout)
- **learning_rate**: 1e-3 with StepLR decay (gamma=0.5 every 10k iterations)
- **training_iterations**: 32,000
- **batch_size**: 8
- **resolution**: Tested on 141×141 and 211×211
