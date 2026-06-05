# Operator Transformer for Burgers' Equation

**Keywords**: ["PDE", "hyperbolic", "nonlinear", "forward-problem", "burgers", "1D", "regular", "Transformer", "spectral-method", "strong-form", "relative-l2", "adam", "pytorch"]

**Problem:** Learning solution operators for 1D viscous Burgers' equation from initial conditions to future time steps using attention-based neural networks. The method learns to map u(x,0) to u(x,T) where the equation models nonlinear wave propagation with viscous dissipation, applicable to shock wave and turbulence modeling.

**Issues addressed:**
- Resolution-invariance: model can query solutions at arbitrary spatial resolutions without retraining
- Energy-preserving predictions for scale-sensitive problems through proper initialization and normalization
- Spectral bias mitigation using Random Fourier Features to capture high-frequency solution components
- Efficient learning without softmax attention, reducing computational complexity from O(n²) to O(n)

## Key Method

Operator Transformer (OFormer) uses linear attention with Rotary Position Embeddings (RoPE) for 1D spatial coordinates. For Burgers' equation, the model employs Fourier-type attention with instance normalization on Query and Key matrices,  enabling scale-sensitive predictions while maintaining discretization flexibility.

**Key Components:**
1. **1D Encoder**: MLPs with linear Fourier-type attention layers and 1D RoPE
2. **Fourier Attention**: Instance normalization on Q and K (not V), preserving energy scales
3. **Orthogonal Initialization**: Special weight initialization with diagonal bias for scale-sensitive problems
4. **Multi-scale RoPE**: Decreasing spatial scales across layers [8, 4, 4, 1, ...]

## Implementation

```python
# 1D Encoder for Burgers' Equation
class Encoder1D(nn.Module):
    def __init__(self,
                 input_channels=2,      # u(x) and coordinate x
                 in_emb_dim=96,         # Embedding dimension
                 out_seq_emb_dim=96,    # Output dimension
                 depth=4,               # Number of transformer layers
                 res=2048,              # Resolution for RoPE scaling
                ):
        super().__init__()

        # Input embedding: concatenate solution value with scaled coordinate
        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        # Fourier-type linear attention with 1D RoPE
        self.transformer = TransformerCatNoCls(
            in_emb_dim, depth,
            heads=1,              # Single attention head for 1D
            dim_head=in_emb_dim,
            mlp_dim=in_emb_dim,
            attn_type='fourier',  # Fourier type: normalize Q and K
            scale=[8., 4., 4., 1.] + [1.]*(depth-4),  # Multi-scale encoding
            relative_emb_dim=1,   # 1D coordinates
            min_freq=1/res,       # Minimum frequency based on resolution
            use_ln=False,         # No layer norm for scale preservation
            dropout=0.05,
            attention_init='orthogonal'  # Special initialization
        )

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False)
        )

    def forward(self, x, input_pos):
        # x: [batch, num_points, 1] - solution values
        # input_pos: [batch, num_points, 1] - spatial coordinates

        # Concatenate solution with scaled coordinates
        x = torch.cat((x, input_pos/16.), dim=-1)
        x = self.to_embedding(x)

        # Apply Fourier attention with 1D RoPE
        x = self.transformer.forward(x, input_pos)
        x = self.project_to_latent(x)
        return x
```

```python
# Special Initialization for Scale-Sensitive Problems
def _init_params(self):
    """Orthogonal initialization with diagonal bias for Fourier attention"""
    for param in self.to_qkv.parameters():
        if param.ndim > 1:
            for h in range(self.heads):
                # Initialize V (value) matrix with orthogonal + diagonal
                v_slice = param[(self.heads * 2 + h) * self.dim_head:
                               (self.heads * 2 + h + 1) * self.dim_head, :]

                # Orthogonal initialization with gain 1/dim_head
                orthogonal_(v_slice, gain=1./self.dim_head)

                # Add diagonal component for identity-like initialization
                v_slice.data += (1./self.dim_head) * torch.diag(
                    torch.ones(param.size(-1), dtype=torch.float32)
                )
```

```python
# Training with Relative L2 Loss for Scale Preservation
def train_burgers():
    encoder = Encoder1D(
        input_channels=2,
        in_emb_dim=96,
        out_seq_emb_dim=96,
        depth=4,
        res=2048  # High resolution
    )

    decoder = PointwiseDecoder1D(
        latent_dim=96,
        out_channels=1,
        hidden_dim=48
    )

    # Adam optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )

    # Key: Use relative L2 norm to preserve energy scales
    def relative_l2_loss(pred, target):
        diff_norms = torch.norm(pred - target, p=2, dim=-1)
        target_norms = torch.norm(target, p=2, dim=-1)
        return torch.mean(diff_norms / (target_norms + 1e-8))

    for iter in range(20000):
        # x: initial condition u(x, 0)
        # y: solution at t=1, u(x, 1)

        z = encoder(x, input_pos)
        pred = decoder(z)

        loss = relative_l2_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Critical Parameters

- **in_emb_dim**: 96 (embedding dimension)
- **depth**: 4 (transformer layers)
- **heads**: 1 (single attention head for 1D)
- **attn_type**: 'fourier' (normalize Q and K, not V, for scale preservation)
- **scales**: [8.0, 4.0, 4.0, 1.0] (RoPE scales across layers)
- **min_freq**: 1/2048 (matched to resolution)
- **attention_init**: 'orthogonal' with diagonal bias (critical for convergence)
- **init_gain**: 1/dim_head (initialization scale)
- **use_ln**: False (no layer normalization to preserve energy scales)
- **learning_rate**: 1e-3 with Adam
- **loss**: Relative L2 norm (preserves solution magnitude information)
- **resolution**: Tested on 512, 2048, 8192 grid points
