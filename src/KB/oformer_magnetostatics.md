# Operator Transformer for Magnetostatics Poisson Equation

**Keywords**: ["PDE", "elliptic", "linear", "forward-problem", "poisson", "2D", "irregular", "dirichlet", "Transformer", "finite_element", "strong-form", "mse", "adam", "pytorch"]

**Problem:** Learning solution operators for 2D magnetostatics Poisson equations on irregular geometries, mapping current density and material properties to magnetic vector potential and field. Similar to electrostatics but for magnetic field calculations, requiring shape extrapolation to unseen geometries and handling variable mesh densities.

**Issues addressed:**
- Handling irregular/unstructured meshes with variable point densities
- Shape extrapolation: generalizing from training geometries (circles, squares, L-shapes) to test geometries (U-shapes)
- Predicting vector magnetic field from scalar potential using learned derivatives
- Mesh-agnostic processing through attention mechanisms with padding masks

## Key Method

Operator Transformer with padding mask support for irregular magnetostatics problems. Uses Galerkin attention with masked instance normalization to handle variable mesh sizes, identical architecture to electrostatics but applied to magnetic field equations.

**Architecture:** Same as electrostatics (IrregSpatialEncoder2D) with:
- Padding masks for variable-length sequences
- Masked instance normalization in Galerkin attention
- Multi-head output for potential and field components

## Implementation

```python
# Magnetostatics Encoder (same architecture as electrostatics)
class IrregSpatialEncoder2D(nn.Module):
    """Encoder for magnetostatics on irregular geometries"""
    def __init__(self,
                 input_channels=11,    # Current density + material properties + coordinates
                 in_emb_dim=96,        # Larger embedding for magnetostatics
                 out_chanels=96,
                 heads=1,
                 depth=2,
                 res=250,
                 use_ln=True,
                ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(0.05)

        self.s_transformer = TransformerWithPad(
            in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
            attn_type='galerkin',
            use_relu=True,
            use_ln=use_ln,
            scale=[res, res//4] + [1]*(depth-2),
            relative_emb_dim=2,
            min_freq=1/res,
            dropout=0.,
            attention_init='orthogonal'
        )

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, out_chanels, bias=False)
        )

    def forward(self, x, input_pos, pad_mask):
        x = torch.cat((x, input_pos), dim=-1)
        x = self.to_embedding(x)
        x = x.masked_fill(~pad_mask, 0.)

        x_skip = x
        x = self.dropout(x)
        x = self.s_transformer.forward(x, input_pos, pad_mask)
        x = x.masked_fill(~pad_mask, 0.)

        x = self.to_out(x)
        x = x.masked_fill(~pad_mask, 0.)

        return x
```

```python
# Multi-output Decoder for Magnetic Fields
class MagneticsDecoder(nn.Module):
    """Decoder for magnetic potential and field"""
    def __init__(self, latent_dim=96, out_channels=3):
        # out_channels=3: Az (potential), Bx, By (field components)
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.ReLU(),
            nn.Linear(latent_dim//2, latent_dim//2),
            nn.ReLU(),
            nn.Linear(latent_dim//2, out_channels)
        )

    def forward(self, z, pad_mask):
        pred = self.decoder(z)
        pred = pred.masked_fill(~pad_mask, 0.)
        return pred
```

```python
# Training
def train_magnetostatics():
    encoder = IrregSpatialEncoder2D(
        input_channels=11,
        in_emb_dim=96,         # Slightly larger than electrostatics
        out_chanels=96,
        heads=1,
        depth=2,
        res=250,
        use_ln=True
    )

    decoder = MagneticsDecoder(
        latent_dim=96,
        out_channels=3         # Az, Bx, By
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )

    for iter in range(32000):
        # x: current density Jz and permeability mu
        # y: [potential Az, field Bx, field By]
        # pos: node coordinates
        # pad_mask: validity mask [b, n, 1]

        z = encoder(x, pos, pad_mask)
        pred = decoder(z, pad_mask)

        # MSE loss only on valid points
        valid_pred = pred[pad_mask.expand_as(pred)]
        valid_y = y[pad_mask.expand_as(y)]

        loss = F.mse_loss(valid_pred, valid_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Critical Parameters

- **in_emb_dim**: 96 (slightly larger than electrostatics)
- **heads**: 1 (single attention head)
- **depth**: 2 (shallow network for irregular grids)
- **attn_type**: 'galerkin' with masked instance normalization
- **scales**: [250, 62] (two-scale RoPE)
- **use_relu**: True (ReLU for stability)
- **use_ln**: True with LayerNorm variant for padding
- **attention_init**: 'orthogonal'
- **learning_rate**: 1e-3 with Adam
- **training_iterations**: 32,000
- **batch_size**: 16
- **mesh_augmentation**: Random transformations (node density, geometric variations)
- **loss**: MSE on valid (non-padded) points
- **output_channels**: 3 (magnetic potential Az and field components Bx, By)
