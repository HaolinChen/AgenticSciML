# Operator Transformer for Electrostatics Poisson Equation

**Keywords**: ["PDE", "elliptic", "linear", "forward-problem", "poisson", "2D", "irregular", "dirichlet", "Transformer", "finite_element", "strong-form", "mse", "adam", "pytorch"]

**Problem:** Learning solution operators for 2D electrostatics Poisson equations on irregular geometries, mapping source terms and boundary conditions to electric potential and field. The model must generalize to unseen geometries (shape extrapolation from training shapes like circles and squares to test shapes like U-shapes), handling variable mesh densities and irregular boundaries.

**Issues addressed:**
- Handling irregular/unstructured meshes where CNNs and spectral methods cannot be applied
- Shape extrapolation: generalizing to geometries not seen during training
- Variable mesh density and adaptive refinement without retraining
- Predicting both scalar potential and vector electric field simultaneously

## Key Method

Operator Transformer with **padding masks** for irregular geometries. Uses Galerkin attention with instance normalization adapted for variable-length sequences through masked operations.

**Key Innovation for Irregular Grids:**
- Padding masks to handle varying numbers of mesh points across samples
- Masked instance normalization to compute statistics only over valid (non-padded) points
- Node-type-agnostic processing (no explicit boundary condition encoding in architecture)

## Implementation

```python
# Irregular Spatial Encoder with Padding Masks
class IrregSpatialEncoder2D(nn.Module):
    """Encoder for steady-state problems on irregular geometries"""
    def __init__(self,
                 input_channels=11,    # Source terms + coordinates + node features
                 in_emb_dim=64,
                 out_chanels=64,
                 heads=1,
                 depth=2,              # Fewer layers for smaller irregular problems
                 res=250,              # Approximate characteristic mesh size
                 use_ln=True,
                ):
        super().__init__()

        # Input embedding with ReLU (more stable for irregular grids)
        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(0.05)

        # Galerkin attention with padding mask support
        self.s_transformer = TransformerWithPad(
            in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
            attn_type='galerkin',
            use_relu=True,        # ReLU instead of GELU for stability
            use_ln=use_ln,
            scale=[res, res//4] + [1]*(depth-2),
            relative_emb_dim=2,
            min_freq=1/res,
            dropout=0.,
            attention_init='orthogonal'
        )

        # Output head
        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, out_chanels, bias=False)
        )

    def forward(self, x, input_pos, pad_mask):
        # x: [batch, num_points, channels] - padded to max_points
        # input_pos: [batch, num_points, 2] - coordinates (padded)
        # pad_mask: [batch, num_points, 1] - True for valid, False for padded

        x = torch.cat((x, input_pos), dim=-1)  # Concatenate coordinates
        x = self.to_embedding(x)

        # Zero out padded positions
        x = x.masked_fill(~pad_mask, 0.)

        x_skip = x
        x = self.dropout(x)

        # Attention with padding mask
        x = self.s_transformer.forward(x, input_pos, pad_mask)

        x = x.masked_fill(~pad_mask, 0.)  # Zero out padded positions again

        x = self.to_out(x)
        x = x.masked_fill(~pad_mask, 0.)  # Final masking

        return x
```

```python
# Masked Instance Normalization
def masked_instance_norm(x, mask, eps=1e-5):
    """
    Instance normalization that only considers non-padded points
    x: [batch*heads, num_points, features]
    mask: [batch*heads, num_points, 1] - True for valid points
    """
    mask = mask.float()

    # Compute mean only over valid points
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))  # [N, C]
    mean = mean.detach()

    # Compute variance only over valid points
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))
    var = var.detach()

    # Normalize
    mean_reshaped = mean.unsqueeze(1).expand_as(x)
    var_reshaped = var.unsqueeze(1).expand_as(x)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)

    return ins_norm
```

```python
# Attention with Padding Mask
class TransformerWithPad(nn.Module):
    """Transformer that handles variable-length sequences via padding masks"""

    def forward(self, x, pos_embedding, pad_mask):
        # x: [b, n, c]
        # pos_embedding: [b, n, 2]
        # pad_mask: [b, n, 1] - True for valid, False for padded

        for attn, ffn in self.layers:
            # Apply attention with masking
            x = attn(x, pos_embedding, padding_mask=pad_mask) + x
            x = ffn(x) + x

        return x

class LinearAttention(nn.Module):
    def forward(self, x, pos=None, padding_mask=None):
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if padding_mask is not None:
            # Compute grid size (number of valid points)
            grid_size = torch.sum(padding_mask, dim=[-1, -2]).view(-1, 1, 1, 1)
            padding_mask = repeat(padding_mask, 'b n d -> (b h) n d', h=self.heads)

            # Masked instance normalization
            k = rearrange(k, 'b h n d -> (b h) n d')
            v = rearrange(v, 'b h n d -> (b h) n d')

            k = masked_instance_norm(k, padding_mask)
            v = masked_instance_norm(v, padding_mask)

            k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
            v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)

            padding_mask = rearrange(padding_mask, '(b h) n d -> b h n d', h=self.heads)

        # Apply RoPE
        if self.relative_emb:
            freqs_x = self.emb_module.forward(pos[..., 0], x.device)
            freqs_y = self.emb_module.forward(pos[..., 1], x.device)
            freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=q.shape[1])
            freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=q.shape[1])

            q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
            k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)

        # Linear attention with masking
        if padding_mask is not None:
            q = q.masked_fill(~padding_mask, 0)
            k = k.masked_fill(~padding_mask, 0)
            v = v.masked_fill(~padding_mask, 0)
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1. / grid_size)  # Normalize by actual grid size
        else:
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1./q.shape[2])

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```

```python
# Training with Mesh Augmentation
def train_electrostatics():
    encoder = IrregSpatialEncoder2D(
        input_channels=11,     # Source, boundary conditions, coordinates, etc.
        in_emb_dim=64,
        out_chanels=64,
        heads=1,
        depth=2,
        res=250,               # Approximate mesh characteristic size
        use_ln=True
    )

    decoder = IrregularDecoder(
        latent_dim=64,
        out_channels=3         # Potential + Electric field (Ex, Ey)
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )

    for iter in range(32000):
        # x: source terms and features
        # y: [potential, field_x, field_y]
        # pos: node coordinates
        # pad_mask: validity mask

        z = encoder(x, pos, pad_mask)
        pred = decoder(z, pad_mask)

        # MSE loss only on valid (non-padded) points
        valid_pred = pred[pad_mask.expand_as(pred)]
        valid_y = y[pad_mask.expand_as(y)]
        loss = F.mse_loss(valid_pred, valid_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Critical Parameters

- **in_emb_dim**: 64 (embedding dimension, smaller for irregular grids)
- **heads**: 1 (single attention head)
- **depth**: 2 (fewer layers, less overfitting on irregular grids)
- **attn_type**: 'galerkin' with masked instance normalization
- **scales**: [250, 62] (two-scale RoPE for depth=2)
- **use_relu**: True (ReLU activation for stability on irregular grids)
- **use_ln**: Layer normalization variant for padded sequences
- **attention_init**: 'orthogonal'
- **learning_rate**: 1e-3 with Adam
- **training_iterations**: 32,000
- **batch_size**: 16
- **mesh_augmentation**: Random transformations on training meshes (varying node density, hole sizes)
- **loss**: MSE on valid points only (ignoring padding)
