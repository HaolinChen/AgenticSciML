# GNOT for 2D Electromagnetic Inductor

**Keywords**: [PDE, elliptic, linear, forward-problem, maxwell-steady, 2D, irregular, dirichlet, multi-scale, heterogeneous, GNOT, Transformer, linear-attention, cross-attention, self-attention, MoE, geometric-gating, MLP, l2-regularization, adamw, mse, pytorch]

**Problem:** GNOT applied to a 2D inductor system governed by steady-state Maxwell's equations. The problem models electromagnetic field distribution in an inductor with coils, where the governing equations are ∇×H = J (Ampère's law), B = ∇×A (magnetic field from vector potential), with constitutive relations J = σE + σv×B + Jₑ and B = μ₀μᵣH. The input functions include: (1) boundary shape geometry (highly irregular inductor coil geometry), (2) global parameter vectors (coil current Icoil, material permeability μᵣ). The goal is to predict the magnetic vector potential Az and magnetic field components (Bₓ, Bᵧ). This problem demonstrates GNOT's ability to handle multiple heterogeneous input types (geometric + parametric) with highly irregular geometries and multi-scale fields (strong fields near coils, weak fields far away).

**Issues addressed:**
- **Highly irregular geometry**: Inductor coils create complex geometric shapes with fine features that cannot be represented on regular grids
- **Multiple heterogeneous inputs**: Combines geometric input (boundary shape) with physical parameters (current, permeability) - different types requiring different encoders
- **Multi-scale fields**: Magnetic field strength varies by orders of magnitude from near coils (strong) to far field (weak)
- **Material heterogeneity**: Different regions have different magnetic permeability μᵣ
- **Electromagnetic coupling**: Vector potential A and magnetic field B are coupled through curl operation
- **Sharp gradients**: Electromagnetic fields have sharp gradients at material interfaces and near current-carrying coils

## Key Method

GNOT with emphasis on handling multiple heterogeneous inputs and multi-scale electromagnetic fields:

1. **Heterogeneous Normalized Cross-Attention** with multiple branch networks:
   - Branch network 1: Encodes boundary shape (irregular coil geometry as point cloud)
   - Branch network 2: Encodes global parameters (Icoil, μᵣ as vectors)
   - Different MLP encoders for geometric vs parametric inputs ensure proper representation

2. **Geometric Gating for Multi-Scale Fields**:
   - Electromagnetic fields have distinct scales: strong near coils, decaying in far field
   - Multiple experts (3-4) help model specialize for different field strength regions
   - Spatial coordinates guide expert selection based on distance from coils

3. **Multi-Output Prediction**: Jointly predicts Az, Bₓ, Bᵧ, learning correlations between potential and field

**Input Encoding Protocol:**
- **Boundary shape** {(xᵢ, yᵢ)}: Points defining coil geometry → MLP₁ → embedding₁
- **Parameter vector** (Icoil, μᵣ, ...): Physical parameters → MLP₂ → embedding₂
- **Query points**: Coordinates where fields are predicted → trunk MLP → query embedding
- Cross-attention fuses all inputs: query_emb ← CrossAttn(query_emb, [emb₁, emb₂])

## Implementation

```python
# GNOT with multiple heterogeneous input encoders

class CGPTNO(nn.Module):
    def __init__(self,
                 trunk_size=2,          # 2D coordinates
                 branch_sizes=[...],    # Different sizes for different inputs
                 output_size=3,         # Az, Bₓ, Bᵧ
                 n_hidden=256,          # Large capacity for multi-scale fields
                 n_experts=3,           # Multi-scale electromagnetic fields
                 ...):

        # Trunk network: query point encoder
        self.trunk_mlp = MLP(trunk_size, n_hidden, n_hidden, n_layers=mlp_layers)

        # Branch networks: separate encoder for each input type
        self.branch_mlps = nn.ModuleList([
            MLP(bsize, n_hidden, n_hidden, n_layers=mlp_layers)
            for bsize in branch_sizes
        ])
        # branch_mlps[0]: boundary shape encoder (inputs: x, y coords)
        # branch_mlps[1]: parameter encoder (inputs: Icoil, μᵣ, ...)

        # Attention blocks with geometric gating
        self.blocks = nn.Sequential(*[
            CrossAttentionBlock(config, n_experts=n_experts)
            for _ in range(n_layers)
        ])

        # Output decoder for multiple fields
        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers)
```

```python
# Encoding multiple heterogeneous inputs for inductor problem
def encode_inductor_inputs(boundary_shape, global_params):
    """
    Encode heterogeneous inputs for electromagnetic problem

    Args:
        boundary_shape: [B, N1, 2] - (x, y) points defining coil geometry
        global_params: [B, P] - (Icoil, μᵣ, ...) physical parameters

    Returns:
        List of conditional embeddings for cross-attention
    """
    # Branch MLP 1: encode geometric boundary
    # Input: spatial coordinates of boundary points
    boundary_embed = branch_mlp_1(boundary_shape)  # [B, N1, D]

    # Branch MLP 2: encode global parameters
    # Input: vector of physical parameters
    # Expand to have sequence dimension for attention
    param_embed = branch_mlp_2(global_params).unsqueeze(1)  # [B, 1, D]

    return [boundary_embed, param_embed]
```

```python
# Heterogeneous cross-attention aggregates different input types
class LinearCrossAttention(nn.Module):
    def forward(self, x, y_list):
        """
        x: query features [B, T, D]
        y_list: [y_boundary, y_params] - list of conditional embeddings

        Returns: updated query features with information from all inputs
        """
        q = self.query(x).softmax(dim=-1)
        out = q  # Initialize with identity

        # Aggregate from boundary shape
        k1 = self.keys[0](y_list[0]).softmax(dim=-1)
        v1 = self.values[0](y_list[0])
        D_inv1 = 1. / (q @ k1.sum(dim=-2, keepdim=True).transpose(-2, -1))
        out = out + (q @ (k1.transpose(-2, -1) @ v1)) * D_inv1

        # Aggregate from global parameters
        k2 = self.keys[1](y_list[1]).softmax(dim=-1)
        v2 = self.values[1](y_list[1])
        D_inv2 = 1. / (q @ k2.sum(dim=-2, keepdim=True).transpose(-2, -1))
        out = out + (q @ (k2.transpose(-2, -1) @ v2)) * D_inv2

        # Average and project
        out = self.proj(out)
        return out
```

```python
# Multi-output training for Az, Bₓ, Bᵧ
def train_electromagnetic(model, data, optimizer):
    """
    Train model to predict multiple electromagnetic fields
    """
    # Unpack inputs
    graph, boundary_shape, global_params = data
    inputs = encode_inductor_inputs(boundary_shape, global_params)

    # Forward pass
    out = model(graph, global_params, inputs)  # [N, 3]

    # Separate predictions for Az, Bₓ, Bᵧ
    Az_pred, Bx_pred, By_pred = out[:, 0], out[:, 1], out[:, 2]
    Az_true, Bx_true, By_true = ...

    # Relative L2 loss for each component
    # Note: Bₓ, Bᵧ are derived from Az via curl, so they're correlated
    loss_Az = rel_l2_loss(Az_pred, Az_true)
    loss_Bx = rel_l2_loss(Bx_pred, Bx_true)
    loss_By = rel_l2_loss(By_pred, By_true)

    loss = loss_Az + loss_Bx + loss_By

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()

    return loss
```

## Critical Parameters

**Architecture (tuned for electromagnetic fields):**
- `n_hidden`: 256 (large capacity for multi-scale fields and multiple inputs)
- `n_layers`: 4 (deep network for complex electromagnetic interactions)
- `n_head`: 8 (captures multi-directional field patterns)
- `n_experts`: 3-4 (different experts for near-coil vs far-field regions)
- `mlp_layers`: 4 (deep encoding for geometry and parameters)
- `branch_sizes`: [2, p] where 2 for (x,y) boundary points, p for number of global parameters
- `output_size`: 3 (Az, Bₓ, Bᵧ)

**Training:**
- `lr`: 1e-3 with OneCycleLR schedule
- `optimizer`: AdamW with weight_decay=5e-6
- `batch_size`: 4-8 (electromagnetic meshes can be large with fine features)
- `epochs`: 500
- `grad_clip`: 1000.0 (important for sharp field gradients)
- `loss_name`: 'rel2' for each field component

**Electromagnetic-specific:**
- **Field normalization**: Az, Bₓ, Bᵧ have different scales - use relative error metric
- **Gradient clipping**: Essential due to sharp gradients at material interfaces and near coils
- **Multiple experts**: Helps handle orders-of-magnitude variation in field strength
- **Separate encoders**: Geometric inputs (2D points) vs parametric inputs (scalars) need different MLP architectures

**Performance (from paper Table 1):**
- Az error: 1.21e-2 (vector potential)
- Bₓ error: 1.92e-2 (x-component of magnetic field)
- Bᵧ error: 3.62e-2 (y-component of magnetic field)
- **50% improvement** over MIONet baseline
- Baselines like Geo-FNO failed ("-") due to inability to handle multiple input types

**Key insight:** Multiple heterogeneous input types require separate branch encoders in GNOT's cross-attention framework. The heterogeneous normalized attention efficiently fuses geometric information (boundary shape as point cloud) with parametric information (physical constants as vectors), which is challenging for methods like FNO or GeoFNO that lack flexible input encoding protocols.
