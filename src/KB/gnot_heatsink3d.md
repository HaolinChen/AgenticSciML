# GNOT for 3D Multi-Physics Heatsink

**Keywords**: [PDE, parabolic, hyperbolic, nonlinear, forward-problem, heat, navier-stokes, 3D, irregular, mixed-bc, multi-scale, strongly-coupled, heterogeneous, GNOT, Transformer, linear-attention, cross-attention, self-attention, MoE, geometric-gating, MLP, l2-regularization, adamw, mse, pytorch]

**Problem:** GNOT applied to a 3D multi-physics heatsink simulation coupling laminar flow and heat conduction. This is a highly complex problem where heat convection from airflow through pipes is coupled with heat conduction in the solid heatsink body. The physics involves coupled Navier-Stokes equations for fluid flow and heat equations for temperature distribution, with bidirectional coupling through convective heat transfer. The input functions include geometric parameters and velocity distribution at the inlet. The goal is to predict both the 3D velocity field (u, v, w) and temperature field (T) throughout the domain. This problem represents the most challenging application of GNOT: 3D geometry, multi-physics coupling, multiple scales, and high computational complexity.

**Issues addressed:**
- **3D irregular geometry**: Heatsink has complex 3D structure with pipes and fins, creating highly irregular 3D mesh
- **Multi-physics coupling**: Fluid velocity affects heat convection; temperature affects fluid properties (bidirectional coupling)
- **Multiple output fields**: Must predict 4 fields (u, v, w, T) simultaneously with different physical units and scales
- **3D multi-scale features**: Flow boundary layers near walls, thermal boundary layers, and wake regions create multi-scale solution structure
- **High dimensionality**: 3D meshes have many more degrees of freedom, requiring efficient O(N) attention complexity
- **Challenging convergence**: Multi-physics problems are harder to learn, showing higher errors (14-25%) compared to single-physics problems

## Key Method

GNOT with all three components crucial for 3D multi-physics:

1. **Heterogeneous Normalized Cross-Attention**: Handles 3D geometric inputs and boundary conditions efficiently with O(N) complexity instead of O(N²), critical for large 3D meshes

2. **Geometric Gating with Multiple Experts**:
   - Uses 4 expert FFNs to handle different physical regions (inlet, outlet, near-wall, far-field)
   - 3D spatial coordinates guide expert selection
   - Helps model specialize for flow vs thermal dominated regions

3. **Multi-field Output**: Single model predicts all 4 fields (u, v, w, T) jointly, learning cross-field correlations

**3D challenges and GNOT solutions:**
- **Mesh size**: 3D meshes can have 100K+ points → O(N) complexity essential
- **Memory**: Linear attention uses memory O(Nd²) instead of O(N²d) for standard attention
- **Multi-scale 3D**: Geometric gating provides soft 3D domain decomposition

## Implementation

```python
# Same CGPTNO architecture, but configured for 3D multi-physics

class CGPTNO(nn.Module):
    def __init__(self,
                 trunk_size=3,          # 3D coordinates (x, y, z)
                 branch_sizes=[...],    # Multiple inputs (geometry params, inlet BC)
                 output_size=4,         # 4 fields: u, v, w, T
                 n_hidden=192,          # Larger capacity for 3D
                 n_layers=4,            # Deeper for multi-physics
                 n_head=8,              # More heads for 3D patterns
                 n_experts=4,           # Multiple experts for different regions
                 ...):
        # Same architecture as 2D cases
        # Key difference: larger capacity (n_hidden, n_layers) for 3D complexity
```

```python
# 3D geometric gating - spatial coordinates guide expert selection
def geometric_gating_3d(spatial_coords, n_experts=4):
    """
    Gating network for 3D multi-physics problems

    Args:
        spatial_coords: [B, N, 3] - (x, y, z) coordinates
        n_experts: number of expert networks

    Returns:
        gate_weights: [B, N, K] - soft weights for K experts
    """
    # MLP maps 3D coords to expert logits
    gate_logits = gating_mlp(spatial_coords)  # [B, N, K]

    # Softmax to get normalized weights
    gate_weights = F.softmax(gate_logits, dim=-1)

    # Different experts specialize based on spatial location:
    # - Expert 1: near inlet (high velocity)
    # - Expert 2: near walls (boundary layers)
    # - Expert 3: wake regions (complex flow)
    # - Expert 4: far field (simpler flow)

    return gate_weights
```

```python
# Training for multi-field output
def train_multifield(model, data, optimizer):
    """
    Training for problems with multiple output fields
    """
    # Forward pass predicts all fields jointly
    out = model(graph, params, inputs)  # [N, 4] for (u, v, w, T)

    # Separate losses for each field
    u_pred, v_pred, w_pred, T_pred = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    u_true, v_true, w_true, T_true = ...

    # Relative L2 loss for each component
    loss_u = rel_l2_loss(u_pred, u_true)
    loss_v = rel_l2_loss(v_pred, v_true)
    loss_w = rel_l2_loss(w_pred, w_true)
    loss_T = rel_l2_loss(T_pred, T_true)

    # Total loss (can weight differently if needed)
    loss = loss_u + loss_v + loss_w + loss_T

    # Gradient clipping important for multi-physics stability
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip=1000.0)
    optimizer.step()

    return loss
```

## Critical Parameters

**Architecture (tuned for 3D multi-physics):**
- `n_hidden`: 192 (large capacity needed, but limited by 3D memory constraints)
- `n_layers`: 4 (deeper helps with multi-physics coupling)
- `n_head`: 8 (captures 3D directional patterns and multi-field correlations)
- `n_experts`: 4 (helps with different flow regimes and thermal zones)
- `mlp_layers`: 4 (deep encoding for 3D geometry)
- `trunk_size`: 3 (3D coordinates)
- `output_size`: 4 (u, v, w, T)

**Training (critical for stability):**
- `lr`: 1e-3 with OneCycleLR (careful scheduling important)
- `optimizer`: AdamW with weight_decay=5e-6
- `batch_size`: 4-8 (limited by 3D mesh size and memory)
- `epochs`: 500+
- `grad_clip`: 1000.0 (essential for multi-physics convergence)
- `loss_name`: 'rel2' for each field

**3D-specific considerations:**
- **Memory management**: 3D meshes with 100K+ points require careful batching
- **Gradient clipping**: Multi-physics coupling can cause gradient instabilities
- **Per-field normalization**: Different fields (velocity vs temperature) have different scales - normalize separately
- **Expert allocation**: 4 experts balance capacity vs computational cost for 3D

**Performance characteristics:**
- Velocity fields (u, v, w): 14-19% error (harder due to complex flow patterns)
- Temperature field (T): 25% error (challenging due to convection-diffusion coupling)
- These higher errors vs 2D problems reflect the increased difficulty of 3D multi-physics
- Still provides 100-1000× speedup vs FEM solvers

**Limitations noted in paper:**
- 3D multi-physics represents failure cases with higher errors
- Suggests incorporating more physics priors could help
- Performance gap vs 2D single-physics problems indicates room for architecture improvements

**Key insight:** 3D multi-physics problems push the limits of current GNOT architecture. The linear attention enables scaling to large 3D meshes, but prediction accuracy remains challenging. Future work could explore physics-informed losses or specialized architectures for multi-physics coupling.
