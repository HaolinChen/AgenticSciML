# GNOT for 2D Elasticity

**Keywords**: [PDE, elliptic, linear, forward-problem, elasticity-steady, 2D, irregular, dirichlet, GNOT, Transformer, linear-attention, cross-attention, self-attention, MLP, l2-regularization, adamw, mse, pytorch]

**Problem:** GNOT (General Neural Operator Transformer) applied to 2D elasticity problems governed by elastokinetics equations. The physical system is a solid body (unit square with irregular cavity) under stress. The architecture addresses operator learning on highly irregular geometric domains where the mesh must conform to complex cavity shapes. The goal is to predict the displacement/stress field from the input mesh geometry, demonstrating GNOT's ability to handle problems where the domain geometry itself varies across samples.

**Issues addressed:**
- **Complex irregular geometries**: The elasticity problem features cavities with varying shapes, creating highly irregular meshes that cannot be handled by grid-based methods like FNO
- **Geometry-dependent solutions**: GNOT's attention mechanism naturally processes the mesh topology and geometry as input, learning how domain shape affects the solution
- **Multi-scale spatial features**: Stress concentrations near cavity boundaries create multi-scale solution features, handled by the geometric gating mechanism
- **Mesh-independence**: The transformer architecture is permutation-equivariant and handles variable mesh resolution across different samples

## Key Method

GNOT uses the same three-component architecture as described in gnot_darcy2d:

1. **Heterogeneous Normalized Cross-Attention**: Fuses information from input mesh geometry (encoded as boundary points) with query points using efficient O(N) linear attention

2. **Normalized Self-Attention**: Propagates information among query points to capture spatial correlations in the solution field

3. **Geometric Gating (optional)**: For this elasticity problem with localized stress concentrations, using multiple expert FFNs with geometric gating helps the model specialize different experts for regions near/far from cavity boundaries

**Input encoding protocol for elasticity:**
- Trunk network: Encodes query point coordinates where solution is needed
- Branch network: Encodes boundary shape points {(xᵢ, yᵢ)} defining the cavity geometry
- The irregular mesh topology is implicitly captured through the point cloud representation

## Implementation

The implementation is identical to gnot_darcy2d. Key architecture components:

```python
# Same CGPTNO architecture as gnot_darcy2d
# Input:
#   - g: DGL graph with node features = query point coordinates
#   - inputs: boundary shape points defining cavity geometry
# Output:
#   - Displacement or stress field at query points

class CGPTNO(nn.Module):
    # Architecture identical to gnot_darcy2d
    # Trunk MLP: encodes query coordinates
    # Branch MLP: encodes boundary geometry
    # N × [Cross-Attn → FFN → Self-Attn → FFN] blocks
    # Output MLP: decodes to solution
```

**Elasticity-specific data handling:**
```python
# The mesh geometry is encoded as boundary points
# Input function: list of (x, y) coordinates defining cavity boundary
# Model learns mapping: boundary_shape → displacement_field

# Training uses relative L2 loss on displacement/stress components
# Gradient clipping and AdamW optimization for stability
```

## Critical Parameters

**Architecture (tuned for elasticity with irregular geometry):**
- `n_hidden`: 96 (sufficient for 2D elasticity patterns)
- `n_layers`: 3-4 (captures multi-scale stress distributions)
- `n_head`: 4-8 (helps capture directional stress patterns)
- `n_experts`: 1-4 (multiple experts help near stress concentrations)
- `mlp_layers`: 3 (adequate encoding depth)

**Training:**
- `lr`: 1e-3 with OneCycleLR schedule
- `optimizer`: AdamW with weight_decay=5e-6
- `batch_size`: 4-8 (elasticity meshes can be large)
- `epochs`: 500
- `grad_clip`: 1000.0
- `loss_name`: 'rel2' (relative L2 error)

**Best practices for elasticity:**
- Normalize input coordinates to unit square for numerical stability
- Use relative error metrics since displacement magnitudes vary with loading
- Gradient clipping is important due to sharp stress gradients near boundaries
- Multiple attention heads help capture anisotropic stress patterns
