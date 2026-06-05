# GNOT for 2D Transonic Airfoil Flow

**Keywords**: [PDE, hyperbolic, nonlinear, forward-problem, euler, 2D, irregular, multi-scale, shock, turbulent, GNOT, Transformer, linear-attention, cross-attention, self-attention, MoE, geometric-gating, MLP, l2-regularization, adamw, mse, pytorch]

**Problem:** GNOT applied to transonic flow over NACA airfoils governed by the Euler equations. This is a classic aerodynamics problem where compressible inviscid flow passes over an airfoil, creating complex flow patterns including shocks, expansion fans, and vortex wakes. The input function is the airfoil boundary shape (irregular mesh describing the airfoil geometry). The goal is to predict the flow field (velocity, pressure, density, Mach number) around the airfoil. This problem is highly challenging due to: (1) irregular mesh conforming to arbitrary airfoil shapes, (2) multi-scale features (thin boundary layers near airfoil, shocks, smooth far-field), (3) nonlinear hyperbolic PDEs with discontinuities (shocks), and (4) high sensitivity to geometry changes.

**Issues addressed:**
- **Complex irregular geometry**: Each airfoil has unique shape requiring conforming mesh - no regular grid possible
- **Multi-scale flow physics**: Solution varies from fine-scale near airfoil to smooth far-field, with sharp shock discontinuities
- **Shock discontinuities**: Transonic flow creates shock waves (discontinuities in solution), challenging for smooth neural networks
- **Geometry sensitivity**: Small changes in airfoil shape cause significant flow changes - model must learn this sensitivity
- **Vortex structures**: Wake regions behind airfoil have complex vortical patterns
- **Nonlinear hyperbolic PDEs**: Euler equations are nonlinear and hyperbolic, more challenging than elliptic/parabolic PDEs

## Key Method

GNOT with all three innovations critical for aerodynamic flows:

1. **Mesh-Independent Architecture**:
   - Attention mechanism naturally handles point clouds, no need for regular grid
   - Each airfoil has different mesh tailored to its shape
   - Permutation-equivariant processing preserves mesh-independence

2. **Geometric Gating for Multi-Scale Flow**:
   - Near airfoil: boundary layer, stagnation point, leading/trailing edge
   - Mid-field: shock waves, expansion fans
   - Far-field: smooth potential flow
   - Multiple experts (3-4) specialize for these different regions
   - Geometric coordinates (distance from airfoil) guide expert selection

3. **High Model Capacity**:
   - Transformer architecture with large capacity can learn complex shock patterns
   - Linear attention enables scaling to fine meshes needed for shock resolution

**Why GNOT works for shocks:**
- Despite shocks being discontinuities, the attention mechanism can learn sharp gradients
- Geometric gating helps specialize experts near shock regions
- High model capacity (wide and deep transformers) improves shock prediction

## Implementation

```python
# GNOT configured for aerodynamic flows

class CGPTNO(nn.Module):
    def __init__(self,
                 trunk_size=2,          # 2D coordinates (x, y)
                 branch_sizes=[2],      # Airfoil boundary points (x, y)
                 output_size=4,         # Flow variables (ρ, u, v, p) or (u, v, p, M)
                 n_hidden=128,          # Sufficient for 2D aerodynamics
                 n_layers=4,            # Deeper for shock resolution
                 n_head=4-8,            # Multi-head for directional flow patterns
                 n_experts=3,           # Near-field, mid-field, far-field
                 ...):
        # Same architecture as other GNOT applications
        # Tuned for: irregular airfoil mesh, multi-scale flow, shocks
```

```python
# Input encoding for airfoil shape
def encode_airfoil_boundary(boundary_points):
    """
    Encode airfoil geometry as input function

    Args:
        boundary_points: [B, N, 2] - (x, y) coordinates defining airfoil shape
                         NACA airfoils have specific parameterizations

    Returns:
        Boundary embedding for cross-attention
    """
    # Branch MLP encodes boundary geometry
    # Network learns to extract features like:
    # - Thickness distribution
    # - Camber
    # - Leading/trailing edge curvature
    # - Overall chord length and orientation

    boundary_embed = branch_mlp(boundary_points)  # [B, N, D]

    return [boundary_embed]  # List for compatibility with multi-input framework
```

```python
# Geometric gating for multi-scale aerodynamic flow
def aerodynamic_gating(spatial_coords, boundary_coords):
    """
    Gating based on aerodynamic flow regions

    Args:
        spatial_coords: [B, N, 2] - query point locations
        boundary_coords: [B, M, 2] - airfoil boundary points

    Returns:
        gate_weights: [B, N, K] - expert weights
    """
    # Compute distance to nearest boundary point
    # Points near airfoil → expert 1 (boundary layer)
    # Points at intermediate distance → expert 2 (shock region)
    # Points far from airfoil → expert 3 (far-field)

    # Simple MLP gating based on coordinates
    gate_logits = gating_mlp(spatial_coords)  # [B, N, K]
    gate_weights = F.softmax(gate_logits, dim=-1)

    # The network learns that:
    # - Near airfoil surface: high gradients, complex patterns
    # - Shock regions: sharp transitions
    # - Far-field: smooth, nearly uniform flow

    return gate_weights
```

```python
# Training on aerodynamic datasets
def train_airfoil(model, data, optimizer):
    """
    Train on airfoil flow prediction task
    """
    # Input: airfoil boundary geometry
    # Output: flow field (velocity, pressure, etc.)

    graph, boundary_points = data
    inputs = encode_airfoil_boundary(boundary_points)

    out = model(graph, None, inputs)

    # Loss on flow variables
    # Could be (u, v, p) or derived quantities like Mach number
    y_pred = out
    y_true = graph.ndata['y']

    loss = rel_l2_loss(y_pred, y_true)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()

    return loss
```

## Critical Parameters

**Architecture (tuned for aerodynamics):**
- `n_hidden`: 96-128 (moderate size for 2D flow)
- `n_layers`: 3-4 (deep enough for shock resolution and boundary layers)
- `n_head`: 4-8 (captures multi-directional flow patterns and anisotropic shocks)
- `n_experts`: 3 (near-field/mid-field/far-field or attached/separated/wake regions)
- `mlp_layers`: 3
- `branch_sizes`: [2] (2D airfoil boundary coordinates)
- `output_size`: 3-4 (depends on which flow variables: velocity, pressure, density, Mach)

**Training:**
- `lr`: 1e-3 with OneCycleLR (important for convergence with shocks)
- `optimizer`: AdamW with weight_decay=5e-6
- `batch_size`: 4-8 (aerodynamic meshes are moderate size)
- `epochs`: 500
- `grad_clip`: 1000.0 (critical for shock stability)
- `loss_name`: 'rel2' (relative L2 error)
- `lr_schedule`: OneCycleLR works well for aerodynamic problems

**Aerodynamics-specific:**
- **Shock handling**: Despite discontinuities, neural networks learn smooth approximations with sharp gradients
- **Mesh refinement**: Finer mesh near airfoil and shocks improves accuracy
- **Normalization**: Flow variables span large ranges - careful normalization essential
- **Data augmentation**: Training on diverse airfoil shapes improves generalization

**Performance (from paper Table 1):**
- GNOT error: 7.57e-3
- Baseline errors: MIONet (13.2%), GK-Transformer (1.61%), Geo-FNO (1.38%), OFormer (1.83%)
- **45% improvement** over best baseline (Geo-FNO)
- Dramatic improvement shows GNOT's effectiveness on complex multi-scale hyperbolic problems

**Key advantages over baselines:**
- **vs FNO/Geo-FNO**: Direct point cloud processing without grid mapping preserves geometric accuracy
- **vs MIONet**: Higher model capacity captures complex nonlinear aerodynamics
- **vs OFormer**: Heterogeneous normalized attention and geometric gating better handle multi-scale features

**Key insight:** Aerodynamic flows with shocks and complex geometries benefit from all three GNOT innovations: (1) mesh-independent attention handles arbitrary airfoil shapes, (2) geometric gating addresses multi-scale flow from boundary layer to far-field, and (3) high transformer capacity learns complex nonlinear patterns including shock waves. The 45% error reduction demonstrates GNOT's effectiveness on practical engineering problems.
