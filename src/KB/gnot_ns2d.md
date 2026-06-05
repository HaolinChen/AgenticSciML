# GNOT for 2D Navier-Stokes (Steady-State)

**Keywords**: [PDE, hyperbolic, elliptic, nonlinear, forward-problem, steady-navier-stokes, 2D, irregular, multi-scale, turbulent, vortex-structure, GNOT, Transformer, linear-attention, cross-attention, self-attention, MoE, geometric-gating, MLP, l2-regularization, adamw, mse, pytorch]

**Problem:** GNOT applied to 2D steady-state Navier-Stokes equations in a rectangular domain with multiple circular obstacles (cavities). The governing equations are (u·∇)u = (1/Re)∇²u - ∇p (momentum) and ∇·u = 0 (incompressibility), with boundary conditions: u=0 on obstacles and walls, u=0 on outlet (pressure), u=y(8-y)/16 on inlet. The input is the geometry mesh (rectangle with varying circular obstacle positions). The goal is to predict the velocity field (u, v) and pressure field (p). This problem demonstrates GNOT on: (1) complex irregular domains (rectangle minus circles), (2) multi-scale flow features (wakes, recirculation zones, boundary layers), (3) coupled nonlinear PDEs (velocity-pressure coupling), and (4) geometric variations (different obstacle configurations).

**Issues addressed:**
- **Complex geometric variations**: Each sample has circles at different positions, creating different flow patterns
- **Multi-scale turbulent-like features**: Flow includes boundary layers, wakes behind obstacles, recirculation zones, vortex shedding
- **Coupled multi-field problem**: Velocity and pressure are coupled through incompressibility and momentum equations
- **Irregular domain**: Rectangle with circular exclusions cannot be represented on regular Cartesian grid
- **Vortex structures**: Flow separation behind obstacles creates coherent vortical structures that require capturing spatial correlations

## Key Method

GNOT architecture tailored for complex fluid dynamics:

1. **Mesh-Independent Processing**:
   - Handles varying obstacle positions naturally through point cloud representation
   - Each configuration has custom mesh conforming to obstacles
   - No need for interpolation or regular grid mapping

2. **Geometric Gating for Flow Regions**:
   - Different flow physics in different regions:
     * Near obstacles: boundary layers, separation points
     * Wake regions: vortices, recirculation
     * Free stream: relatively uniform flow
   - Multiple experts specialize for these distinct flow patterns
   - Spatial coordinates guide soft domain decomposition

3. **Multi-Field Coupling**:
   - Jointly predicts (u, v, p) to learn velocity-pressure coupling
   - Attention mechanism captures long-range dependencies (e.g., pressure gradient affects velocity far downstream)

**Why GNOT excels for Navier-Stokes:**
- Attention captures long-range spatial dependencies critical for incompressible flow
- Linear complexity O(N) enables fine meshes for boundary layer resolution
- Geometric gating handles multi-scale features from viscous layers to potential flow

## Implementation

```python
# GNOT for 2D Navier-Stokes with geometric obstacles

class CGPTNO(nn.Module):
    def __init__(self,
                 trunk_size=2,          # 2D coordinates
                 branch_sizes=[2],      # Obstacle boundary points
                 output_size=3,         # u, v, p
                 n_hidden=256,          # Large capacity for complex flows
                 n_layers=4,            # Deep network for multi-scale features
                 n_head=8,              # Multi-directional flow patterns
                 n_experts=3-4,         # Different flow regions
                 ...):
        # Same GNOT architecture
        # Key: large capacity (n_hidden=256, n_layers=4) for complex NS physics
```

```python
# Input encoding for NS with geometric obstacles
def encode_ns_geometry(obstacle_boundaries):
    """
    Encode obstacle geometry for Navier-Stokes

    Args:
        obstacle_boundaries: [B, N, 2] - points defining circular obstacles
                             (could be mesh nodes on obstacle surfaces)

    Returns:
        Geometry embedding for cross-attention
    """
    # Branch MLP encodes obstacle positions and shapes
    # Network learns how obstacle configuration affects flow:
    # - Obstacle size influences wake size
    # - Spacing between obstacles affects flow channeling
    # - Positions determine vortex interaction patterns

    geom_embed = branch_mlp(obstacle_boundaries)  # [B, N, D]

    return [geom_embed]
```

```python
# Multi-field prediction for coupled u, v, p
def predict_ns_fields(model, graph, geometry):
    """
    Predict velocity and pressure fields for Navier-Stokes

    Returns:
        u, v, p: velocity components and pressure
    """
    inputs = encode_ns_geometry(geometry)

    # Forward pass - single model predicts all fields jointly
    out = model(graph, None, inputs)  # [N, 3]

    u = out[:, 0]  # x-velocity
    v = out[:, 1]  # y-velocity
    p = out[:, 2]  # pressure

    # Joint prediction allows model to learn:
    # - Pressure gradient drives velocity
    # - Velocity divergence relates to pressure (incompressibility)
    # - Vorticity patterns in velocity affect pressure distribution

    return u, v, p
```

```python
# Training with multi-component loss
def train_ns(model, data, optimizer):
    """
    Train on Navier-Stokes with separate losses for u, v, p
    """
    graph, geometry = data

    # Predict all fields
    out = model(graph, None, [geometry])
    u_pred, v_pred, p_pred = out[:, 0], out[:, 1], out[:, 2]

    # Ground truth from FEM solver
    u_true = graph.ndata['u']
    v_true = graph.ndata['v']
    p_true = graph.ndata['p']

    # Separate relative L2 losses
    # (Velocity and pressure have different scales/units)
    loss_u = rel_l2_loss(u_pred, u_true)
    loss_v = rel_l2_loss(v_pred, v_true)
    loss_p = rel_l2_loss(p_pred, p_true)

    # Combined loss
    loss = loss_u + loss_v + loss_p

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()

    return loss_u, loss_v, loss_p
```

## Critical Parameters

**Architecture (tuned for NS with obstacles):**
- `n_hidden`: 256 (large capacity for complex vortical flows)
- `n_layers`: 4 (deep network for multi-scale boundary layers to far-field)
- `n_head`: 8 (captures multi-directional flow and pressure gradients)
- `n_experts`: 3-4 (near-obstacle, wake, free-stream regions)
- `mlp_layers`: 4 (deep encoding for geometry and flow)
- `output_size`: 3 (u, v, p)

**Training:**
- `lr`: 1e-3 with OneCycleLR
- `optimizer`: AdamW with weight_decay=5e-6
- `batch_size`: 4-16 (depends on mesh size)
- `epochs`: 500
- `grad_clip`: 1000.0 (important for flow stability)
- `loss_name`: 'rel2' for each component

**NS-specific considerations:**
- **Field normalization**: u, v, p have different scales - use per-field normalizers
- **Reynolds number**: Flow characteristics depend on Re (controls viscosity) - could be input parameter
- **Boundary conditions**: Dirichlet BC (u=0) on obstacles enforced through data/loss
- **Incompressibility**: Not explicitly enforced in loss, but learned implicitly through data

**Performance (from paper Table 1):**
- GNOT errors: u (6.73e-3), v (1.55e-2), p (7.41e-3)
- Baseline comparisons:
  * MIONet: u (2.74e-2), v (5.51e-2), p (2.74e-2)
  * GK-Transformer: u (1.52e-2), v (3.15e-2), p (1.59e-2)
  * Geo-FNO: u (1.41e-2), v (2.98e-2), p (1.62e-2)
  * OFormer: u (2.33e-2), v (4.83e-2), p (2.43e-2)
- **50-70% improvement** over baselines across all fields
- Particularly strong on pressure field prediction

**Ablation insights:**
- Cross-attention before self-attention works best (vs other orders)
- Multiple attention heads (8) better than single head for anisotropic flow
- Geometric gating (n_experts=3-4) improves accuracy for multi-scale wakes

**Key insight:** GNOT's heterogeneous normalized cross-attention efficiently encodes varying obstacle geometries, while geometric gating handles the multi-scale flow physics from viscous boundary layers to inviscid wakes. The dramatic error reduction (50-70%) over baselines demonstrates GNOT's effectiveness for complex coupled nonlinear PDEs with irregular domains. The linear attention complexity O(N) is critical for fine meshes needed to resolve boundary layers and vortical structures.
