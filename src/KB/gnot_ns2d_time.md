# GNOT for 2D Time-Dependent Navier-Stokes

**Keywords**: [PDE, hyperbolic, nonlinear, forward-problem, navier-stokes, 2D, periodic, time-delay, vortex-structure, turbulent, GNOT, Transformer, linear-attention, cross-attention, self-attention, MLP, l2-regularization, adamw, mse, pytorch]

**Problem:** GNOT applied to 2D time-dependent Navier-Stokes equation in vorticity form on a unit torus with periodic boundary conditions. The governing equation is ∂w/∂t + u·∇w = ν∇²w where w is vorticity and u is velocity derived from w via streamfunction. This is a classic turbulence problem where vortical structures evolve, merge, and decay over time. The input is the initial vorticity field (or first few time frames). The goal is to predict future vorticity fields (last few frames from first few frames). This problem demonstrates GNOT's capability for temporal operator learning: mapping from initial conditions to future states, which is fundamentally different from steady-state problems.

**Issues addressed:**
- **Temporal evolution**: Learning time-dependent operator G: u(t=0) → u(t=T), not just spatial mapping
- **Chaotic dynamics**: Navier-Stokes can exhibit chaotic behavior where small initial differences lead to diverging trajectories
- **Long-time integration**: Predicting multiple time steps ahead amplifies errors
- **Vortex interactions**: Vortices merge, split, and interact in complex nonlinear ways
- **Energy cascade**: Turbulent flows transfer energy across scales from large to small structures
- **Data efficiency and scalability**: GNOT shows strong scaling behavior - error reduces 3× (13.8% → 4.4%) when dataset size increases

## Key Method

GNOT for time-dependent problems with spatial-only attention:

**Key Insight**: The problem is formulated as spatial operator learning, not spatio-temporal. GNOT treats the temporal aspect through the data formulation:
- Input: Initial state u(x, t=0) or sequence of early states u(x, t₁), ..., u(x, tₖ)
- Output: Future state u(x, t=T) or sequence of late states u(x, tₙ₋ₖ), ..., u(x, tₙ)
- The model learns: G: u(·, t_initial) → u(·, t_final)

**Architecture choices:**
1. **Spatial attention only**: Attention operates over spatial points, not time
2. **Temporal encoding via data**: Initial frames encoded as "input function", future frames as "output"
3. **No recurrence**: Single feedforward pass, not autoregressive time-stepping
4. **Periodic boundary handling**: Attention naturally handles periodic domains through point cloud representation

**Why this works:**
- For smooth enough dynamics, the operator G: u₀ → uₜ can be learned directly
- Avoids error accumulation from autoregressive time-stepping
- More efficient than RNN/LSTM approaches for long-time prediction
- Works when the mapping u₀ → uₜ is sufficiently regular (not too chaotic)

## Implementation

```python
# GNOT for time-dependent NS - spatial operator formulation

class CGPTNO(nn.Module):
    def __init__(self,
                 trunk_size=2,          # 2D spatial coordinates (periodic torus)
                 branch_sizes=[3],      # Vorticity + spatial coords for initial state
                 output_size=1,         # Future vorticity field
                 n_hidden=128,          # Moderate size sufficient for 2D NS
                 n_layers=4,            # Captures temporal evolution patterns
                 n_head=8,              # Multi-directional vorticity gradients
                 ...):
        # Same spatial GNOT architecture
        # Temporal dynamics captured through input-output data pairing
```

```python
# Data formulation for temporal prediction
def prepare_temporal_data(vorticity_sequence):
    """
    Prepare time-dependent NS data for spatial operator learning

    Args:
        vorticity_sequence: [B, T, H, W] - vorticity field over time
                            T = total time steps (e.g., 20 frames)

    Returns:
        input_frames: Early time frames (e.g., frames 0-9)
        output_frames: Late time frames (e.g., frames 10-19)
    """
    # Split sequence into input (initial conditions) and output (future states)
    T_split = T // 2

    # Input: first half of sequence
    input_frames = vorticity_sequence[:, :T_split, :, :]  # [B, T/2, H, W]

    # Output: second half of sequence
    output_frames = vorticity_sequence[:, T_split:, :, :]  # [B, T/2, H, W]

    # Convert to point cloud format
    # input_features: (x, y, w(x,y,t_0), w(x,y,t_1), ..., w(x,y,t_{T/2-1}))
    # output_features: w(x,y,t_{T/2}), ..., w(x,y,t_{T-1})

    return input_frames, output_frames
```

```python
# Encoding temporal initial conditions
def encode_initial_conditions(spatial_coords, vorticity_history):
    """
    Encode initial vorticity fields as input function

    Args:
        spatial_coords: [B, N, 2] - (x, y) coordinates on periodic torus
        vorticity_history: [B, N, K] - vorticity at K initial time steps

    Returns:
        Initial condition embedding
    """
    # Concatenate spatial coords with temporal history
    input_features = torch.cat([spatial_coords, vorticity_history], dim=-1)  # [B, N, 2+K]

    # Branch MLP encodes spatio-temporal initial state
    init_embed = branch_mlp(input_features)  # [B, N, D]

    # Network learns to extract features like:
    # - Current vorticity distribution
    # - Temporal trends (increasing/decreasing)
    # - Spatial gradients that drive evolution
    # - Vortex strength and positions

    return [init_embed]
```

```python
# Training for temporal prediction
def train_temporal_ns(model, data, optimizer):
    """
    Train on time-dependent NS prediction task
    """
    # Unpack: initial frames → future frames
    graph_in, graph_out = data

    # Input: spatial coordinates + initial vorticity history
    coords = graph_in.ndata['coords']
    w_history = graph_in.ndata['vorticity_history']  # First K frames
    inputs = encode_initial_conditions(coords, w_history)

    # Predict: future vorticity field(s)
    w_pred = model(graph_out, None, inputs)  # Predict at future time(s)

    # Ground truth: actual future vorticity
    w_true = graph_out.ndata['vorticity']

    # Relative L2 loss over spatial domain
    loss = rel_l2_loss(w_pred, w_true)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()

    return loss
```

```python
# Handling periodic boundary conditions
# GNOT naturally handles periodic BC through point cloud representation
# No special treatment needed - attention operates on all points equally
# Periodicity is implicit in the data (points near x=0 and x=1 are close in physical space)
```

## Critical Parameters

**Architecture (tuned for time-dependent NS):**
- `n_hidden`: 96-128 (sufficient for 2D vorticity patterns)
- `n_layers`: 3-4 (captures temporal evolution dynamics)
- `n_head`: 4-8 (multi-directional vorticity transport)
- `n_experts`: 1-4 (can help with different vortex evolution regimes)
- `mlp_layers`: 3
- `branch_sizes`: [2+K] where K is number of input time frames
- `output_size`: M (number of future time frames to predict)

**Training:**
- `lr`: 1e-3 with OneCycleLR
- `optimizer`: AdamW with weight_decay=5e-6
- `batch_size`: 4-32 (time-series data can use larger batches)
- `epochs`: 500
- `grad_clip`: 1000.0
- `loss_name`: 'rel2'

**Temporal prediction specific:**
- **Time horizon**: Longer prediction time (larger T_final - T_initial) is harder - error grows
- **Input frames**: More initial frames (larger K) generally improves prediction but increases input dim
- **Prediction frames**: Can predict single future frame or multiple frames jointly
- **Data scaling**: Critical for temporal problems - performance improves dramatically with more data

**Performance (from paper Table 1):**
- **Part dataset** (fewer samples): 13.8% error
- **Full dataset** (5× more samples): 4.42% error
- **3× error reduction** with more data demonstrates excellent scalability
- Outperforms FNO-interp (8.20%), GK-Transformer (7.92%), OFormer (6.46%)

**Scaling behavior:**
- GNOT shows best scaling curve with data size among all methods tested
- Larger model capacity (transformer) better utilizes additional data than FNO or MIONet
- Data-efficient: achieves good performance even with smaller datasets

**Limitations:**
- Assumes smooth enough operator G: u₀ → uₜ exists (may fail for highly chaotic regimes)
- Longer time horizons require more data for accurate prediction
- No guarantee of physical constraints (energy conservation, enstrophy) without explicit enforcement

**Key insight:** GNOT's approach to time-dependent PDEs is to learn the temporal evolution operator directly from initial to final states, rather than iterative time-stepping. This spatial-only attention with temporal encoding via data avoids error accumulation and leverages GNOT's scalability. The 3× error reduction with more data confirms that GNOT's large transformer capacity is well-suited for complex temporal dynamics. For extremely long-time or highly chaotic predictions, combining GNOT with physics-informed constraints or multi-step strategies may be beneficial.
