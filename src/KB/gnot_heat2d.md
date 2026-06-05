# GNOT for 2D Multi-Scale Heat Conduction

**Keywords**: [PDE, parabolic, linear, forward-problem, heat, 2D, irregular, mixed-bc, multi-scale, heterogeneous, GNOT, Transformer, linear-attention, cross-attention, self-attention, MoE, geometric-gating, MLP, l2-regularization, adamw, mse, pytorch]

**Problem:** GNOT applied to a 2D steady-state heat conduction problem with multiple physically distinct subdomains. The domain is a rectangle [0,9]² divided into three parts by two splines, where each subdomain has different thermal properties. This is a prototypical multi-scale problem where solution behavior varies dramatically across subdomains. The input functions include (1) boundary temperature distribution on the top boundary, and (2) parameters defining the spline curves that segment the domain. The goal is to predict the temperature field T throughout the domain. This problem demonstrates GNOT's capability to handle multiple heterogeneous inputs and multi-scale physics.

**Issues addressed:**
- **Multi-scale physics**: Different subdomains have vastly different thermal conductivities and heat source distributions, creating solution features at multiple spatial scales
- **Multiple heterogeneous inputs**: Must process both boundary conditions (distributed function) and geometric parameters (vectors) defining domain decomposition
- **Soft domain decomposition**: The geometric gating mechanism with multiple experts provides a learnable, soft version of classical domain decomposition methods
- **Subdomain specialization**: Different expert networks can specialize for different physical subdomains without explicit hard partitioning
- **Scalability with data**: Performance improves significantly (17.4% → 4.13% → 2.56% error) as dataset size grows from 1100 to 5500 samples, demonstrating GNOT's large model capacity

## Key Method

GNOT with emphasis on **Geometric Gating for Multi-Scale Problems**:

1. **Heterogeneous Normalized Cross-Attention**: Handles multiple input types:
   - Branch network 1: Encodes boundary temperature distribution {(xᵢ, Tᵢ)}
   - Branch network 2: Encodes spline parameters defining subdomain boundaries
   - Separate MLPs for each input ensure proper capacity

2. **Geometric Gating (Mixture-of-Experts)**: Critical for this multi-scale problem
   - Uses K=3 expert FFN networks (matching the 3 physical subdomains)
   - Gating network G(x): R² → R³ takes spatial coordinates as input
   - Each expert specializes in a different subdomain/scale
   - Soft weighting allows smooth transitions at subdomain boundaries
   - Update: zₜ ← zₜ + Σᵢ₌₁³ pᵢ(xₜ) · Eᵢ(zₜ) where pᵢ(xₜ) = softmax(G(xₜ))ᵢ

3. **Normalized Self-Attention**: Propagates multi-scale information spatially

**Why geometric gating helps:**
- Classical domain decomposition uses hard subdomain boundaries and separate models
- GNOT's soft gating learns subdomain assignment from data
- Geometric coordinates naturally correlate with physical subdomain structure
- Smooth gating weights prevent discontinuities at subdomain boundaries

## Implementation

```python
# Same base CGPTNO architecture with n_experts > 1
# The key difference is in the FFN layers which become mixture-of-experts

class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        # ... (same attention layers as gnot_darcy2d)

        # Instead of single FFN, use K expert FFNs
        self.n_experts = config.n_experts  # Set to 3 for Heat problem

        if self.n_experts > 1:
            # Multiple expert networks
            self.expert_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.n_embd, config.n_inner),
                    self.act(),
                    nn.Linear(config.n_inner, config.n_embd),
                )
                for _ in range(self.n_experts)
            ])

            # Gating network: spatial coords → expert weights
            self.gating_network = nn.Sequential(
                nn.Linear(config.space_dim, 64),  # space_dim=2 for 2D
                self.act(),
                nn.Linear(64, self.n_experts)  # Output logits for K experts
            )
        else:
            # Single FFN (standard transformer)
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, config.n_inner),
                self.act(),
                nn.Linear(config.n_inner, config.n_embd),
            )

    def forward(self, x, y, spatial_coords):
        # Cross-attention
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln_branchs(y)))

        # FFN with geometric gating
        if self.n_experts > 1:
            # Compute gating weights from spatial coordinates
            gate_logits = self.gating_network(spatial_coords)  # [B, T, K]
            gate_weights = F.softmax(gate_logits, dim=-1)  # [B, T, K]

            # Weighted combination of expert outputs
            mlp_out = 0
            for i in range(self.n_experts):
                expert_out = self.expert_mlps[i](self.ln3(x))  # [B, T, D]
                mlp_out = mlp_out + gate_weights[:, :, i:i+1] * expert_out

            x = x + mlp_out
        else:
            x = x + self.mlp(self.ln3(x))

        # Self-attention (similar gating applied here too)
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        x = x + self.mlp2(self.ln5(x))  # (or gated version)

        return x
```

```python
# Input encoding for multiple heterogeneous inputs
def encode_heat_inputs(boundary_temp, spline_params):
    """
    Encode multiple input functions for heat conduction problem

    Args:
        boundary_temp: {(xᵢ, Tᵢ)} - temperature distribution on boundary
        spline_params: θ ∈ R^p - parameters defining subdomain boundaries

    Returns:
        List of conditional embeddings for cross-attention
    """
    # Branch MLP 1: encode boundary temperature (distributed function)
    boundary_embed = branch_mlp_1(boundary_temp)  # [B, N1, D]

    # Branch MLP 2: encode spline parameters (vector)
    param_embed = branch_mlp_2(spline_params)  # [B, 1, D]

    return [boundary_embed, param_embed]
```

## Critical Parameters

**Architecture (critical for multi-scale problems):**
- `n_hidden`: 256 (larger capacity needed for multiple scales)
- `n_layers`: 4 (deeper network for complex multi-scale interactions)
- `n_head`: 8 (helps capture multi-directional heat flow)
- `n_experts`: **3** (matches number of physical subdomains - critical parameter!)
- `n_inner`: 4 (FFN hidden dimension = 4 × n_hidden = 1024)
- `mlp_layers`: 4 (deeper encoding for multiple input types)

**Training:**
- `lr`: 1e-3 with OneCycleLR schedule
- `optimizer`: AdamW with weight_decay=5e-6
- `batch_size`: 4-16 (multi-scale problems benefit from larger batches)
- `epochs`: 500
- `grad_clip`: 1000.0
- `loss_name`: 'rel2'

**Multi-scale specific:**
- `n_experts=3` is crucial: ablation study shows best performance when number of experts matches number of subdomains
- Using too many experts (≥8) degrades performance due to over-specialization
- Geometric gating with spatial coordinates outperforms random gating or no gating

**Data scaling:**
- Small dataset (1100 samples): 4.13% error
- Full dataset (5500 samples): 2.56% error
- GNOT benefits more from additional data than MIONet, demonstrating better scalability

**Key insight:** For multi-scale problems with K known subdomains, set `n_experts=K` and use geometric gating. This provides a learnable soft domain decomposition that outperforms single-model approaches.
