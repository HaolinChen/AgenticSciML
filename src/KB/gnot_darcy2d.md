# GNOT for 2D Darcy Flow

**Keywords**: [PDE, elliptic, linear, forward-problem, darcy, 2D, irregular, GNOT, Transformer, linear-attention, multi-head, cross-attention, self-attention, MLP, l2-regularization, adamw, mse, pytorch]

**Problem:** GNOT (General Neural Operator Transformer) is a transformer-based neural operator architecture for learning PDE solution operators on irregular meshes with multiple input functions. The method addresses three key challenges in operator learning: (1) handling irregular discretized meshes where standard methods like FNO (limited to uniform grids) fail, (2) processing multiple heterogeneous input functions (boundary shapes, source functions, parameter vectors) efficiently, and (3) learning multi-scale solutions where different spatial regions have vastly different solution complexity. In this implementation, GNOT is applied to the 2D Darcy flow problem, a second-order linear elliptic PDE where the goal is to predict the pressure field from the spatially-varying permeability coefficient defined on a unit square.

**Issues addressed:**
- **Irregular mesh handling**: Unlike FNO which requires regular grids and FFT, GNOT uses attention mechanisms that naturally handle arbitrary point clouds and irregular meshes without interpolation or remeshing
- **Multiple input functions**: The heterogeneous normalized cross-attention (HNA) layer efficiently fuses information from arbitrary numbers of input functions (boundary shapes, domain-distributed functions, parameter vectors) through normalized linear attention with O(N) complexity
- **Multi-scale problems**: The geometric gating mechanism (mixture-of-experts with spatial coordinates as gates) acts as a soft domain decomposition, allowing different expert networks to specialize in different spatial regions or scales
- **Scalability**: Linear complexity O(N) with respect to sequence length enables processing meshes with thousands to millions of points, whereas standard attention has O(N²) complexity
- **Data efficiency**: Transformer architecture with large capacity scales better with increasing data compared to methods like MIONet or DeepONet

## Key Method

GNOT is a transformer-based neural operator with three key innovations:

**1. Heterogeneous Normalized Attention (HNA) Block:**
- **Cross-attention layer**: Takes query points and multiple conditional embeddings (from different input functions) as input
- **Key innovation**: Uses normalized softmax on queries and keys separately, then computes attention without the expensive softmax normalization in the denominator
- **Efficiency**: The normalized linear attention can be reordered as q̃ₜ · (Σᵢ k̃ᵢ ⊗ vᵢ), reducing complexity from O(NM) to O(N + M) where N is query length and M is key/value length
- **Heterogeneous**: Different MLPs for keys/values from different input functions ensure model capacity
- **Aggregation**: Averages outputs from all input functions with 1/L normalization for numerical stability

**2. Normalized Self-Attention:**
- Follows the cross-attention layer to further process query features
- Also uses normalized linear attention with same efficiency benefits
- The cascade of cross-attention → self-attention proved most effective in ablations

**3. Geometric Gating Mechanism (Soft Domain Decomposition):**
- Uses K expert FFN networks instead of a single FFN
- Gating network G(x): R^d → R^K takes spatial coordinates as input and outputs mixing weights
- Update: zₜ ← zₜ + Σᵢ pᵢ(xₜ) · Eᵢ(zₜ) where pᵢ(xₜ) = exp(Gᵢ(xₜ))/Σⱼexp(Gⱼ(xₜ))
- **Interpretation**: Acts as soft domain decomposition where different experts specialize in different spatial regions based on geometry
- Particularly effective for multi-scale problems with distinct subdomains

**Architecture Overview:**
1. **Input encoding**: Separate MLPs encode query points (trunk) and each input function (branches)
2. **Attention blocks (×N layers)**: Each block contains HNA cross-attention → FFN with gating → normalized self-attention → FFN with gating
3. **Output decoder**: MLP maps final embeddings to solution values

## Implementation

```python
# Heterogeneous Normalized Linear Cross-Attention
# Key innovation: Softmax normalization on q and k separately, then linear attention
class LinearCrossAttention(nn.Module):
    def __init__(self, config):
        super(LinearCrossAttention, self).__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd)
        # Separate key/value networks for each input function (heterogeneous)
        self.keys = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd)
                                   for _ in range(config.n_inputs)])
        self.values = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd)
                                     for _ in range(config.n_inputs)])
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_inputs = config.n_inputs
        self.attn_type = 'l1'

    def forward(self, x, y=None, layer_past=None):
        # x: query features [B, T1, C]
        # y: list of conditional embeddings [y[0], y[1], ..., y[m]]
        y = x if y is None else y
        B, T1, C = x.size()

        # Compute query and apply softmax normalization
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T1, hs]
        q = q.softmax(dim=-1)  # Normalize query

        out = q  # Initialize with query (skip connection)

        # Aggregate information from all input functions
        for i in range(self.n_inputs):
            _, T2, _ = y[i].size()
            # Compute key and value for i-th input function
            k = self.keys[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.values[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.softmax(dim=-1)  # Normalize key

            # Normalization coefficient for numerical stability
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)

            # Efficient linear attention: q @ (k^T @ v) instead of (q @ k^T) @ v
            # This reduces complexity from O(T1*T2) to O(T1 + T2)
            out = out + 1 * (q @ (k.transpose(-2, -1) @ v)) * D_inv

        # Output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out
```

```python
# Normalized Linear Self-Attention
# Similar to cross-attention but q, k, v all come from same sequence
class LinearAttention(nn.Module):
    def __init__(self, config):
        super(LinearAttention, self).__init__()
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.attn_type = 'l1'

    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()

        # Project to query, key, value
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)

        # Normalized linear attention (l1 normalization)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)
        k_cumsum = k.sum(dim=-2, keepdim=True)
        D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)

        # Efficient computation: q @ (k^T @ v) with O(N) complexity
        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        # Output projection
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y
```

```python
# Cross-Attention Block: combines cross-attention and self-attention
# Order matters: cross-attention first, then self-attention performs best
class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(CrossAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        # Separate layer norms for each input branch
        self.ln2_branch = nn.ModuleList([nn.LayerNorm(config.n_embd)
                                         for _ in range(config.n_inputs)])
        self.n_inputs = config.n_inputs
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)
        self.ln5 = nn.LayerNorm(config.n_embd)

        # Attention layers
        self.selfattn = LinearAttention(config)
        self.crossattn = LinearCrossAttention(config)

        # Activation function
        if config.act == 'gelu':
            self.act = GELU
        elif config.act == "tanh":
            self.act = Tanh
        elif config.act == 'relu':
            self.act = ReLU

        # Dropout for regularization
        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        self.resid_drop2 = nn.Dropout(config.resid_pdrop)

        # Feed-forward networks (can be extended to MoE with gating)
        self.mlp1 = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        )

    def ln_branchs(self, y):
        # Apply layer norm to each input branch separately
        return MultipleTensors([self.ln2_branch[i](y[i]) for i in range(self.n_inputs)])

    def forward(self, x, y):
        # Cross-attention: aggregate information from input functions y into query points x
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln_branchs(y)))
        x = x + self.mlp1(self.ln3(x))

        # Self-attention: propagate information among query points
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        x = x + self.mlp2(self.ln5(x))

        return x
```

```python
# Main GNOT Architecture
class CGPTNO(nn.Module):
    def __init__(self,
                 trunk_size=2,         # Dimension of query point coordinates
                 branch_sizes=None,    # List of dimensions for each input function
                 output_size=3,        # Dimension of output solution
                 n_layers=2,           # Number of attention blocks
                 n_hidden=64,          # Hidden dimension
                 n_head=1,             # Number of attention heads
                 n_inner=4,            # Inner dimension multiplier for FFN
                 mlp_layers=2,         # Layers in encoder/decoder MLPs
                 attn_type='linear',   # Type of attention
                 act='gelu',           # Activation function
                 ffn_dropout=0.0,      # Dropout rate
                 attn_dropout=0.0,
                 horiz_fourier_dim=0,  # Optional Fourier feature embedding
                 ):
        super(CGPTNO, self).__init__()

        self.horiz_fourier_dim = horiz_fourier_dim
        self.trunk_size = trunk_size
        self.branch_sizes = branch_sizes
        self.output_size = output_size

        # Encoder MLP for query points (trunk network)
        self.trunk_mlp = MLP(self.trunk_size, n_hidden, n_hidden, n_layers=mlp_layers, act=act)

        # Encoder MLPs for input functions (branch networks)
        if branch_sizes:
            self.n_inputs = len(branch_sizes)
            self.branch_mlps = nn.ModuleList([MLP(bsize, n_hidden, n_hidden,
                                                   n_layers=mlp_layers, act=act)
                                              for bsize in self.branch_sizes])
        else:
            self.n_inputs = 0

        # Configure attention blocks
        self.gpt_config = GPTConfig(attn_type=attn_type, embd_pdrop=ffn_dropout,
                                    resid_pdrop=ffn_dropout, attn_pdrop=attn_dropout,
                                    n_embd=n_hidden, n_head=n_head, n_layer=n_layers,
                                    block_size=128, act=act, branch_sizes=branch_sizes,
                                    n_inputs=self.n_inputs, n_inner=n_inner)

        # Stack of attention blocks
        self.blocks = nn.Sequential(*[CrossAttentionBlock(self.gpt_config)
                                      for _ in range(self.gpt_config.n_layer)])

        # Decoder MLP for output
        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers)

        self.__name__ = 'CGPT'

    def forward(self, g, u_p, inputs):
        # g: DGL graph containing query points
        # u_p: additional parameters
        # inputs: list of input functions

        # Unbatch graph and extract node features (query point coordinates)
        gs = dgl.unbatch(g)
        x = pad_sequence([_g.ndata['x'] for _g in gs]).permute(1, 0, 2)  # [B, T1, F]

        # Concatenate with parameters
        x = torch.cat([x, u_p.unsqueeze(1).repeat([1, x.shape[1], 1])], dim=-1)

        # Optional Fourier feature embedding
        if self.horiz_fourier_dim > 0:
            x = horizontal_fourier_embedding(x, self.horiz_fourier_dim)

        # Encode query points
        x = self.trunk_mlp(x)

        # Encode input functions
        if self.n_inputs:
            z = MultipleTensors([self.branch_mlps[i](inputs[i])
                                for i in range(self.n_inputs)])
        else:
            z = MultipleTensors([x])

        # Apply attention blocks
        for block in self.blocks:
            x = block(x, z)

        # Decode to output
        x = self.out_mlp(x)

        # Concatenate outputs for all graphs in batch
        x_out = torch.cat([x[i, :num] for i, num in enumerate(g.batch_num_nodes())], dim=0)

        return x_out
```

```python
# Training loop with key components
def train_batch(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.999):
    optimizer.zero_grad()

    # Unpack data: g (graph), u_p (parameters), g_u (input functions)
    g, u_p, g_u = data
    g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)

    # Forward pass
    out = model(g, u_p, g_u)

    # Compute loss (relative L2 + regularization)
    y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
    loss, reg, _ = loss_func(g, y_pred, y)
    loss = loss + reg

    # Backward pass with gradient clipping
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # Update learning rate (step-wise scheduler)
    if lr_scheduler:
        lr_scheduler.step()

    return (loss.item(), reg.item())
```

## Critical Parameters

**Architecture:**
- `n_hidden`: Hidden dimension (64-256), controls model capacity
- `n_layers`: Number of attention blocks (3-4 typical), deeper = more expressive but slower
- `n_head`: Number of attention heads (1-8), more heads can capture diverse patterns
- `n_inner`: Inner FFN dimension multiplier (4 typical), controls FFN capacity
- `mlp_layers`: Layers in encoder/decoder MLPs (2-3), affects encoding quality
- `n_experts`: Number of expert FFNs for geometric gating (1 for simple problems, 3-4 for multi-scale)

**Training:**
- `lr`: Learning rate (1e-3 typical with OneCycleLR scheduler)
- `lr_method`: 'cycle' (OneCycleLR), 'step' (StepLR), or 'warmup' (LambdaLR warming)
- `optimizer`: AdamW with weight_decay=5e-6 for regularization
- `batch_size`: 4-32 depending on mesh size and GPU memory
- `epochs`: 500 typical
- `grad_clip`: Gradient clipping threshold (1000.0 default)
- `loss_name`: 'rel2' (relative L2 error) is standard metric

**Regularization:**
- `ffn_dropout`: Dropout rate for FFN layers (0.0-0.1)
- `attn_dropout`: Dropout rate for attention (0.0-0.1)
- `weight_decay`: L2 regularization on weights (5e-6)

**Data:**
- `normalize_x`: Input normalization ('unit', 'minmax', 'none')
- `use_normalizer`: Output normalization ('unit', 'minmax', 'quantile', 'none')

**Key Insight:** The number of experts should match the number of physically distinct subdomains for multi-scale problems (e.g., 3 experts for Heat dataset with 3 subdomains gave best results).
