# Convolutional Neural Operator Foundation Model (CNO-FM)

**Keywords**: PDE, forward-problem, FNO, CNN, U-Net, ResNet, multi-resolution, transfer-learning, meta-learning, pytorch, adam, mae, gpu

**Problem:** CNO-FM extends the Convolutional Neural Operator architecture to create a foundation model that can be pretrained on multiple diverse PDE types and then efficiently fine-tuned for new problems. Traditional neural operator models are trained from scratch for each specific PDE, requiring substantial training data and computational resources. CNO-FM addresses this by leveraging transfer learning: a single model is pretrained on a diverse mixture of PDEs (Navier-Stokes, Euler, wave, Allen-Cahn, etc.) and can then be rapidly adapted to new tasks with minimal data and training time through fine-tuning.

**Issues addressed:**
- **Data inefficiency**: Training neural operators from scratch requires large datasets. CNO-FM enables learning with significantly fewer samples through transfer learning.
- **Computational cost**: Pretraining once on diverse PDEs eliminates the need for expensive training from scratch for each new problem.
- **Aliasing errors**: Inherits CNO's bandlimit-preserving operations to avoid aliasing across different resolutions.
- **Domain adaptation**: Can adapt to new PDE types, different input/output dimensions, and varying problem domains with minimal fine-tuning.
- **Time-varying dynamics**: Uses conditional instance normalization (FiLM) to handle time-dependent PDEs, allowing the model to condition predictions on temporal evolution.

## Key Method

CNO-FM builds upon the standard CNO architecture with several key enhancements for foundation modeling:

1. **Time-Conditional Normalization (FiLM)**: Instead of standard batch normalization, CNO-FM uses Feature-wise Linear Modulation (FiLM) that conditions the normalization on timestep information. This allows a single model to handle temporal dynamics across different PDEs.

2. **Multi-Task Pretraining**: The model is pretrained on a diverse mixture of PDEs:
   - Navier-Stokes (multiple variants: Brownian forcing, PWC forcing, Gaussian forcing, vortex, shear layer)
   - Compressible Euler (Kelvin-Helmholtz, Riemann problems, Gaussian initial conditions)
   - Wave equations (seismic, Gaussian)
   - Allen-Cahn phase-field
   - Darcy flow
   - Additional problems: Rayleigh-Taylor, Kolmogorov flow, etc.

3. **Adaptive Lift/Project Layers**: When fine-tuning on problems with different input/output dimensions than pretraining, the lift and projection layers are replaced while keeping the core CNO encoder-decoder frozen or lightly fine-tuned.

4. **Bandlimit-Preserving Operations**: Inherits all CNO innovations (filtered activations, bandlimit-preserving convolutions, operator U-Net structure, CDE property).

5. **Optional Attention in Bottleneck**: Can incorporate Vision Transformer (ViT) blocks in the bottleneck for enhanced representation capacity.

## Implementation

### Time-Conditional Instance Normalization (FiLM)

```python
class FILM(torch.nn.Module):
    """
    Feature-wise Linear Modulation for time-conditional normalization.
    Learns to modulate normalization parameters based on timestep.
    """
    def __init__(self, channels, dim=[0,2,3], s=128, intermediate=128):
        super(FILM, self).__init__()
        self.channels = channels
        self.s = s

        # MLPs for learning scale and bias from timestep
        self.inp2lat_scale = nn.Linear(1, intermediate, bias=True)
        self.lat2scale = nn.Linear(intermediate, channels)
        self.inp2lat_bias = nn.Linear(1, intermediate, bias=True)
        self.lat2bias = nn.Linear(intermediate, channels)

        # Initialize to identity transformation (scale=1, bias=0)
        self.inp2lat_scale.weight.data.fill_(0)
        self.lat2scale.weight.data.fill_(0)
        self.lat2scale.bias.data.fill_(1)      # Start with scale=1
        self.inp2lat_bias.weight.data.fill_(0)
        self.lat2bias.weight.data.fill_(0)     # Start with bias=0

        # Choose normalization type based on dim
        if dim == [0,2,3]:
            self.norm = nn.BatchNorm2d(channels)      # Batch norm
        elif dim == [2,3]:
            self.norm = nn.InstanceNorm2d(channels, affine=True)  # Instance norm
        elif dim == [1,2,3]:
            self.norm = nn.LayerNorm([channels, s, s])  # Layer norm

    def forward(self, x, timestep):
        """
        Args:
            x: Feature tensor (B, C, H, W)
            timestep: Time information (B,)
        Returns:
            Modulated features
        """
        # Normalize features
        x = self.norm(x)

        # Compute time-dependent scale and bias
        timestep = timestep.reshape(-1, 1).type_as(x)
        scale = self.lat2scale(self.inp2lat_scale(timestep))  # (B, C)
        bias = self.lat2bias(self.inp2lat_bias(timestep))      # (B, C)

        # Reshape and expand to match spatial dimensions
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)  # (B, C, H, W)
        bias = bias.unsqueeze(2).unsqueeze(3).expand_as(x)    # (B, C, H, W)

        # Apply affine transformation: scale * x + bias
        return x * scale + bias
```

### Pretraining Configuration

```python
# Training hyperparameters for foundation model pretraining
training_properties = {
    "learning_rate": 0.00075,
    "weight_decay": 1e-6,
    "scheduler_step": 1,
    "scheduler_gamma": 0.9,
    "epochs": 100,
    "batch_size": 32,
    "time_steps": 7,          # Number of time steps to use
    "dt": 1,                  # Time step interval (1=all steps, 2=every other)
    "training_samples": 32,   # Samples per PDE type
    "time_input": 1,          # Include time as input channel
    "allowed": 'all',         # Training mode: 'all'=All2All, 'one2all'=One2All, 'one'=AR
}

# Model architecture for foundation model
model_architecture = {
    # Larger architecture than single-task CNO
    "N_layers": 4,            # More layers (vs 3 for single-task)
    "channel_multiplier": 32,
    "N_res": 8,               # More residual blocks (vs 4)
    "N_res_neck": 8,          # More bottleneck blocks (vs 6)

    # Time-conditional normalization
    "batch_norm": 1,          # Use standard BN if is_time==0
    "is_time": 1,             # Use conditional normalization (FiLM)
    "nl_dim": "23",           # Normalization dims: '23'=InstanceNorm, '023'=BatchNorm, '123'=LayerNorm

    # Grid and architecture
    "in_size": 128,           # Higher resolution for foundation model
    "activation": 'cno_lrelu',

    # Optional attention in bottleneck
    "is_att": False,          # Can add Vision Transformer
    "patch_size": 1,
    "dim_multiplier": 1,
    "depth": 2,
    "heads": 2,
    "dim_head_multiplier": 0.5,
    "mlp_dim_multiplier": 1.0,
    "emb_dropout": 0.
}

# For pretraining on diverse PDE mixture
which_example = "eul_ns_mix1"  # Mix of Euler and Navier-Stokes

# Available pretraining datasets:
# Navier-Stokes: "ns_brownian", "ns_pwc", "ns_gauss", "ns_sin", "ns_vortex", "ns_shear"
# Euler: "eul_kh", "eul_riemann", "eul_riemann_kh", "eul_riemann_cur", "eul_gauss"
# Others: "rich_mesh", "rayl_tayl", "kolmogorov", "wave_seismic", "wave_gauss",
#         "allen_cahn", "airfoil", "poisson_gauss", "helmholtz"
```

### Fine-Tuning Configuration

```python
# Fine-tuning hyperparameters
properties = {
    "num_trajectories": 128,   # Few-shot: only 128 samples needed
    "epochs": 221,

    # Different learning rates for different parts
    "lr": 0.00005,             # LR for frozen/lightly-tuned CNO backbone
    "lr_norm": 0.00125,        # LR for normalization layers (FiLM)
    "lr_emb": 0.0005,          # LR for lift/project layers

    "scheduler_step": 5,
    "scheduler_gamma": 0.9,

    # Adapt to different dimensions
    "is_different_dim": 1,     # New problem has different I/O dims?
    "in_dim_tune": 4,          # New input channels
    "out_dim_tune": 3,         # New output channels

    "steps": 7                 # Time steps in fine-tuning dataset
}

# Load pretrained foundation model
folder = "path/to/pretrained/model"
in_dim = 5   # Foundation model input dim
out_dim = 4  # Foundation model output dim

# Load model and dataset
model, loader_dict = load_model(folder,
                               which_example=which_example,  # e.g., "kolmogorov"
                               in_dim=in_dim,
                               out_dim=out_dim,
                               steps=steps)

# Initialize fine-tuning: replace lift/project if dimensions changed
model = initialize_FT(model=model,
                     old_in_dim=5,
                     new_in_dim=in_dim_tune,
                     new_out_dim=out_dim_tune,
                     old_out_dim=4)

# Fine-tune with different LRs for different components
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': lr},           # Base CNO
    {'params': norm_params, 'lr': lr_norm},          # FiLM layers
    {'params': lift_project_params, 'lr': lr_emb}    # Lift/Project
], weight_decay=weight_decay)
```

## Critical Parameters

**Foundation Model Architecture (Larger than Single-Task):**
- `N_layers` (default: 4): More encoder/decoder levels than single-task CNO (3) for increased capacity.
- `channel_multiplier` (default: 32): Same as single-task.
- `N_res` (default: 8): Double the residual blocks (vs 4 for single-task) for better representation.
- `N_res_neck` (default: 8): Larger bottleneck (vs 6) to handle diverse PDE types.
- `in_size` (default: 128): Higher resolution than standard CNO (64 or 128 depending on problem).

**Time-Conditional Normalization:**
- `is_time` (default: 1): Enable FiLM for time-dependent PDEs. Set to 0 for standard batch norm.
- `nl_dim` (default: "23"): Normalization type - "23"=InstanceNorm, "023"=BatchNorm, "123"=LayerNorm.

**Pretraining Parameters:**
- `time_steps` (default: 7): Number of temporal snapshots per trajectory.
- `dt` (default: 1): Temporal subsampling (1=all steps, 2=every other step).
- `training_samples` (default: 32): Samples per PDE type (fewer needed due to multi-task learning).
- `allowed` (default: 'all'): Training mode - 'all'=All2All (predict all steps), 'one2all'=One2All, 'one'=autoregressive.

**Fine-Tuning Parameters:**
- `num_trajectories` (default: 128): Few-shot learning with minimal data.
- `lr` (default: 0.00005): Small LR for backbone (often frozen or lightly tuned).
- `lr_norm` (default: 0.00125): Higher LR for FiLM layers (actively adapted).
- `lr_emb` (default: 0.0005): Medium LR for lift/project (replaced if dimensions change).
- `is_different_dim` (default: 1): Adapt to problems with different I/O dimensions.

**Filtering Parameters (Same as Standard CNO - Do Not Modify):**
- `cutoff_den`, `lrelu_upsampling`, `half_width_mult`, `filter_size`, `radial_filter`: Same defaults as standard CNO to maintain CDE property.

**Key Insight:** CNO-FM demonstrates transfer learning for PDEs. The foundation model is pretrained on diverse PDE types with time-conditional normalization (FiLM), enabling adaptation to new problems with minimal data (128 trajectories vs 256+ for training from scratch). Different learning rates for backbone, normalization, and lift/project layers allow efficient fine-tuning. The model can handle out-of-context scenarios (different I/O dimensions, new PDE types) by replacing only the lift/project layers while leveraging the pretrained encoder-decoder. This dramatically reduces training time and data requirements for new PDE applications.
