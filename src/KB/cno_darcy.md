# Convolutional Neural Operator (CNO) for Darcy Flow

**Keywords**: PDE, elliptic, linear, forward-problem, darcy, 2D, dirichlet, heterogeneous, FNO, CNN, U-Net, ResNet, multi-resolution, pytorch, adam, mae, gpu

**Problem:** CNO is a neural operator architecture designed to learn mappings between infinite-dimensional function spaces for PDE solutions. Traditional CNNs suffer from aliasing errors when applied to operator learning because standard convolutional operations do not respect the continuous-discrete equivalence (CDE) property - meaning that discretizing a continuous operator and then applying it does not yield the same result as applying the operator continuously and then discretizing. CNO addresses this by modifying all CNN operations (convolution, upsampling, downsampling, activation) to be bandlimit-preserving, ensuring that the discrete network faithfully represents its continuous counterpart across different resolutions.

**Issues addressed:**
- **Aliasing errors**: Standard CNNs introduce aliasing when learning operators due to improper handling of high-frequency components during upsampling/downsampling and nonlinear activations. CNO eliminates this through careful filtering.
- **Resolution dependence**: CNO maintains consistent performance across different grid resolutions due to its continuous-discrete equivalence property.
- **Function space inconsistency**: CNO ensures that operators work consistently in continuous function spaces by preserving bandlimits throughout all operations.
- **Heterogeneous coefficients**: The multi-resolution U-Net structure enables CNO to handle Darcy flow with spatially-varying permeability fields, capturing both smooth and heterogeneous regions.

## Key Method

CNO implements a modified U-Net architecture where every operation is designed to preserve bandlimited functions. For Darcy flow, the operator learns to map from permeability coefficient fields to pressure/hydraulic head solutions of the elliptic PDE −∇·(a(x)∇u(x)) = f.

The key innovations are:

1. **Filtered Activation Layers**: Upsample → activate → downsample process ensures outputs remain bandlimited.

2. **Bandlimit-Preserving Convolutions**: Convolutions with careful cutoff frequency design to avoid aliasing.

3. **Operator U-Net Structure**: Lift → Encoder → Bottleneck → Decoder → Project architecture with skip connections.

4. **Continuous-Discrete Equivalence (CDE)**: Resolution-independent operator learning.

The Darcy flow equation is an elliptic PDE modeling subsurface flow in porous media, where heterogeneous permeability fields create complex pressure distributions. CNO's multi-scale architecture is well-suited for capturing both local heterogeneity and global pressure patterns.

## Implementation

The implementation for Darcy flow uses the same CNO architecture as other problems. See cno_allen_cahn entry for complete code details of CNOBlock, ResidualBlock, and CNO class.

### Problem-Specific Configuration

```python
# Training hyperparameters for Darcy flow
training_properties = {
    "learning_rate": 0.001,
    "weight_decay": 1e-6,
    "scheduler_step": 10,
    "scheduler_gamma": 0.98,
    "epochs": 1000,
    "batch_size": 16,
    "exp": 1,                    # L1 loss
    "training_samples": 256
}

# Model architecture for Darcy flow
model_architecture = {
    "N_layers": 3,
    "channel_multiplier": 32,
    "N_res": 4,
    "N_res_neck": 6,
    "in_size": 64,               # 64x64 grid resolution
    "kernel_size": 3,
    "activation": 'cno_lrelu',

    # Critical filtering parameters for CDE
    "cutoff_den": 2.0001,
    "lrelu_upsampling": 2,
    "half_width_mult": 0.8,
    "filter_size": 6,
    "radial_filter": 0
}

# Load Darcy flow problem
from Problems.CNOBenchmarks import Darcy
example = Darcy(model_architecture, device, batch_size, training_samples)
```

## Critical Parameters

**Architecture Parameters:**
- `N_layers` (default: 3): Encoder/decoder depth for multi-scale feature extraction.
- `channel_multiplier` (default: 32): Channel growth controlling model capacity.
- `in_size` (default: 64): Grid resolution (64x64) for Darcy flow.

**Filtering Parameters (Do Not Modify):**
- `cutoff_den` (default: 2.0001): Maintains Nyquist-safe bandlimit preservation.
- `lrelu_upsampling` (default: 2): Ensures filtered activation preserves bandlimits.
- `half_width_mult` (default: 0.8): Filter transition width.
- `filter_size` (default: 6): Filter quality (taps = 2*filter_size).

**Training Parameters:**
- `learning_rate` (default: 0.001): AdamW initial learning rate.
- `batch_size` (default: 16): Training batch size.
- `training_samples` (default: 256): Number of permeability-pressure pairs for training.

**Key Insight:** The bandlimit-preserving operations are essential for CNO's resolution-independent performance. For Darcy flow, the architecture parameters can be tuned, but filtering parameters should remain at default values to maintain the CDE property.
