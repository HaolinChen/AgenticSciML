# Convolutional Neural Operator (CNO) for Discontinuous Transport Equation

**Keywords**: PDE, hyperbolic, linear, forward-problem, advection, 2D, periodic, discontinuous, FNO, CNN, U-Net, ResNet, multi-resolution, pytorch, adam, mae, gpu

**Problem:** CNO is a neural operator architecture designed to learn mappings between infinite-dimensional function spaces for PDE solutions. Traditional CNNs suffer from aliasing errors when applied to operator learning because standard convolutional operations do not respect the continuous-discrete equivalence (CDE) property - meaning that discretizing a continuous operator and then applying it does not yield the same result as applying the operator continuously and then discretizing. CNO addresses this by modifying all CNN operations (convolution, upsampling, downsampling, activation) to be bandlimit-preserving, ensuring that the discrete network faithfully represents its continuous counterpart across different resolutions.

**Issues addressed:**
- **Aliasing errors**: Standard CNNs introduce aliasing when learning operators due to improper handling of high-frequency components during upsampling/downsampling and nonlinear activations. CNO eliminates this through careful filtering.
- **Resolution dependence**: CNO maintains consistent performance across different grid resolutions due to its continuous-discrete equivalence property.
- **Function space inconsistency**: CNO ensures that operators work consistently in continuous function spaces by preserving bandlimits throughout all operations.
- **Discontinuous solutions**: The filtered activation layers and multi-resolution U-Net structure enable CNO to handle transport problems with sharp discontinuities and jumps, which are particularly challenging for standard neural operators.

## Key Method

CNO implements a modified U-Net architecture where every operation is designed to preserve bandlimited functions. The key innovations are:

1. **Filtered Activation Layers**: Upsample → activate → downsample process ensures bandlimit preservation even through nonlinearities.

2. **Bandlimit-Preserving Convolutions**: Convolutions with careful cutoff frequency design.

3. **Operator U-Net Structure**: Lift → Encoder → Bottleneck → Decoder → Project with skip connections.

4. **Continuous-Discrete Equivalence (CDE)**: Resolution-independent operator learning.

The discontinuous transport equation describes advection of quantities with sharp fronts or step discontinuities. This is particularly challenging for neural operators because discontinuities introduce high-frequency content that can cause aliasing in standard CNN-based methods. CNO's bandlimit-preserving design helps mitigate these aliasing errors while maintaining representation accuracy.

## Implementation

The implementation for discontinuous transport uses the same CNO architecture as other problems. See cno_allen_cahn entry for complete code details of CNOBlock, ResidualBlock, and CNO class.

### Problem-Specific Configuration

```python
# Training hyperparameters for discontinuous transport
training_properties = {
    "learning_rate": 0.001,
    "weight_decay": 1e-6,
    "scheduler_step": 10,
    "scheduler_gamma": 0.98,
    "epochs": 1000,
    "batch_size": 16,
    "exp": 1,                    # L1 loss for robustness to discontinuities
    "training_samples": 256
}

# Model architecture for discontinuous transport
model_architecture = {
    "N_layers": 3,
    "channel_multiplier": 32,
    "N_res": 4,
    "N_res_neck": 6,
    "in_size": 64,               # 64x64 grid resolution
    "kernel_size": 3,
    "activation": 'cno_lrelu',

    # Critical filtering parameters for CDE property
    "cutoff_den": 2.0001,
    "lrelu_upsampling": 2,
    "half_width_mult": 0.8,
    "filter_size": 6,
    "radial_filter": 0
}

# Load discontinuous transport problem
from Problems.CNOBenchmarks import DiscContTranslation
example = DiscContTranslation(model_architecture, device, batch_size, training_samples)
```

## Critical Parameters

**Architecture Parameters:**
- `N_layers` (default: 3): Number of encoder/decoder levels for multi-scale feature extraction.
- `channel_multiplier` (default: 32): Channel growth factor controlling model capacity.
- `in_size` (default: 64): Grid resolution (64x64) for discontinuous transport.

**Filtering Parameters (Critical for CDE - Do Not Modify):**
- `cutoff_den` (default: 2.0001): Bandlimit cutoff frequency for Nyquist-safe operations.
- `lrelu_upsampling` (default: 2): Upsampling factor ensuring bandlimit preservation through activations.
- `half_width_mult` (default: 0.8): Filter transition bandwidth.
- `filter_size` (default: 6): Filter tap count = 2*filter_size for frequency response quality.

**Training Parameters:**
- `learning_rate` (default: 0.001): AdamW initial learning rate.
- `exp` (default: 1): L1 loss is particularly suitable for discontinuous solutions as it's more robust to outliers than L2.
- `training_samples` (default: 256): Number of discontinuous transport trajectories for training.

**Key Insight:** For discontinuous transport, the bandlimit-preserving operations are crucial for avoiding Gibbs-like oscillations near discontinuities. While the filtering cannot eliminate all artifacts from discontinuities (which introduce infinite frequency content), it significantly reduces aliasing compared to standard CNNs. The L1 loss provides additional robustness to the sharp features.
