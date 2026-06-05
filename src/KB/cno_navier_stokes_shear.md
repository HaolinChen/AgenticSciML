# Convolutional Neural Operator (CNO) for Navier-Stokes Shear Layer

**Keywords**: PDE, hyperbolic, parabolic, nonlinear, forward-problem, navier-stokes, 2D, periodic, turbulent, vortex-structure, FNO, CNN, U-Net, ResNet, multi-resolution, pytorch, adam, mae, gpu

**Problem:** CNO is a neural operator architecture designed to learn mappings between infinite-dimensional function spaces for PDE solutions. Traditional CNNs suffer from aliasing errors when applied to operator learning because standard convolutional operations do not respect the continuous-discrete equivalence (CDE) property - meaning that discretizing a continuous operator and then applying it does not yield the same result as applying the operator continuously and then discretizing. CNO addresses this by modifying all CNN operations (convolution, upsampling, downsampling, activation) to be bandlimit-preserving, ensuring that the discrete network faithfully represents its continuous counterpart across different resolutions.

**Issues addressed:**
- **Aliasing errors**: Standard CNNs introduce aliasing when learning operators due to improper handling of high-frequency components during upsampling/downsampling and nonlinear activations. CNO eliminates this through careful filtering.
- **Resolution dependence**: CNO maintains consistent performance across different grid resolutions due to its continuous-discrete equivalence property.
- **Function space inconsistency**: CNO ensures that operators work consistently in continuous function spaces by preserving bandlimits throughout all operations.
- **Turbulent flows and vortex structures**: The shear layer problem exhibits complex turbulent dynamics with vortex formation and interactions. CNO's multi-resolution U-Net structure enables capturing both large-scale flow patterns and small-scale turbulent features.

## Key Method

CNO implements a modified U-Net architecture where every operation is designed to preserve bandlimited functions. For the shear layer problem, the operator learns the dynamics of incompressible Navier-Stokes equations with initial shear profiles that develop into turbulent mixing layers with Kelvin-Helmholtz instabilities.

The key innovations are:

1. **Filtered Activation Layers**: Upsample → activate → downsample process ensures bandlimit preservation through nonlinearities.

2. **Bandlimit-Preserving Convolutions**: Convolutions with careful cutoff frequency design to avoid aliasing.

3. **Operator U-Net Structure**: Lift → Encoder → Bottleneck → Decoder → Project with skip connections for multi-scale feature extraction.

4. **Continuous-Discrete Equivalence (CDE)**: Resolution-independent operator learning enabling generalization across grid resolutions.

The Navier-Stokes shear layer is a classic benchmark in fluid dynamics, featuring a transition from laminar shear to turbulent mixing with coherent vortex structures. This problem tests the operator's ability to capture both the large-scale rollup of vortices and the small-scale turbulent features.

## Implementation

The implementation for Navier-Stokes shear layer uses the same CNO architecture as other problems. See cno_allen_cahn entry for complete code details of CNOBlock, ResidualBlock, and CNO class.

### Problem-Specific Configuration

```python
# Training hyperparameters for Navier-Stokes shear layer
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

# Model architecture for shear layer
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

# Load shear layer problem
from Problems.CNOBenchmarks import ShearLayer
example = ShearLayer(model_architecture, device, batch_size, training_samples, size=64)
```

## Critical Parameters

**Architecture Parameters:**
- `N_layers` (default: 3): Number of encoder/decoder levels for capturing multi-scale turbulent structures.
- `channel_multiplier` (default: 32): Channel growth factor controlling model capacity.
- `N_res` (default: 4): Residual blocks in skip connections for feature mixing.
- `N_res_neck` (default: 6): Residual blocks in bottleneck for processing coarsest-scale features.
- `in_size` (default: 64): Grid resolution (64x64) for shear layer simulation.

**Filtering Parameters (Critical for CDE - Do Not Modify):**
- `cutoff_den` (default: 2.0001): Bandlimit cutoff frequency ensuring Nyquist-safe operations.
- `lrelu_upsampling` (default: 2): Upsampling factor in filtered activations ensuring bandlimit preservation.
- `half_width_mult` (default: 0.8): Filter transition bandwidth affecting filter sharpness.
- `filter_size` (default: 6): Filter tap count = 2*filter_size controlling frequency response quality.
- `radial_filter` (default: 0): Use separable (0) vs radial (1) filters. Separable is faster.

**Training Parameters:**
- `learning_rate` (default: 0.001): AdamW initial learning rate.
- `weight_decay` (default: 1e-6): L2 regularization for preventing overfitting.
- `batch_size` (default: 16): Training batch size balancing memory and gradient quality.
- `exp` (default: 1): L1 loss for robustness to outliers in turbulent features.
- `training_samples` (default: 256): Number of shear layer trajectories for training.

**Key Insight:** The Navier-Stokes shear layer problem is particularly challenging due to the development of turbulent structures from initially smooth shear profiles. The bandlimit-preserving operations in CNO help avoid aliasing errors that could corrupt the representation of fine-scale vortical structures. The multi-resolution U-Net architecture captures both the large-scale vortex rollup and the small-scale turbulent mixing. The filtering parameters should not be modified as they maintain the CDE property essential for resolution-independent performance.
