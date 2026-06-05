# Convolutional Neural Operator (CNO) for Wave Equation

**Keywords**: PDE, hyperbolic, linear, forward-problem, wave, 2D, periodic, oscillatory, FNO, CNN, U-Net, ResNet, multi-resolution, pytorch, adam, mae, gpu

**Problem:** CNO is a neural operator architecture designed to learn mappings between infinite-dimensional function spaces for PDE solutions. Traditional CNNs suffer from aliasing errors when applied to operator learning because standard convolutional operations do not respect the continuous-discrete equivalence (CDE) property - meaning that discretizing a continuous operator and then applying it does not yield the same result as applying the operator continuously and then discretizing. CNO addresses this by modifying all CNN operations (convolution, upsampling, downsampling, activation) to be bandlimit-preserving, ensuring that the discrete network faithfully represents its continuous counterpart across different resolutions.

**Issues addressed:**
- **Aliasing errors**: Standard CNNs introduce aliasing when learning operators due to improper handling of high-frequency components during upsampling/downsampling and nonlinear activations. CNO eliminates this through careful filtering.
- **Resolution dependence**: CNO maintains consistent performance across different grid resolutions due to its continuous-discrete equivalence property.
- **Function space inconsistency**: CNO ensures that operators work consistently in continuous function spaces by preserving bandlimits throughout all operations.
- **Oscillatory solutions**: Wave equations produce oscillatory solutions with temporal and spatial periodicities. CNO's bandlimit-preserving design helps accurately represent these oscillations without introducing spurious frequencies through aliasing.

## Key Method

CNO implements a modified U-Net architecture where every operation is designed to preserve bandlimited functions. For the wave equation, the operator learns the propagation dynamics of waves with various initial conditions and frequencies.

The key innovations are:

1. **Filtered Activation Layers**: Upsample → activate → downsample process ensures bandlimit preservation even through nonlinear activations.

2. **Bandlimit-Preserving Convolutions**: Convolutions with careful cutoff frequency design to avoid introducing spurious high-frequency components.

3. **Operator U-Net Structure**: Lift → Encoder → Bottleneck → Decoder → Project architecture with skip connections for multi-scale feature extraction.

4. **Continuous-Discrete Equivalence (CDE)**: The architecture ensures that discretizing the continuous operator and applying it yields the same result as applying the continuous operator and then discretizing, enabling resolution-independent learning.

The wave equation ∂²u/∂t² = c²∇²u is a fundamental hyperbolic PDE describing wave propagation in various physical systems (acoustics, electromagnetics, elasticity). Accurately capturing oscillatory behavior without aliasing is critical for wave problems.

## Implementation

The implementation for the wave equation uses the same CNO architecture as other problems. See cno_allen_cahn entry for complete code details of CNOBlock, ResidualBlock, and CNO class.

### Problem-Specific Configuration

```python
# Training hyperparameters for wave equation
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

# Model architecture for wave equation
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

# Load wave equation problem
from Problems.CNOBenchmarks import WaveEquation
example = WaveEquation(model_architecture, device, batch_size, training_samples)
```

## Critical Parameters

**Architecture Parameters:**
- `N_layers` (default: 3): Number of encoder/decoder levels for multi-scale wave feature extraction.
- `channel_multiplier` (default: 32): Channel growth factor controlling model capacity.
- `N_res` (default: 4): Residual blocks in skip connections for better feature mixing.
- `N_res_neck` (default: 6): Residual blocks in bottleneck processing coarsest-scale features.
- `in_size` (default: 64): Grid resolution (64x64) for wave simulation.

**Filtering Parameters (Critical for CDE - Do Not Modify):**
- `cutoff_den` (default: 2.0001): Bandlimit cutoff frequency for Nyquist-safe operations. Critical for avoiding aliasing in oscillatory wave solutions.
- `lrelu_upsampling` (default: 2): Upsampling factor N_σ in filtered activations ensuring bandlimit preservation.
- `half_width_mult` (default: 0.8): Filter transition bandwidth coefficient c_h affecting filter sharpness.
- `filter_size` (default: 6): Filter tap count = 2*filter_size. Larger values provide better frequency response but are slower.
- `radial_filter` (default: 0): Use separable (0) or radial (1) filters. Separable is computationally faster.

**Training Parameters:**
- `learning_rate` (default: 0.001): AdamW initial learning rate.
- `weight_decay` (default: 1e-6): L2 regularization to prevent overfitting.
- `scheduler_step` (default: 10): Decay learning rate every 10 epochs.
- `scheduler_gamma` (default: 0.98): LR decay multiplier (exponential decay).
- `batch_size` (default: 16): Training batch size balancing memory and gradient quality.
- `exp` (default: 1): L1 loss (exp=1) is more robust than L2 (exp=2) to outliers.
- `training_samples` (default: 256): Number of wave propagation trajectories for training.

**Key Insight:** For the wave equation, the bandlimit-preserving operations are particularly important because waves naturally contain oscillatory content that can easily alias if not handled carefully. The filtering parameters (cutoff_den, lrelu_upsampling, half_width_mult, filter_size) are designed to maintain the CDE property and should not be modified. The architecture parameters (N_layers, channel_multiplier, N_res) can be tuned for different wave propagation problems, but the default values provide a good balance between accuracy and computational cost.
