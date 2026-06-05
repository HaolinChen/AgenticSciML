# Convolutional Neural Operator (CNO) for Compressible Euler Equations (Airfoil)

**Keywords**: PDE, hyperbolic, nonlinear, forward-problem, euler, 2D, irregular, FNO, CNN, U-Net, ResNet, multi-resolution, pytorch, adam, mae, gpu

**Problem:** CNO is a neural operator architecture designed to learn mappings between infinite-dimensional function spaces for PDE solutions. Traditional CNNs suffer from aliasing errors when applied to operator learning because standard convolutional operations do not respect the continuous-discrete equivalence (CDE) property - meaning that discretizing a continuous operator and then applying it does not yield the same result as applying the operator continuously and then discretizing. CNO addresses this by modifying all CNN operations (convolution, upsampling, downsampling, activation) to be bandlimit-preserving, ensuring that the discrete network faithfully represents its continuous counterpart across different resolutions.

**Issues addressed:**
- **Aliasing errors**: Standard CNNs introduce aliasing when learning operators due to improper handling of high-frequency components during upsampling/downsampling and nonlinear activations. CNO eliminates this through careful filtering.
- **Resolution dependence**: CNO maintains consistent performance across different grid resolutions due to its continuous-discrete equivalence property.
- **Function space inconsistency**: CNO ensures that operators work consistently in continuous function spaces by preserving bandlimits throughout all operations.
- **Complex geometries**: The airfoil problem involves flow around irregular shapes, requiring the network to handle masked regions representing solid boundaries. CNO's architecture can accommodate such geometric constraints.

## Key Method

CNO implements a modified U-Net architecture where every operation is designed to preserve bandlimited functions. For the airfoil problem, the operator learns to predict flow fields (density, velocity, pressure) around airfoils with varying shapes and flow conditions, governed by the compressible Euler equations.

The key innovations are:

1. **Filtered Activation Layers**: Upsample → activate → downsample process ensures bandlimit preservation.

2. **Bandlimit-Preserving Convolutions**: Convolutions with careful cutoff frequency design.

3. **Operator U-Net Structure**: Lift → Encoder → Bottleneck → Decoder → Project with skip connections.

4. **Continuous-Discrete Equivalence (CDE)**: Resolution-independent operator learning.

5. **Geometry Masking**: For airfoil problems, the solid airfoil shape is masked during training and prediction to enforce boundary conditions.

The compressible Euler equations are nonlinear hyperbolic PDEs modeling inviscid compressible flow. Airfoil flows present challenges including shock waves, complex pressure distributions, and irregular geometries.

## Implementation

The implementation for airfoil uses the same CNO architecture with special handling for the airfoil geometry. See cno_allen_cahn entry for complete code details of CNOBlock, ResidualBlock, and CNO class.

### Problem-Specific Configuration

```python
# Training hyperparameters for airfoil
training_properties = {
    "learning_rate": 0.001,
    "weight_decay": 1e-6,
    "scheduler_step": 10,
    "scheduler_gamma": 0.98,
    "epochs": 1000,
    "batch_size": 16,
    "exp": 1,
    "training_samples": 256
}

# Model architecture for airfoil (higher resolution)
model_architecture = {
    "N_layers": 3,
    "channel_multiplier": 32,
    "N_res": 4,
    "N_res_neck": 6,
    "in_size": 128,              # 128x128 grid (higher than other problems)
    "kernel_size": 3,
    "activation": 'cno_lrelu',

    # Critical filtering parameters for CDE
    "cutoff_den": 2.0001,
    "lrelu_upsampling": 2,
    "half_width_mult": 0.8,
    "filter_size": 6,
    "radial_filter": 0
}

# Load airfoil problem
from Problems.CNOBenchmarks import Airfoil
model_architecture["in_size"] = 128  # Airfoil requires higher resolution
example = Airfoil(model_architecture, device, batch_size, training_samples)
```

### Airfoil Masking in Training

```python
# During training, mask the airfoil shape
for input_batch, output_batch in train_loader:
    output_pred_batch = model(input_batch)

    # Mask the airfoil shape (where input_batch == 1 represents solid)
    output_pred_batch[input_batch == 1] = 1
    output_batch[input_batch == 1] = 1

    loss_value = loss(output_pred_batch, output_batch) / \
                loss(torch.zeros_like(output_batch), output_batch)
    loss_value.backward()
```

## Critical Parameters

**Architecture Parameters:**
- `N_layers` (default: 3): Encoder/decoder depth for multi-scale features.
- `channel_multiplier` (default: 32): Channel growth factor.
- `in_size` (default: 128): **Higher resolution (128x128) compared to other problems (64x64)** to capture airfoil geometry and flow details.

**Filtering Parameters (Do Not Modify):**
- `cutoff_den` (default: 2.0001): Bandlimit cutoff for Nyquist-safe operations.
- `lrelu_upsampling` (default: 2): Ensures bandlimit preservation through activations.
- `half_width_mult` (default: 0.8): Filter transition width.
- `filter_size` (default: 6): Filter quality (taps = 2*filter_size).

**Training Parameters:**
- `learning_rate` (default: 0.001): AdamW initial learning rate.
- `batch_size` (default: 16): Training batch size.
- `training_samples` (default: 256): Number of airfoil-flow pairs for training.

**Key Insight:** The airfoil problem uses 128x128 resolution (compared to 64x64 for other problems) to better capture the geometry and flow features around the airfoil. The masking approach enforces boundary conditions at the airfoil surface by setting predicted values to a constant within the solid region. The bandlimit-preserving operations help maintain accuracy despite the irregular geometry.
