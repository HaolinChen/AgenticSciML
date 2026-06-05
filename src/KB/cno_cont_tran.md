# Convolutional Neural Operator (CNO) for Smooth Transport Equation

**Keywords**: PDE, hyperbolic, linear, forward-problem, advection, 2D, periodic, FNO, CNN, U-Net, ResNet, multi-resolution, pytorch, adam, mae, gpu

**Problem:** CNO is a neural operator architecture designed to learn mappings between infinite-dimensional function spaces for PDE solutions. Traditional CNNs suffer from aliasing errors when applied to operator learning because standard convolutional operations do not respect the continuous-discrete equivalence (CDE) property - meaning that discretizing a continuous operator and then applying it does not yield the same result as applying the operator continuously and then discretizing. CNO addresses this by modifying all CNN operations (convolution, upsampling, downsampling, activation) to be bandlimit-preserving, ensuring that the discrete network faithfully represents its continuous counterpart across different resolutions.

**Issues addressed:**
- **Aliasing errors**: Standard CNNs introduce aliasing when learning operators due to improper handling of high-frequency components during upsampling/downsampling and nonlinear activations. CNO eliminates this through careful filtering.
- **Resolution dependence**: CNO maintains consistent performance across different grid resolutions due to its continuous-discrete equivalence property.
- **Function space inconsistency**: CNO ensures that operators work consistently in continuous function spaces by preserving bandlimits throughout all operations.
- **Smooth transport dynamics**: The filtered activation layers and multi-resolution U-Net structure enable CNO to accurately capture smooth advection and translation patterns in hyperbolic transport equations.

## Key Method

CNO implements a modified U-Net architecture where every operation is designed to preserve bandlimited functions. The key innovations are:

1. **Filtered Activation Layers**: Instead of applying activation functions directly (which introduce high frequencies), CNO uses a three-step process: upsample → activate → downsample. This ensures the output remains bandlimited to the same frequency range as the input.

2. **Bandlimit-Preserving Convolutions**: Convolutions are performed with careful consideration of the cutoff frequency, using filters designed to avoid aliasing.

3. **Operator U-Net Structure**:
   - **Lift Block**: Projects input to higher-dimensional latent space
   - **Encoder Path**: Sequence of downsampling CNOBlocks that progressively reduce resolution while increasing channels
   - **Bottleneck**: Multiple residual blocks at the coarsest resolution
   - **Decoder Path**: Sequence of upsampling CNOBlocks with skip connections from encoder
   - **Projection Block**: Maps latent representation back to output space

4. **Continuous-Discrete Equivalence (CDE)**: The architecture is designed so that discretizing the continuous operator and applying it yields the same result as applying the continuous operator and then discretizing, enabling resolution-independent learning.

The smooth transport equation (continuous translation) describes advection of a quantity without distortion, making it an ideal benchmark for testing CNO's ability to preserve shape and structure while handling hyperbolic dynamics.

## Implementation

The implementation for smooth transport uses the same CNO architecture as other problems. The core components are CNOBlock, ResidualBlock, and the main CNO class with operator U-Net structure. See the cno_allen_cahn entry for complete implementation details.

### Problem-Specific Configuration

```python
# Training hyperparameters for smooth transport
training_properties = {
    "learning_rate": 0.001,
    "weight_decay": 1e-6,
    "scheduler_step": 10,
    "scheduler_gamma": 0.98,
    "epochs": 1000,
    "batch_size": 16,
    "exp": 1,                    # L1 loss for robustness
    "training_samples": 256
}

# Model architecture for smooth transport
model_architecture = {
    "N_layers": 3,
    "channel_multiplier": 32,
    "N_res": 4,
    "N_res_neck": 6,
    "in_size": 64,               # 64x64 grid resolution
    "retrain": 4,
    "kernel_size": 3,
    "FourierF": 0,
    "activation": 'cno_lrelu',

    # Critical filtering parameters for CDE property
    "cutoff_den": 2.0001,
    "lrelu_upsampling": 2,
    "half_width_mult": 0.8,
    "filter_size": 6,
    "radial_filter": 0
}

# Load smooth transport problem
from Problems.CNOBenchmarks import ContTranslation
example = ContTranslation(model_architecture, device, batch_size, training_samples)
model = example.model
train_loader = example.train_loader
val_loader = example.val_loader
```

### Training Loop

```python
# Standard training loop for CNO on smooth transport
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
loss = torch.nn.L1Loss()  # L1 loss

for epoch in range(epochs):
    model.train()
    for input_batch, output_batch in train_loader:
        optimizer.zero_grad()
        output_pred = model(input_batch)

        # Relative loss normalized by target magnitude
        loss_value = loss(output_pred, output_batch) / \
                    loss(torch.zeros_like(output_batch), output_batch)

        loss_value.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for input_batch, output_batch in val_loader:
            output_pred = model(input_batch)
            # Compute relative L1 error in percentage
            rel_error = torch.mean(abs(output_pred - output_batch)) / \
                       torch.mean(abs(output_batch)) * 100

    scheduler.step()
```

## Critical Parameters

**Architecture Parameters:**
- `N_layers` (default: 3): Number of encoder/decoder levels. Controls spatial hierarchy.
- `channel_multiplier` (default: 32): Channel growth factor at each level. Controls model capacity.
- `N_res` (default: 4): Residual blocks in skip connections for better feature mixing.
- `N_res_neck` (default: 6): Residual blocks in bottleneck for coarsest-scale processing.
- `in_size` (default: 64): Grid resolution (must be power of 2). Smooth transport uses 64x64.

**Filtering Parameters (Critical for CDE property):**
- `cutoff_den` (default: 2.0001): Bandlimit cutoff frequency. Values near 2 ensure Nyquist-safe operations. **DO NOT change** unless you understand aliasing theory.
- `lrelu_upsampling` (default: 2): Upsampling factor in filtered activations. Ensures bandlimit preservation.
- `half_width_mult` (default: 0.8): Filter transition bandwidth. Affects filter sharpness.
- `filter_size` (default: 6): Filter tap count = 2*filter_size. Larger = better frequency response but slower.
- `radial_filter` (default: 0): Separable (0) vs radial (1) filters. Separable is faster.

**Training Parameters:**
- `learning_rate` (default: 0.001): Initial LR for AdamW optimizer.
- `weight_decay` (default: 1e-6): L2 regularization to prevent overfitting.
- `scheduler_step` (default: 10): Decay LR every 10 epochs.
- `scheduler_gamma` (default: 0.98): LR decay multiplier (exponential decay).
- `batch_size` (default: 16): Balance between memory and gradient quality.
- `exp` (default: 1): L1 (exp=1) or L2 (exp=2) loss. L1 is more robust.
- `training_samples` (default: 256): Number of smooth transport trajectories for training.

**Key Insight:** The filtering parameters (cutoff_den, lrelu_upsampling, half_width_mult, filter_size) maintain the CDE property and should generally not be modified. The architecture parameters (N_layers, channel_multiplier, N_res) can be tuned for different problems. For smooth transport, the default parameters work well for capturing continuous translation dynamics.
