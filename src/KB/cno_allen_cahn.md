# Convolutional Neural Operator (CNO) for Allen-Cahn Equation

**Keywords**: PDE, parabolic, nonlinear, forward-problem, allen-cahn, 2D, periodic, FNO, CNN, U-Net, ResNet, multi-resolution, pytorch, adam, mae, gpu

**Problem:** CNO is a neural operator architecture designed to learn mappings between infinite-dimensional function spaces for PDE solutions. Traditional CNNs suffer from aliasing errors when applied to operator learning because standard convolutional operations do not respect the continuous-discrete equivalence (CDE) property - meaning that discretizing a continuous operator and then applying it does not yield the same result as applying the operator continuously and then discretizing. CNO addresses this by modifying all CNN operations (convolution, upsampling, downsampling, activation) to be bandlimit-preserving, ensuring that the discrete network faithfully represents its continuous counterpart across different resolutions.

**Issues addressed:**
- **Aliasing errors**: Standard CNNs introduce aliasing when learning operators due to improper handling of high-frequency components during upsampling/downsampling and nonlinear activations. CNO eliminates this through careful filtering.
- **Resolution dependence**: CNO maintains consistent performance across different grid resolutions due to its continuous-discrete equivalence property.
- **Function space inconsistency**: CNO ensures that operators work consistently in continuous function spaces by preserving bandlimits throughout all operations.
- **Sharp transitions and discontinuities**: The filtered activation layers and multi-resolution U-Net structure help CNO handle problems with sharp gradients like phase transitions in Allen-Cahn equations.

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

The Allen-Cahn equation is a parabolic PDE describing phase separation with sharp interfaces, making it an ideal benchmark for testing CNO's ability to handle problems with steep gradients and metastable dynamics.

## Implementation

### Core CNO Block

```python
class CNOBlock(nn.Module):
    """
    Basic building block of CNO architecture.
    Performs: Convolution → Batch Norm → Filtered Activation

    Can operate in three modes:
    - Downsampling (D): Reduces resolution by factor of 2
    - Upsampling (U): Increases resolution by factor of 2
    - Invariant (I): Maintains resolution
    """
    def __init__(self, in_channels, out_channels, in_size, out_size,
                 cutoff_den=2.0001,      # Cutoff frequency denominator
                 conv_kernel=3,          # Convolution kernel size
                 filter_size=6,          # Filter tap size (actual taps = 2*filter_size)
                 lrelu_upsampling=2,     # Upsampling factor for activation (N_σ)
                 half_width_mult=0.8,    # Half-width multiplier (c_h)
                 radial=False,           # Use radially symmetric filter?
                 batch_norm=True,
                 activation='cno_lrelu'):
        super(CNOBlock, self).__init__()

        self.batch_norm_ = batch_norm
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel,
                             padding=(conv_kernel - 1) // 2)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        # Determine operation type and cutoff frequencies
        if (out_size == in_size):
            # Invariant block - no resolution change
            cutoff_den_act = cutoff_den
            cutoff_num_act = 1.0
        elif (out_size < in_size):
            # Downsampling block
            cutoff_den_act = cutoff_den
            cutoff_num_act = out_size / in_size
        else:
            # Upsampling block
            cutoff_den_act = cutoff_den * out_size / in_size
            cutoff_num_act = 1.0

        # Create filtered activation layer
        # This is the key innovation: upsample → activate → filter → downsample
        self.activation_func = FilteredLRelu(out_size, out_channels,
                                            cutoff_num_act, cutoff_den_act,
                                            filter_size, lrelu_upsampling,
                                            half_width_mult, radial, activation)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm_:
            x = self.batch_norm(x)
        x = self.activation_func(x)
        return x
```

### Residual Block

```python
class ResidualBlock(nn.Module):
    """
    Residual block with skip connection: x + F(x)
    Used in both bottleneck and skip connections between encoder/decoder
    """
    def __init__(self, channels, in_size, kernel_size, cutoff_den,
                 filter_size, lrelu_upsampling, half_width_mult,
                 radial, batch_norm, activation):
        super(ResidualBlock, self).__init__()

        self.batch_norm_ = batch_norm
        self.convolution1 = nn.Conv2d(channels, channels, kernel_size,
                                     padding=(kernel_size - 1) // 2)
        self.convolution2 = nn.Conv2d(channels, channels, kernel_size,
                                     padding=(kernel_size - 1) // 2)

        if batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(channels)
            self.batch_norm2 = nn.BatchNorm2d(channels)

        # Filtered activation for resolution-invariant nonlinearity
        self.activation = FilteredLRelu(in_size, channels, 1, cutoff_den,
                                       filter_size, lrelu_upsampling,
                                       half_width_mult, radial, activation)

    def forward(self, x):
        out = self.convolution1(x)
        if self.batch_norm_:
            out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.convolution2(out)
        if self.batch_norm_:
            out = self.batch_norm2(out)
        return x + out  # Skip connection
```

### Main CNO Architecture

```python
class CNO(nn.Module):
    """
    Complete Convolutional Neural Operator architecture.
    Implements an operator U-Net: Lift → Encoder → Bottleneck → Decoder → Project
    """
    def __init__(self, in_dim, in_size, N_layers, N_res=4, N_res_neck=6,
                 channel_multiplier=32, conv_kernel=3, cutoff_den=2.0001,
                 filter_size=6, lrelu_upsampling=2, half_width_mult=0.8,
                 radial=False, batch_norm=True, out_dim=1, out_size=None,
                 latent_lift_proj_dim=64, add_inv=True, activation='cno_lrelu'):
        super(CNO, self).__init__()

        if out_size is None:
            out_size = in_size

        # LIFT: Project input to latent space
        self.lift_block = LiftProjectBlock(in_dim, latent_lift_proj_dim, in_size,
                                          conv_kernel, cutoff_den, filter_size,
                                          lrelu_upsampling, half_width_mult,
                                          radial, batch_norm, activation, 'lift')

        # ENCODER: Downsampling path
        self.encoder_blocks = nn.ModuleList()
        for i in range(N_layers):
            # Resolution decreases by factor of 2 at each layer
            in_size_layer = in_size // (2**i)
            out_size_layer = in_size // (2**(i+1))
            in_ch = latent_lift_proj_dim * (channel_multiplier**i)
            out_ch = latent_lift_proj_dim * (channel_multiplier**(i+1))

            # Downsampling CNOBlock
            self.encoder_blocks.append(
                CNOBlock(in_ch, out_ch, in_size_layer, out_size_layer,
                        cutoff_den, conv_kernel, filter_size, lrelu_upsampling,
                        half_width_mult, radial, batch_norm, activation)
            )

        # BOTTLENECK: Process at coarsest resolution with residual blocks
        bottleneck_size = in_size // (2**N_layers)
        bottleneck_ch = latent_lift_proj_dim * (channel_multiplier**N_layers)
        self.bottleneck = nn.ModuleList([
            ResidualBlock(bottleneck_ch, bottleneck_size, conv_kernel, cutoff_den,
                         filter_size, lrelu_upsampling, half_width_mult,
                         radial, batch_norm, activation)
            for _ in range(N_res_neck)
        ])

        # DECODER: Upsampling path with skip connections
        self.decoder_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()  # Residual connections from encoder

        for i in range(N_layers):
            # Resolution increases by factor of 2 at each layer
            level = N_layers - 1 - i
            in_size_layer = in_size // (2**(level+1))
            out_size_layer = in_size // (2**level)
            in_ch = latent_lift_proj_dim * (channel_multiplier**(level+1))
            out_ch = latent_lift_proj_dim * (channel_multiplier**level)

            # Upsampling CNOBlock
            self.decoder_blocks.append(
                CNOBlock(in_ch, out_ch, in_size_layer, out_size_layer,
                        cutoff_den, conv_kernel, filter_size, lrelu_upsampling,
                        half_width_mult, radial, batch_norm, activation)
            )

            # Residual blocks for skip connections
            self.skip_connections.append(nn.ModuleList([
                ResidualBlock(out_ch, out_size_layer, conv_kernel, cutoff_den,
                            filter_size, lrelu_upsampling, half_width_mult,
                            radial, batch_norm, activation)
                for _ in range(N_res)
            ]))

        # Optional invariant block at original resolution
        if add_inv:
            self.invariant_block = CNOBlock(latent_lift_proj_dim, latent_lift_proj_dim,
                                           in_size, in_size, cutoff_den, conv_kernel,
                                           filter_size, lrelu_upsampling, half_width_mult,
                                           radial, batch_norm, activation)

        # PROJECT: Map latent space back to output
        self.project_block = LiftProjectBlock(latent_lift_proj_dim, out_dim, out_size,
                                             conv_kernel, cutoff_den, filter_size,
                                             lrelu_upsampling, half_width_mult,
                                             radial, batch_norm, activation, 'project')

    def forward(self, x):
        # Lift input to latent space
        x = self.lift_block(x)

        # Encoder path - save intermediate features for skip connections
        encoder_outputs = []
        for encoder_block in self.encoder_blocks:
            encoder_outputs.append(x)
            x = encoder_block(x)

        # Bottleneck processing
        for bottleneck_block in self.bottleneck:
            x = bottleneck_block(x)

        # Decoder path - add skip connections from encoder
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x)
            # Apply residual connection from corresponding encoder layer
            enc_out = encoder_outputs[-(i+1)]
            for skip_block in self.skip_connections[i]:
                enc_out = skip_block(enc_out)
            x = x + enc_out  # Combine decoder output with encoder feature

        # Optional invariant processing at original resolution
        if hasattr(self, 'invariant_block'):
            x = self.invariant_block(x)

        # Project back to output space
        x = self.project_block(x)
        return x
```

### Training Configuration

```python
# Training hyperparameters
training_properties = {
    "learning_rate": 0.001,      # Initial learning rate
    "weight_decay": 1e-6,        # L2 regularization
    "scheduler_step": 10,        # LR scheduler step size
    "scheduler_gamma": 0.98,     # LR decay factor
    "epochs": 1000,              # Maximum epochs
    "batch_size": 16,            # Batch size
    "exp": 1,                    # 1 for L1 loss, 2 for L2 loss
    "training_samples": 256      # Number of training samples
}

# Model architecture parameters
model_architecture = {
    # Core architecture parameters (tune these)
    "N_layers": 3,               # Number of encoder/decoder levels
    "channel_multiplier": 32,    # Channel growth factor (d_e)
    "N_res": 4,                  # Residual blocks in skip connections
    "N_res_neck": 6,             # Residual blocks in bottleneck

    # Grid and domain parameters
    "in_size": 64,               # Input grid resolution (64x64)
    "retrain": 4,                # Random seed

    # Convolution parameters
    "kernel_size": 3,            # Convolution kernel size
    "FourierF": 0,               # Fourier features (0 = none)
    "activation": 'cno_lrelu',   # Activation type

    # Critical filtering parameters (preserve these for CDE property)
    "cutoff_den": 2.0001,        # Cutoff frequency denominator
    "lrelu_upsampling": 2,       # Activation upsampling factor (N_σ)
    "half_width_mult": 0.8,      # Filter half-width multiplier (c_h)
    "filter_size": 6,            # Filter tap count = 2 * filter_size
    "radial_filter": 0,          # 0 = separable, 1 = radial filter
}

# Optimizer setup
optimizer = torch.optim.AdamW(model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=scheduler_step,
                                           gamma=scheduler_gamma)

# Loss function (L1 for robustness to outliers)
loss = torch.nn.L1Loss()

# Training loop with relative error monitoring
for epoch in range(epochs):
    # Training phase
    model.train()
    for input_batch, output_batch in train_loader:
        optimizer.zero_grad()
        output_pred = model(input_batch)

        # Relative loss: normalize by target magnitude
        loss_value = loss(output_pred, output_batch) / \
                    loss(torch.zeros_like(output_batch), output_batch)

        loss_value.backward()
        optimizer.step()

    # Validation phase - compute relative L1 error
    model.eval()
    with torch.no_grad():
        for input_batch, output_batch in val_loader:
            output_pred = model(input_batch)
            # Relative error in percentage
            rel_error = torch.mean(abs(output_pred - output_batch)) / \
                       torch.mean(abs(output_batch)) * 100

    scheduler.step()

    # Early stopping based on validation error
    if val_error < best_error:
        torch.save(model, 'model.pkl')
```

## Critical Parameters

**Architecture Parameters:**
- `N_layers` (default: 3): Number of encoder/decoder levels in U-Net. More layers = more spatial hierarchy but higher memory.
- `channel_multiplier` (default: 32): Factor by which channels grow at each level. Controls model capacity.
- `N_res` (default: 4): Residual blocks in skip connections. More blocks = better feature mixing.
- `N_res_neck` (default: 6): Residual blocks in bottleneck. Processes coarsest-scale features.
- `in_size` (default: 64): Grid resolution. Must be power of 2 for clean downsampling. Allen-Cahn uses 64x64.

**Filtering Parameters (Critical for CDE property):**
- `cutoff_den` (default: 2.0001): Controls the bandlimit cutoff frequency. Values near 2 ensure Nyquist-safe operations. DO NOT change unless you understand aliasing theory.
- `lrelu_upsampling` (default: 2): Upsampling factor N_σ in filtered activations. Factor of 2 ensures bandlimit preservation.
- `half_width_mult` (default: 0.8): Filter transition bandwidth coefficient c_h. Affects filter sharpness.
- `filter_size` (default: 6): Number of filter taps = 2*filter_size. Larger = better frequency response but slower.
- `radial_filter` (default: 0): Use separable (0) or radial (1) filters. Separable is faster, radial is more isotropic.

**Training Parameters:**
- `learning_rate` (default: 0.001): Initial LR for AdamW. Standard value works well.
- `weight_decay` (default: 1e-6): L2 regularization to prevent overfitting.
- `scheduler_step` (default: 10): Decay LR every 10 epochs.
- `scheduler_gamma` (default: 0.98): LR multiplier (exponential decay).
- `batch_size` (default: 16): Balance between memory and gradient quality.
- `exp` (default: 1): Use L1 (exp=1) or L2 (exp=2) loss. L1 is more robust to outliers.
- `training_samples` (default: 256): Number of Allen-Cahn trajectories for training.

**Key Insight:** The filtering parameters (cutoff_den, lrelu_upsampling, half_width_mult, filter_size) are carefully designed to maintain the CDE property. Changing these can break the theoretical guarantees of resolution independence. The architecture parameters (N_layers, channel_multiplier, N_res) can be tuned for different problems, but the filtering parameters should generally be kept at their default values.
