# Fourier-DeepONet for Full Waveform Inversion (CurveFault-A Dataset)

**Keywords**: [PDE, hyperbolic, nonlinear, inverse-problem, acoustic, 2D, wave, forward-problem, DeepONet, Fourier-transform, MLP, CNN, U-Net, adam, l2-regularization, mae, pytorch, deepxde]

**Problem:** Full waveform inversion (FWI) infers subsurface velocity structures from seismic waveform data by solving a non-convex optimization problem governed by the acoustic wave equation. Traditional data-driven FWI methods lack generalization to varying source parameters (frequencies and locations), leading to poor performance when source configurations differ from training data. This method develops Fourier-DeepONet to enable generalization across variable source frequencies (5-25 Hz) and source locations, addressing the critical need for models that can handle diverse field survey conditions without retraining.

**Issues addressed:**
- Poor generalization of CNN-based FWI methods to varying source frequencies and locations
- Sensitivity to Gaussian noise in seismic data (robust up to 10% noise standard deviation)
- Robustness to missing receiver traces (up to 50% missing traces)
- Noise in source wavelets (robust to Gaussian noise with std up to 0.1)
- Blurred subsurface structure predictions from vanilla DeepONet's inner product decoder

## Key Method

Fourier-DeepONet enhances the DeepONet architecture by:

1. **Branch-Trunk Structure with Parameter Space**: Unlike vanilla DeepONet which uses output coordinates as trunk input, Fourier-DeepONet uses source parameters (frequency and/or locations) as trunk inputs. The branch network processes seismic data while the trunk network encodes source parameters.

2. **Fourier-Enhanced Decoder**: Replaces the vanilla DeepONet's inner product merger with a sophisticated decoder combining:
   - One Fourier layer (spectral convolution in frequency domain)
   - Three U-Fourier layers (Fourier Neural Operator + U-Net)
   - This captures both global patterns (via Fourier) and local features (via U-Net)

3. **Architecture Components**:
   - **Branch Net**: Linear layer lifting 5-channel seismic data (from 5 sources) to 64 channels
   - **Trunk Net**: Linear layer encoding source parameters (frequency and/or locations) to 64 channels
   - **Merger**: Pointwise multiplication of branch and trunk outputs
   - **Decoder**: 1 Fourier layer + 3 U-Fourier layers with progressive downsampling
   - **Projection**: Final linear layers mapping to 70x70 velocity maps

4. **Spectral Convolution**: Uses 2D FFT to learn in frequency domain, enabling efficient capture of multi-scale features while maintaining global receptive field.

## Implementation

```python
# Fourier-DeepONet Architecture
class FourierDeepONet(dde.nn.pytorch.NN):
    """
    Fourier-enhanced DeepONet for FWI with variable source parameters.

    Args:
        num_parameter: Number of source parameters (1 for freq OR locations, 6 for both)
        width: Number of channels (default 64)
        modes1, modes2: Number of Fourier modes for spectral convolution
        merge_operation: 'mul' or 'add' for combining branch and trunk outputs
    """
    def __init__(self, num_parameter, width=64, modes1=20, modes2=20,
                 regularization=None, merge_operation="mul"):
        super().__init__()
        self.num_parameter = num_parameter
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2

        # Branch network: processes seismic data (5 sources x 1000 time x 70 receivers)
        self.branch = Branch(self.width)

        # Trunk network: encodes source parameters (frequency and/or locations)
        self.trunk = Trunk(self.width, self.num_parameter)

        # Decoder: Fourier layers + U-Fourier layers for high-quality reconstruction
        self.merger = decoder(self.modes1, self.modes2, self.width)

        self.b = nn.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.merge_operation = merge_operation

    def forward(self, inputs):
        # Branch processes seismic data
        x1 = self.branch(inputs[0])  # Shape: (batch, 64, 72, 1000)

        # Trunk processes source parameters
        x2 = self.trunk(inputs[1])    # Shape: (batch, 64, 1, 1)

        # Merge branch and trunk outputs via pointwise multiplication
        if self.merge_operation == "mul":
            x = torch.mul(x1, x2)
        elif self.merge_operation == "add":
            x = x1 + x2
        x = x + self.b

        # Decoder transforms to velocity map
        x = self.merger(x)           # Shape: (batch, 1, 70, 70)
        return x
```

```python
# Spectral Convolution Layer (2D Fourier Neural Operator)
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer performing:
    1. FFT to transform to frequency domain
    2. Linear transformation on selected Fourier modes
    3. Inverse FFT back to spatial domain
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in dimension 1
        self.modes2 = modes2  # Number of Fourier modes in dimension 2

        self.scale = (1 / (in_channels * out_channels))
        # Learnable weights for Fourier modes (complex-valued)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                   self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                   self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Complex multiplication in frequency domain
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Transform to frequency domain via 2D FFT
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])

        # Multiply relevant Fourier modes with learnable weights
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2),
                            x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Transform back to physical space via inverse FFT
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x
```

```python
# U-Fourier Decoder combining Fourier layers and U-Net
class decoder(nn.Module):
    """
    Decoder with 1 Fourier layer + 3 U-Fourier layers.
    Progressively downsamples from (1000, 72) to (70, 72) and projects to (70, 70).
    """
    def __init__(self, modes1, modes2, width):
        super(decoder, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Fourier layer (global pattern capture)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)

        # U-Fourier layers (combining global and local features)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.unet1 = U_net(self.width, self.width, 3, 0)

        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.unet2 = U_net(self.width, self.width, 3, 0)

        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.unet3 = U_net(self.width, self.width, 3, 0)

        # Downsampling linear layers
        self.linear1 = nn.Linear(1000, 512)  # 1000 -> 512
        self.linear2 = nn.Linear(512, 256)   # 512 -> 256
        self.linear3 = nn.Linear(256, 70)    # 256 -> 70

        # Final projection to velocity map
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        # Fourier layer: (batch, 64, 1000, 72)
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        # U-Fourier layer 1 with downsampling: (batch, 64, 1000, 72) -> (batch, 64, 512, 72)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet1(x)
        x = x1 + x2 + x3
        x = self.linear1(x)  # Downsample time dimension
        x = F.relu(x)

        # U-Fourier layer 2 with downsampling: (batch, 64, 512, 72) -> (batch, 64, 256, 72)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, 512)
        x3 = self.unet2(x)
        x = x1 + x2 + x3
        x = self.linear2(x)  # Downsample time dimension
        x = F.relu(x)

        # U-Fourier layer 3 with downsampling: (batch, 64, 256, 72) -> (batch, 64, 70, 72)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, 256)
        x3 = self.unet3(x)
        x = x1 + x2 + x3
        x = self.linear3(x)  # Downsample to 70
        x = F.relu(x)

        # Project to velocity map: (batch, 64, 70, 72) -> (batch, 1, 70, 70)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(batchsize, 1, size_x, 70)[:, :, 1:-1, :]  # Slice to (70, 70)

        return x
```

```python
# Training loop
def main(dataset, task):
    # Load data for specific dataset and task
    # task: 'f' (frequency), 'loc' (location), or 'loc_f' (both)
    if dataset == 'cfa':
        X_train, y_train, X_test, y_test = data_cfa_train(task=task)
    # ... similar for other datasets

    data = Dataset(X_train, y_train, X_test, y_test)

    # Initialize Fourier-DeepONet
    # num_parameter: 1 for 'f' or 'loc', 6 for 'loc_f' (5 locations + 1 frequency)
    net = FourierDeepONet(
        num_parameter=X_train[1].shape[1],
        width=64,
        modes1=20,
        modes2=20,
        regularization=["l2", 3e-6]
    )
    model = dde.Model(data, net)

    # L1 loss function for training
    def loss_func_L1(y_true, y_pred):
        return torch.nn.L1Loss()(y_pred, y_true)

    # Compile with Adam optimizer, learning rate 1e-3, step decay every 5000 iterations
    model.compile(
        "adam",
        lr=1e-3,
        loss=loss_func_L1,
        decay=("step", 5000, 0.9),
        metrics=[
            lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),  # MAE
            lambda y_true, y_pred: np.sqrt(np.mean(((y_true - y_pred) ** 2)))  # RMSE
        ]
    )

    # Train for 100,000 iterations with batch size 32
    checker = dde.callbacks.ModelCheckpoint(
        f"./model_{dataset}_{task}/model",
        save_better_only=False,
        period=10000
    )
    losshistory, train_state = model.train(
        iterations=100000,
        batch_size=32,
        display_every=100,
        callbacks=[checker]
    )
```

## Critical Parameters

- **width**: 64 channels for branch, trunk, and decoder networks
- **modes1, modes2**: 20 Fourier modes in each spatial dimension for spectral convolution
- **num_parameter**: 1 for frequency-only or location-only tasks, 6 for combined tasks (5 source locations + 1 frequency)
- **learning_rate**: 1e-3 with step decay (factor 0.9 every 5000 iterations)
- **regularization**: L2 regularization with weight 3e-6
- **loss_function**: L1 loss (MAE) for training
- **batch_size**: 32
- **iterations**: 100,000
- **merge_operation**: "mul" (pointwise multiplication of branch and trunk outputs)
- **optimizer**: Adam
- **decoder_layers**: 1 Fourier layer + 3 U-Fourier layers
- **velocity_range**: [1500, 4500] m/s (normalized to [-1, 1])
- **source_frequency_range**: [5, 25] Hz
- **source_location_range**: Variable within [0, 690] m
