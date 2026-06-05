# Fourier-DeepONet for Full Waveform Inversion (CurveVel-A Dataset)

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
# Training loop for CurveVel-A dataset
def main(dataset='cva', task='loc_f'):
    # Load CurveVel-A dataset with varying source frequencies and locations
    X_train, y_train, X_test, y_test = data_cva_train(task=task)

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
- **dataset**: CurveVel-A (CVA) - curved layers with distinct interfaces
