# Fourier Neural Operator (FNO) for 2D Darcy Flow

**Keywords**: [PDE, elliptic, linear, forward-problem, darcy, 2D, regular, dirichlet, heterogeneous, FNO, spectral-method, adam, mse, relative-l2, gpu, pytorch]

**Problem:** Fourier Neural Operator (FNO) is a neural operator architecture that learns mappings between infinite-dimensional function spaces by parameterizing integral kernels in Fourier space. Unlike traditional neural networks that discretize on fixed grids, FNO learns solution operators that are resolution-invariant—trained on one resolution but can evaluate on any other resolution without retraining. For the 2D Darcy flow equation modeling subsurface flow through porous media, FNO maps heterogeneous permeability coefficient fields a(x,y) to pressure solution fields u(x,y), where the governing PDE is:
```
-∇·(a(x,y)∇u(x,y)) = f(x,y),  (x,y) ∈ [0,1]², u = 0 on ∂Ω
```

**Issues addressed:**
- **Resolution dependence**: Traditional CNN-based methods trained on specific grid resolutions cannot generalize to different resolutions. FNO learns in Fourier space and achieves resolution-invariant operator learning.
- **Limited receptive fields**: Standard CNNs require very deep architectures to capture global interactions. FNO provides global receptive fields through spectral convolutions in Fourier space.
- **Mesh-dependent methods**: Finite element or finite difference methods require re-solving on each new mesh. FNO is mesh-independent and provides fast surrogate modeling.
- **Computational efficiency**: FFT-based operations in O(N log N) time vs O(N²) for full attention mechanisms or dense layers.
- **Elliptic PDE structure**: For elliptic PDEs like Darcy flow, boundary conditions affect the entire domain—FNO's global receptive field naturally captures this long-range dependence.

## Key Method

The Fourier Neural Operator (FNO) architecture consists of three main components:

### 1. Spectral Convolution in Fourier Space

The core innovation is the **SpectralConv2d** layer that:
- Applies 2D FFT to transform input from physical space to Fourier space
- Multiplies selected low-frequency Fourier modes by learnable complex-valued weights
- Applies inverse 2D FFT to return to physical space
- Truncates high-frequency modes (acts as implicit regularization and compression)

**Key idea**: Instead of learning convolution kernels in physical space, FNO learns linear transformations on Fourier coefficients. This is equivalent to learning integral kernels that are parameterized in Fourier space.

### 2. FNO Architecture

The full FNO2d model follows a **Lift-Apply-Project** structure:

1. **Lifting**: Map input from 3 channels (a(x,y), x, y) to width=32 hidden channels
2. **Fourier Layers**: Stack of 4 layers, each containing:
   - Spectral convolution (global operation in Fourier space)
   - Skip connection via 1×1 convolution (local operation in physical space)
   - ReLU activation (nonlinearity)
3. **Projection**: Map from hidden channels back to 1 output channel (pressure u)

**Dual-path design**: Each layer combines:
- Spectral path: captures global, smooth features via Fourier modes
- Skip path: captures local, sharp features via pointwise convolutions

### 3. Resolution Invariance

FNO achieves discretization-invariant learning:
- Training grid: s×s (e.g., 85×85 or 141×141)
- Test grid: can be any resolution (e.g., 421×421)
- Mechanism: Fourier modes are continuous representations; FFT handles any grid size

This is fundamentally different from CNNs which have fixed receptive fields tied to grid resolution.

## Implementation

### Spectral Convolution Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer performing spectral convolution.

    Key operations:
    1. FFT: Transform to Fourier space
    2. Linear transform: Multiply selected Fourier modes by learnable weights
    3. Inverse FFT: Transform back to physical space

    This implements a continuous convolution in function space.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in x-direction (at most floor(N/2) + 1)
        self.modes2 = modes2  # Number of Fourier modes in y-direction

        # Learnable complex-valued weights for Fourier mode multiplication
        self.scale = (1 / (in_channels * out_channels))
        # weights1: for positive frequencies in x-direction
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                   dtype=torch.cfloat)
        )
        # weights2: for negative frequencies in x-direction (due to real FFT symmetry)
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                   dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space using Einstein summation"""
        # (batch, in_channel, x, y) × (in_channel, out_channel, x, y)
        # -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Step 1: Apply 2D Real FFT
        # Input shape: (batch, channels, height, width)
        # Output shape: (batch, channels, height, width//2 + 1) complex
        x_ft = torch.fft.rfft2(x)

        # Step 2: Multiply relevant Fourier modes with learnable weights
        # Initialize output Fourier coefficients (all zeros)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # Multiply low-frequency modes (top-left corner in frequency domain)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Multiply low-frequency modes (bottom-left corner, negative frequencies in x)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # High-frequency modes are implicitly set to zero (truncation/filtering)

        # Step 3: Apply inverse 2D Real FFT to return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
```

### FNO2d Architecture

```python
class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for Darcy flow.

    Architecture:
    - Lift: 3 channels (a, x, y) -> width channels
    - 4 Fourier layers: spectral conv + skip connection + activation
    - Project: width -> 128 -> 1

    Input: (a(x,y), x, y) permeability and coordinates, shape (batch, s, s, 3)
    Output: u(x,y) pressure field, shape (batch, s, s, 1)
    """
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        self.modes1 = modes1  # Number of Fourier modes in x
        self.modes2 = modes2  # Number of Fourier modes in y
        self.width = width    # Hidden channel dimension

        # Lifting layer: map input to hidden representation
        self.fc0 = nn.Linear(3, self.width)  # (a(x,y), x, y) -> width channels

        # 4 Fourier layers (spectral convolutions)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Skip connections (local operations in physical space)
        # 1×1 convolutions implemented as Conv1d over flattened spatial dimensions
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # Projection layers: map back to output space
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        # Lift: (batch, s, s, 3) -> (batch, s, s, width)
        x = self.fc0(x)
        # Permute to (batch, width, s, s) for convolution operations
        x = x.permute(0, 3, 1, 2)

        # Fourier Layer 1: Spectral conv + Skip + ReLU
        x1 = self.conv0(x)  # Global operation in Fourier space
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)  # Local operation
        x = F.relu(x1 + x2)

        # Fourier Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.relu(x1 + x2)

        # Fourier Layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.relu(x1 + x2)

        # Fourier Layer 4 (no activation on last layer)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        # Project: (batch, width, s, s) -> (batch, s, s, width) -> (batch, s, s, 1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

### Training Configuration

```python
from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss, count_params
import numpy as np

# Data configuration
TRAIN_PATH = 'piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'piececonst_r421_N1024_smooth2.mat'

ntrain = 1000  # Training samples
ntest = 200    # Test samples

# Resolution configuration
train_res = 85  # Can be 29, 85, 141, 211, 421
r = (421 - 1) // (train_res - 1)  # Subsampling rate

# Model hyperparameters
modes = 12      # Number of Fourier modes to learn (out of ~64 available)
width = 32      # Hidden channel dimension

# Training hyperparameters
batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 100  # Learning rate decay step
gamma = 0.5      # Learning rate decay factor

# Load and subsample data
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :train_res, :train_res]
y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :train_res, :train_res]

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :train_res, :train_res]
y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :train_res, :train_res]

# Normalize data
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

# Append spatial coordinates to input
grid = np.linspace(0, 1, 421).reshape(421, 1).astype(np.float64)
grid = grid[::r, :]
grid_mesh = np.vstack([xx.ravel() for xx in np.meshgrid(grid, grid)]).T
grid_mesh = grid_mesh.reshape(1, train_res, train_res, 2)
grid_mesh = torch.tensor(grid_mesh, dtype=torch.float)

# Concatenate: (a(x,y), x, y) at each grid point
x_train = torch.cat([x_train.reshape(ntrain, train_res, train_res, 1),
                     grid_mesh.repeat(ntrain, 1, 1, 1)], dim=3)
x_test = torch.cat([x_test.reshape(ntest, train_res, train_res, 1),
                    grid_mesh.repeat(ntest, 1, 1, 1)], dim=3)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=batch_size, shuffle=False
)

# Initialize model, optimizer, scheduler
model = FNO2d(modes, modes, width).cuda()
print(f"Model parameters: {count_params(model)}")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Loss function
myloss = LpLoss(size_average=False)  # Relative L2 loss
y_normalizer.cuda()

# Training loop
for ep in range(epochs):
    model.train()
    train_l2 = 0
    train_mse = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, train_res, train_res)

        # Backward pass on MSE loss
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        mse.backward()

        # Track relative L2 loss (for monitoring)
        out = y_normalizer.decode(out)
        y_decoded = y_normalizer.decode(y)
        loss = myloss(out.view(batch_size, -1), y_decoded.view(batch_size, -1))

        optimizer.step()
        train_mse += mse.item()
        train_l2 += loss.item()

    scheduler.step()

    # Evaluation
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).reshape(batch_size, train_res, train_res)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    print(f"Epoch {ep}: Train MSE: {train_mse:.3e}, Train L2: {train_l2:.4f}, Test L2: {test_l2:.4f}")
```

## Critical Parameters

### 1. Architecture Configuration
- **Fourier modes**: modes1 = modes2 = 12
  - Out of ~64 available modes (for 85×85 grid) or ~211 (for 421×421 grid)
  - Low-frequency truncation provides implicit regularization
  - Typical range: 8-16 modes for 2D problems
- **Hidden width**: 32 channels
  - Typical range: 20-64 for 2D Darcy flow
- **Number of layers**: 4 Fourier layers
- **Activation**: ReLU (applied after first 3 layers, not after last layer)

### 2. Training Configuration
- **Batch size**: 20
- **Learning rate**: 0.001 (initial)
- **Weight decay**: 1e-4 (L2 regularization)
- **LR scheduler**: StepLR (decay by 0.5 every 100 epochs)
- **Epochs**: 500
- **Loss function**: MSE for backprop, relative L2 for monitoring

### 3. Data Configuration
- **Training samples**: 1000
- **Test samples**: 200
- **Spatial domain**: [0,1]²
- **Boundary conditions**: Dirichlet (u = 0 on ∂Ω)
- **Input**: Piecewise constant permeability field a(x,y) + coordinates (x,y)
- **Output**: Pressure field u(x,y)
- **Normalization**: Unit Gaussian normalization for both input and output

### 4. Resolution Invariance
- **Training resolution**: Can be 29×29, 85×85, 141×141, or 211×211
- **Test resolution**: Can evaluate on any resolution (e.g., 421×421)
- **Subsampling**: From full 421×421 dataset
- **Key advantage**: Train once on coarse grid, evaluate on fine grid without retraining

### 5. Computational Efficiency
- **FFT complexity**: O(N log N) per Fourier layer
- **Parameter count**: ~600K parameters (for modes=12, width=32)
- **Hardware**: CUDA-enabled GPU
- **Speedup**: ~1000× faster than traditional PDE solvers for inference
