# Fourier Neural Operator (FNO) for 2D Advection Equations (Initial Conditions II & III)

**Keywords**: [PDE, hyperbolic, linear, forward-problem, advection, 2D, FNO, CNN, adam, mse, relative-l2, gpu, pytorch]

**Problem:** Learning the solution operator for the 1D linear advection equation:
```
∂u/∂t + ∂u/∂x = 0,  x ∈ [0,1], t ∈ [0,1], periodic BC
```
FNO maps initial conditions u₀(x) to solutions u(x,t) on a 40×40 grid using spectral convolutions in Fourier space. Unlike DeepONet, FNO processes entire fields through global Fourier transforms, naturally capturing periodicity and long-range dependencies.

**Issues addressed:**
- Standard CNNs have limited receptive fields requiring very deep networks for global patterns
- Difficulty capturing long-range spatial dependencies in PDE solutions
- Mesh-dependent traditional numerical methods requiring re-discretization for different resolutions
- Need for resolution-invariant operator learning (mesh-independent)
- Computational efficiency for operator learning on regular grids

## Key Method

Fourier Neural Operator (FNO) learns operators between function spaces by:

1. **Lifting**: Map input to higher-dimensional representation via point-wise transformation
2. **Fourier Layers**: Apply spectral convolutions in Fourier space
   - Transform to Fourier domain via FFT
   - Apply learnable linear transformation to low-frequency Fourier modes
   - Transform back to physical space via inverse FFT
   - Add skip connection with 1×1 convolution
3. **Projection**: Map from representation space to output via point-wise transformation

**Architecture**: 4 Fourier layers with SpectralConv2d (modes1=16, modes2=16 out of 40 available), width=64 channels. Input: (u₀, x, t) at each grid point. Each layer: FFT → multiply 16×16 low-frequency modes → IFFT + skip. Resolution-invariant: trained on 40×40, generalizes to different resolutions.

**Key Innovation**: Operations in Fourier space are:
- **Global**: Each Fourier mode interacts with all spatial points
- **Efficient**: FFT provides O(N log N) complexity
- **Resolution-invariant**: Trained model generalizes to different grid resolutions

**Architecture**:
```
Input (u₀, x, t) → Lift → [Fourier Layer + Skip] × 4 → Project → Output u(x,t)
```

Each Fourier layer computes:
```
v_{l+1}(x) = σ(W·v_l(x) + K(v_l)(x))
```
where K is the integral kernel operator implemented via Fourier transform.

**Data Format**:
- Input: (u₀, x, t) at each grid point, shape (N, nt, nx, 3)
- Output: u(x,t) solution values, shape (N, nt, nx)

## Implementation

### Spectral Convolution Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer implementing spectral convolution.
    Applies learnable linear transformation in Fourier space.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in first dimension
        self.modes2 = modes2  # Number of Fourier modes in second dimension

        # Initialize learnable weights for Fourier modes
        # weights are complex-valued
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                   self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                   self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space"""
        # (batch, in_channel, x, y) × (in_channel, out_channel, x, y)
        # -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Step 1: Apply 2D Real FFT to transform to Fourier space
        x_ft = torch.fft.rfft2(x)

        # Step 2: Multiply relevant Fourier modes with learnable weights
        # Initialize output Fourier coefficients
        out_ft = torch.zeros(batchsize, self.out_channels,
                            x.size(-2), x.size(-1)//2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # Multiply lower Fourier modes (low frequencies)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Multiply higher Fourier modes (preserving symmetry)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Step 3: Transform back to physical space via inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
```

### FNO Architecture

```python
class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator with 4 Fourier layers.

    Input: (u₀(x), x, t) at each grid point, shape (batch, nt, nx, 3)
    Output: u(x,t), shape (batch, nt, nx, 1)
    """
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1  # Number of Fourier modes to use (dimension 1)
        self.modes2 = modes2  # Number of Fourier modes to use (dimension 2)
        self.width = width    # Hidden channel dimension

        # Lifting layer: map input (3 channels) to hidden dimension
        self.fc0 = nn.Linear(3, self.width)  # Input: (u₀, x, t)

        # 4 Fourier layers with skip connections
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Skip connections: 1×1 convolutions
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # Projection layers: map from hidden dimension to output (1 channel)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        # Lift: (batch, nt, nx, 3) -> (batch, nt, nx, width)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # -> (batch, width, nt, nx)

        # Fourier layer 1 with skip connection
        x1 = self.conv0(x)  # Spectral convolution
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2  # Add skip connection
        x = F.relu(x)

        # Fourier layer 2 with skip connection
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer 3 with skip connection
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer 4 with skip connection (no activation)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        # Project: (batch, width, nt, nx) -> (batch, nt, nx, 1)
        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc2(x)
        return x
```

### Training Loop

```python
from utilities3 import LpLoss, UnitGaussianNormalizer
import numpy as np

# Hyperparameters
batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 100  # Learning rate decay step
gamma = 0.5      # Learning rate decay factor

modes1 = 16   # Fourier modes in temporal dimension (at most nt)
modes2 = 16   # Fourier modes in spatial dimension (at most nx/2+1)
width = 64    # Hidden channel dimension

# Load data
nt, nx = 40, 40
ntrain, ntest = 1000, 1000

data = np.load('train_IC1.npz')
x, t, u_train = data["x"], data["t"], data["u"]  # u_train: (N, nt, nx)
u0_train = u_train[:, 0, :]  # Initial conditions: (N, nx)

# Normalize data
x_normalizer = UnitGaussianNormalizer(torch.from_numpy(u0_train))
u0_train = x_normalizer.encode(torch.from_numpy(u0_train)).numpy()
y_normalizer = UnitGaussianNormalizer(torch.from_numpy(u_train))
u_train = y_normalizer.encode(torch.from_numpy(u_train)).numpy()

# Prepare input: concatenate (u₀, x, t) at each grid point
x = np.repeat(x[0:1, :], ntrain, axis=0)  # Replicate x coordinates
x_train = np.concatenate((u0_train[:, :, None], x[:, :, None]), axis=-1)  # (N, nx, 2)
x_train = np.repeat(x_train[:, None, :, :], nt, axis=1)  # (N, nt, nx, 2)

# Add time channel
t = np.repeat(t.reshape(1, nt, nx), ntrain, axis=0)  # (N, nt, nx)
x_train = np.concatenate((x_train, t[:, :, :, None]), axis=-1)  # (N, nt, nx, 3)

x_train = torch.from_numpy(x_train)
u_train = torch.from_numpy(u_train)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, u_train),
    batch_size=batch_size, shuffle=True
)

# Similarly prepare test data...

# Initialize model
model = FNO2d(modes1, modes2, width).cuda()

# Optimizer with learning rate decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Loss function: Lp loss (relative L2 norm)
myloss = LpLoss(size_average=False)
y_normalizer.cuda()

# Training loop
for ep in range(epochs):
    model.train()
    train_l2 = 0
    train_mse = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()

        # Forward pass
        out = model(x).reshape(batch_size, nt, nx)

        # Denormalize for loss computation
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        # Compute MSE loss and backpropagate
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        mse.backward()

        # Track relative L2 error
        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))

        optimizer.step()
        train_mse += mse.item()
        train_l2 += loss.item()

    scheduler.step()

    # Evaluation on test set
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).reshape(batch_size, nt, nx)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    print(f"Epoch: {ep}, Train MSE: {train_mse:.3e}, Train l2: {train_l2:.4f}, Test l2: {test_l2:.4f}")
```

## Critical Parameters

1. **Fourier Layer Configuration**:
   - modes1 = 16 (temporal Fourier modes, at most nt = 40)
   - modes2 = 16 (spatial Fourier modes, at most nx/2 + 1 = 21)
   - Number of Fourier layers: 4
   - Truncation: only low-frequency modes are learned, high-frequency modes ignored

2. **Network Width**:
   - Hidden channel dimension: 64
   - Limited by GPU memory (max 190 on RTX 2080 Ti)
   - Input channels: 3 (u₀, x, t)
   - Output channels: 1 (u)

3. **Projection Layers**:
   - Lifting: 3 → 64
   - Projection: 64 → 128 → 128 → 128 → 1
   - All projection layers use ReLU activation except final layer

4. **Training Configuration**:
   - Batch size: 20
   - Optimizer: Adam
   - Initial learning rate: 0.001
   - Learning rate schedule: StepLR with step_size=100, gamma=0.5
   - Epochs: 500
   - Loss function: MSE for backpropagation, L2 relative error for evaluation

5. **Data Configuration**:
   - Grid resolution: 40×40 (nt × nx)
   - Training samples: 1000
   - Test samples: 1000
   - Data normalization: UnitGaussianNormalizer (zero mean, unit variance)

6. **Hardware**: CUDA-enabled GPU (tested on RTX 2080 Ti)

7. **Loss Functions**:
   - Training: MSE (mean squared error)
   - Evaluation: Lp loss (relative L2 norm)
