# Fourier Neural Operator (FNO) for 1D Burgers' Equation

**Keywords**: [PDE, parabolic, nonlinear, forward-problem, burgers, 1D, periodic, FNO, CNN, adam, mse, relative-l2, gpu, pytorch]

**Problem:** Learning the solution operator for the 1D viscous Burgers' equation:
```
∂u/∂t + u∂u/∂x = ν∂²u/∂x²,  x ∈ [0,1], periodic BC, ν = 0.1
```
FNO maps initial conditions a(x) from GRF to steady-state u(x) at t=1, on 128-point grid. Spectral convolutions in Fourier space naturally handle periodicity and provide global receptive fields for this nonlinear convection-diffusion problem.

**Issues addressed:**
- Limited receptive fields of standard CNNs requiring very deep architectures
- Mesh-dependent methods that don't generalize across resolutions
- Need for resolution-invariant operator learning
- Computational efficiency via FFT operations (O(N log N))
- Global interactions captured naturally in Fourier domain

## Key Method

FNO for 1D problems uses:

1. **SpectralConv1d**: 1D Fourier layer applying learnable transformations to Fourier modes
   - FFT → multiply low-frequency modes → inverse FFT
2. **FNO1d Architecture**: 4 Fourier layers with skip connections
   - Lifting: (a(x), x) → width channels
   - Fourier layers: spectral convolution + skip connection + ReLU
   - Projection: width → 128 → 1

**Architecture**: 4 SpectralConv1d layers with modes=16 (out of 65 available in 128-point FFT), width=64. Input: (a(x), x) concatenated. 1D FFT naturally captures periodic BC. Resolution-invariant: train on 128, test on 256, 512, etc.

**Key advantage**: Trained on one resolution, generalizes to other resolutions without retraining.

## Implementation

### 1D Spectral Convolution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer implementing spectral convolution.
    Applies learnable linear transformation to Fourier modes.
    """
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep (at most floor(N/2) + 1)

        # Initialize complex-valued weights for Fourier mode multiplication
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        """Complex multiplication in Fourier space"""
        # (batch, in_channel, x) × (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Step 1: Apply 1D Real FFT
        x_ft = torch.fft.rfft(x)

        # Step 2: Multiply first 'modes1' Fourier modes with learnable weights
        # Higher modes are truncated (set to zero)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,
                            device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Step 3: Inverse FFT to return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
```

### FNO1d Architecture

```python
class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator with 4 Fourier layers.

    Input: (a(x), x) initial condition and location, shape (batch, s, 2)
    Output: u(x) steady-state solution, shape (batch, s, 1)
    """
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        self.modes1 = modes  # Number of Fourier modes
        self.width = width   # Hidden channel dimension

        # Lifting: map input (2 channels) to hidden representation
        self.fc0 = nn.Linear(2, self.width)  # (a(x), x) -> width

        # 4 Fourier layers
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        # Skip connections: 1D convolutions
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # Projection layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Lift: (batch, s, 2) -> (batch, s, width) -> (batch, width, s)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # Fourier layer 1
        x1 = self.conv0(x)    # Spectral convolution
        x2 = self.w0(x)       # Skip connection
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer 4 (no activation)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Project: (batch, width, s) -> (batch, s, width) -> (batch, s, 1)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

### Training Setup

```python
from utilities3 import MatReader, LpLoss, count_params
import numpy as np

# Configuration
ntrain = 1000
ntest = 200
s = 128              # Spatial resolution (can be 128, 256, 512, etc.)
sub = 2**13 // s     # Subsampling rate from full 8192 grid

batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 100      # LR decay step
gamma = 0.5          # LR decay factor

modes = 16           # Number of Fourier modes to use
width = 64           # Hidden channel dimension

# Load data
dataloader = MatReader('burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::sub]  # Initial conditions
y_data = dataloader.read_field('u')[:,::sub]  # Steady-state solutions

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

# Append spatial coordinates to input
grid = np.linspace(0, 1, 2**13).reshape(2**13, 1).astype(np.float64)
grid = grid[::sub,:]
grid = torch.tensor(grid, dtype=torch.float)

# Create input: (a(x), x) at each grid point
x_train = torch.cat([x_train.reshape(ntrain, s, 1),
                     grid.repeat(ntrain, 1, 1)], dim=2)
x_test = torch.cat([x_test.reshape(ntest, s, 1),
                    grid.repeat(ntest, 1, 1)], dim=2)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True
)

# Initialize model
model = FNO1d(modes, width).cuda()
print(f"Model parameters: {count_params(model)}")

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Loss function
myloss = LpLoss(size_average=False)

# Training loop
for ep in range(epochs):
    model.train()
    train_l2 = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()
    train_l2 /= ntrain

    print(f"Epoch {ep}: Train L2 error = {train_l2:.4f}")
```

## Critical Parameters

1. **Fourier Layer Configuration**:
   - modes = 16 (number of low-frequency Fourier modes to learn)
   - Truncation: modes out of N/2+1 available modes
   - 4 Fourier layers with skip connections

2. **Network Architecture**:
   - Hidden width: 64 channels
   - Lifting: 2 → 64
   - Projection: 64 → 128 → 1
   - Activation: ReLU

3. **Training Configuration**:
   - Batch size: 20
   - Learning rate: 0.001
   - Weight decay: 1e-4
   - LR schedule: StepLR (step=100, gamma=0.5)
   - Epochs: 500

4. **Data Configuration**:
   - Spatial resolution: 128 points (subsampled from 8192)
   - Training samples: 1000
   - Test samples: 200
   - Domain: [0,1] with periodic BC
   - Viscosity: ν = 0.1 (R=10)

5. **Loss Function**: Lp loss (relative L2 norm)

6. **Hardware**: CUDA-enabled GPU

7. **Resolution Invariance**: Model can be trained on s=128 and evaluated on different resolutions (e.g., s=256, s=512)
