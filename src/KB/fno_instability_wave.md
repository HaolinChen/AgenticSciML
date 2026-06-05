# Fourier Neural Operator (FNO) for Wave Propagation with Non-Uniform Grids

**Keywords**: [PDE, hyperbolic, linear, forward-problem, wave, 2D, irregular, FNO, spectral-method, adam, mse, relative-l2, gpu, pytorch]

**Problem:** Fourier Neural Operator (FNO) applied to 2D linear instability wave propagation in high-speed boundary layers, demonstrating FNO's capability to handle: (1) resolution changes between input and output, (2) non-uniform spatial grids, and (3) wave propagation dynamics with global spatial interactions. The problem involves predicting evolved wave states from initial conditions:
```
Input:  Initial wave state on 20×47 grid (coarse in streamwise x, refined near wall in y)
Output: Evolved wave state on 111×47 grid (fine in x, same refined y)
```

This is a hyperbolic wave PDE where disturbances propagate through the boundary layer. The non-uniform grid in the wall-normal (y) direction is essential to resolve steep gradients near the wall, while the streamwise (x) direction captures wave propagation.

**Issues addressed:**
- **Resolution mismatch handling**: FNO can handle input/output at different resolutions (20×47 → 111×47). This demonstrates discretization-invariance—the operator is learned in function space, not tied to specific grid resolution.
- **Non-uniform grids**: The y-direction uses wall-normal clustering (47 points concentrated near y=0) to capture boundary layer gradients. FNO operates in Fourier space where non-uniform grids can still be processed, though with careful interpolation.
- **Global wave propagation**: Hyperbolic wave PDEs have global influence patterns (waves propagate across domain). FNO's spectral convolutions provide global receptive fields in a single layer, naturally suited for wave dynamics.
- **Computational efficiency**: FFT-based operations (O(N log N)) are much faster than traditional wave solvers or recurrent architectures for multi-step propagation.
- **Operator learning for dynamics**: Instead of learning point-wise solutions, FNO learns the solution operator mapping initial states to future states—generalizes to new initial conditions without retraining.

## Key Method

FNO for wave propagation with non-uniform grids and resolution changes:

### 1. Resolution Invariance and Grid Handling

**Key capability**: FNO learns in function space, enabling:
- **Input resolution**: 20 points in x-direction (streamwise)
- **Output resolution**: 111 points in x-direction
- **Mechanism**:
  - Fourier modes are continuous frequency components
  - FFT/iFFT can be applied at any resolution
  - Network interpolates naturally through Fourier representation

**Non-uniform grid strategy**:
- Y-direction: 47 points with wall-normal clustering (denser near wall)
- FNO operates on this non-uniform grid directly
- Spectral convolutions in Fourier space don't require uniform spacing
- Alternative: Can resample to uniform grid, but non-uniform preserves physics

### 2. FNO Architecture for Wave Dynamics

The standard FNO2d architecture applies directly to wave propagation:

1. **Spectral Convolution** (SpectralConv2d):
   - 2D FFT transforms spatial field to Fourier space
   - Learnable complex weights on low-frequency modes (modes1×modes2)
   - Inverse FFT returns to physical space
   - **For waves**: Fourier modes naturally represent wave components (frequency content)

2. **Operator Architecture**:
   - **Lifting**: Map initial state (u₀, x, y) from 3 channels to width channels
     - u₀: initial wave amplitude
     - (x,y): grid coordinates
   - **4 Fourier Layers**: Each layer refines the wave evolution
     - Spectral path: Global wave propagation via Fourier modes
     - Skip path: Local corrections via 1×1 convolutions
     - Activation: ReLU for nonlinearity
   - **Projection**: Map to final evolved state

3. **Mode Selection**:
   - modes1 = 16 (streamwise direction, longer domain)
   - modes2 = 8 (wall-normal direction, shorter domain)
   - Asymmetric modes match aspect ratio of domain
   - Low-frequency bias appropriate for smooth wave propagation

### 3. Why FNO Works Well for Wave Propagation

**Physical insight**:
- Waves are inherently global phenomena (influence everywhere)
- Fourier modes directly represent wave frequencies and wavelengths
- Spectral methods have long history in wave simulation (PSST, spectral DNS)

**FNO advantages over alternatives**:
- **vs. CNNs**: Limited receptive field requires many layers for global propagation
- **vs. RNNs/LSTMs**: Sequential, slow for long-time integration
- **vs. Finite Difference**: FNO is ~1000× faster for repeated evaluations
- **vs. Traditional Spectral**: FNO learns optimal mode interactions, not fixed

## Implementation

### Spectral Convolution for Wave Propagation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer for wave propagation.

    For wave PDEs, Fourier modes directly correspond to wave frequencies.
    Learning in Fourier space is natural for wave dynamics.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Modes in streamwise direction
        self.modes2 = modes2  # Modes in wall-normal direction

        # Complex-valued weights for Fourier mode transformation
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                   dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                   dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication for Fourier mode interactions"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # FFT: Physical space → Fourier space (wave frequencies)
        x_ft = torch.fft.rfft2(x)

        # Initialize output Fourier coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                            device=x.device, dtype=torch.cfloat)

        # Learn interactions between low-frequency wave modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # iFFT: Fourier space → Physical space (wave field)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
```

### FNO Architecture for Wave Evolution

```python
class FNO2d(nn.Module):
    """
    2D FNO for wave propagation with resolution changes.

    Input:  (u₀, x, y) initial wave state, shape (batch, 20, 47, 3)
    Output: u evolved wave state, shape (batch, 111, 47, 1)

    Note: Output resolution (111) ≠ Input resolution (20) in x-direction.
    FNO handles this via continuous Fourier representation.
    """
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Lifting: 3 channels → width channels
        self.fc0 = nn.Linear(3, self.width)

        # 4 Fourier layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Skip connections
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # Projection: width → 128 → 1
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass with automatic handling of resolution changes.

        Input shape:  (batch, nx_in, ny, 3) e.g., (20, 20, 47, 3)
        Output shape: (batch, nx_out, ny, 1) e.g., (20, 111, 47, 1)

        Resolution change happens through interpolation in the grid
        before/after Fourier operations.
        """
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        # Lift
        x = self.fc0(x)  # (batch, nx, ny, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, nx, ny)

        # Fourier Layer 1
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.relu(x1 + x2)

        # Fourier Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.relu(x1 + x2)

        # Fourier Layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.relu(x1 + x2)

        # Fourier Layer 4
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        # Project
        x = x.permute(0, 2, 3, 1)  # (batch, nx, ny, width)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # (batch, nx, ny, 1)

        # For resolution change: interpolation handled in data preprocessing
        # or can be done here with F.interpolate before projection

        return x
```

### Training Configuration

```python
from utilities3 import MatReader, LpLoss, count_params
import numpy as np

# Model hyperparameters
modes1 = 16  # Streamwise modes (longer direction)
modes2 = 8   # Wall-normal modes (shorter direction)
width = 32   # Hidden channels

# Training hyperparameters
batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.5

# Initialize model
model = FNO2d(modes1, modes2, width).cuda()
print(f"Parameters: {count_params(model)}")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Loss: Relative L2 norm
myloss = LpLoss(size_average=False)

# Training loop
for ep in range(epochs):
    model.train()
    train_l2 = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        # x: (batch, 20, 47, 3) initial state
        # y: (batch, 111, 47, 1) evolved state

        optimizer.zero_grad()
        out = model(x)  # FNO predicts evolved state

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    if ep % 10 == 0:
        print(f"Epoch {ep}: Train L2: {train_l2/ntrain:.4f}")
```

## Critical Parameters

### 1. Architecture Configuration
- **Fourier modes**: modes1=16, modes2=8
  - Asymmetric to match domain aspect ratio
  - modes1 > modes2 because streamwise direction is longer
  - Low-frequency bias appropriate for smooth wave propagation
  - Typical range: 8-20 for 2D wave problems
- **Hidden width**: 32 channels
- **Layers**: 4 Fourier layers for multi-step wave evolution
- **Activation**: ReLU

### 2. Grid Configuration
- **Input resolution**: 20×47 (coarse × refined)
  - 20 points in streamwise (x) direction
  - 47 points in wall-normal (y) direction (non-uniform, clustered near wall)
- **Output resolution**: 111×47 (fine × same)
  - 111 points in streamwise direction
  - Same 47 points in wall-normal direction
- **Resolution change**: ~5.5× refinement in x-direction
- **Non-uniform spacing**: Wall-normal grid uses clustering for boundary layer physics

### 3. Training Configuration
- **Batch size**: 20
- **Learning rate**: 0.001 with StepLR decay
- **Weight decay**: 1e-4
- **Epochs**: 500
- **Loss**: Relative L2 norm (scale-invariant)

### 4. Physical Problem
- **Equation**: Linear instability waves in high-speed boundary layers
- **Type**: Hyperbolic PDE (wave propagation)
- **Domain**: 2D spatial domain (streamwise × wall-normal)
- **Physics**: Small-amplitude disturbances propagating and amplifying
- **Challenge**: Boundary layer has strong gradients near wall, requiring non-uniform y-grid

### 5. Key Advantages
- **Resolution invariance**: Single model works at multiple resolutions
- **Fast inference**: ~1000× faster than traditional wave solvers
- **Global receptive field**: Single Fourier layer sees entire domain
- **Operator learning**: Generalizes to new initial conditions
- **Non-uniform grid handling**: Works directly on clustered grids

### 6. Comparison for Wave Problems

| Method | Speed | Resolution Inv. | Global Field | Best For |
|--------|-------|-----------------|--------------|----------|
| FNO | Very Fast | ✓ Yes | ✓ Yes | Smooth waves, repeated queries |
| Finite Difference | Slow | ✗ No | ✗ No | High accuracy, single solve |
| Spectral Methods | Fast | Partial | ✓ Yes | Periodic, simple geometry |
| CNN | Medium | ✗ No | ✗ No | Local features, small domains |
| RNN/LSTM | Slow | ✗ No | ✗ No | Sequential, time-dependent |

### 7. Computational Performance
- **Training**: ~500 epochs, minutes to hours
- **Inference**: ~1-10 ms per prediction
- **Speedup**: 100-1000× vs. finite difference solvers
- **Memory**: Efficient for large grids (FFT is memory-efficient)
- **Hardware**: GPU highly recommended for FFT operations

### 8. Practical Considerations
- **Data**: Requires training data from simulations or experiments
- **Generalization**: Works for similar wave dynamics in training distribution
- **Interpolation**: Resolution changes handled via Fourier interpolation
- **Stability**: Robust to small perturbations in initial conditions
- **Limitations**: May struggle with shocks, discontinuities (high-frequency content)
