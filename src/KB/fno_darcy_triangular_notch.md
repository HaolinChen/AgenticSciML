# Fourier Neural Operator (FNO) for Irregular Geometries

**Keywords**: [PDE, elliptic, linear, forward-problem, darcy, 2D, irregular, dirichlet, heterogeneous, FNO, spectral-method, adam, mse, relative-l2, gpu, pytorch]

**Problem:** Fourier Neural Operator (FNO) for irregular geometries demonstrates how neural operators based on spectral methods can handle complex, non-rectangular domains through embedding techniques. For the 2D Darcy flow on a triangular domain with a notch (irregular geometry with 2295 non-uniform mesh points), FNO learns the solution operator mapping permeability fields a(x,y) to pressure fields u(x,y) for the governing PDE:
```
-∇·(a(x,y)∇u(x,y)) = f,  on triangular domain with notch, u = 0 on ∂Ω
```

The key challenge is that FNO's spectral convolutions naturally operate on regular grids in Fourier space, but many real-world problems (e.g., flow around obstacles, complex geological formations, airfoils) require handling irregular geometries with non-uniform meshes.

**Issues addressed:**
- **Irregular geometry handling**: FNO can be applied to non-rectangular domains by embedding the irregular domain into a regular grid (with masking or zero-padding). This allows spectral methods to be used even for complex shapes, avoiding the need for specialized mesh-based or graph-based architectures.
- **Resolution invariance on irregular domains**: Even with irregular geometries, FNO maintains discretization-invariance—the model learns the operator in function space, not tied to specific mesh resolution or structure. Train on coarse mesh, evaluate on fine mesh.
- **Mesh-independent learning**: Traditional mesh-based methods (FEM, FVM) require re-meshing and re-solving for each new geometry or resolution. FNO provides a mesh-free surrogate that generalizes across different discretizations without architecture changes.
- **Global receptive field for elliptic PDEs**: Boundary conditions on complex geometries affect the entire domain through long-range interactions. FNO's Fourier-based global receptive field naturally captures these dependencies in a single layer.
- **Computational efficiency**: Embedding irregular domains into regular grids allows efficient FFT operations (O(N log N)) rather than expensive sparse matrix operations, graph neural networks (O(N²)), or attention mechanisms (O(N²)).

## Key Method

FNO for irregular geometries extends the standard FNO architecture with geometry handling strategies:

### 1. Irregular Geometry Handling Strategy

**Embedding approach**:
- Irregular domain (e.g., triangle with notch, 2295 points) is embedded into a bounding regular rectangular grid
- Zero-padding or masking is applied outside the physical domain
- FNO operates on the regular grid using standard 2D FFT-based spectral convolutions
- Output is extracted only on the physical domain points (or mask is applied)

**Why this works**:
- Fourier transform naturally extends functions beyond their domain
- Learnable weights adapt to the specific geometry through training
- Coordinate encoding (including x,y as input channels) helps network understand domain shape
- Skip connections in physical space provide local geometric information

**Advantages over alternatives**:
- **vs. Graph Neural Networks**: O(N log N) FFT vs. O(N²) message passing; resolution-invariant
- **vs. PointNet/Transformer**: O(N log N) vs. O(N²) attention; better for smooth PDE solutions
- **vs. Finite Element Methods**: 1000× faster inference; no re-solving or remeshing needed

### 2. FNO Architecture

The Fourier Neural Operator uses the same core architecture as for regular domains:

1. **Spectral Convolution** (SpectralConv2d):
   - 2D FFT transforms the embedded grid to Fourier space
   - Learnable complex-valued weights multiply selected low-frequency Fourier modes
   - Inverse 2D FFT returns to physical space
   - Mode truncation (keeping only modes1 × modes2 out of N × N available) provides:
     - Implicit regularization (removes high-frequency noise)
     - Compression (reduces parameters)
     - Smoothness bias (appropriate for Darcy flow)

2. **Operator Architecture** (SimpleBlock2d):
   - **Lifting**: Map input (a(x,y), x, y) from 3 channels to width=32 hidden channels
     - Coordinates (x,y) encode geometry information
     - Permeability a(x,y) is the PDE coefficient
   - **Fourier Layers**: 4 layers, each containing:
     - **Spectral path**: Global features via Fourier modes (long-range interactions)
     - **Skip path**: Local features via 1×1 convolutions (boundary details)
     - **Activation**: GELU or ReLU for nonlinearity
   - **Projection**: Map hidden channels to 1 output channel (pressure u)

3. **Resolution Invariance**:
   - Training: Can use coarser embedded mesh (e.g., fewer grid points)
   - Testing: Can evaluate on finer mesh or different point distributions
   - Mechanism: Fourier representation is continuous; FFT interpolation happens naturally

### 3. Key Innovation: Spectral Methods on Irregular Domains

Traditional wisdom: Spectral methods (Fourier, Chebyshev) work best on regular, periodic, or simple rectangular domains.

**FNO's insight**: By learning in Fourier space rather than using fixed spectral basis functions, the neural operator can:
- Automatically adapt Fourier representations to irregular boundaries
- Learn which Fourier modes are relevant for the specific geometry class
- Combine global spectral features with local physical-space operations (skip connections)
- Handle geometry variations without architectural changes

This makes FNO applicable to a much wider range of geometries than classical spectral methods.

## Implementation

### Spectral Convolution Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer for spectral convolution.

    Performs:
    1. FFT: Transform to Fourier space
    2. Linear transform: Multiply selected Fourier modes by learnable weights
    3. Inverse FFT: Return to physical space

    Works on regular grids, including embeddings of irregular domains.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in x-direction (≤ N/2 + 1)
        self.modes2 = modes2  # Number of Fourier modes in y-direction (≤ N/2 + 1)

        # Learnable complex-valued weights for Fourier mode multiplication
        # Initialize with scaled random values
        self.scale = (1 / (in_channels * out_channels))

        # weights1: for low-frequency modes (top-left corner in frequency domain)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                   dtype=torch.cfloat)
        )

        # weights2: for low-frequency modes (bottom-left corner, negative frequencies in x)
        # Due to real FFT symmetry
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

        # Step 1: Apply 2D Real FFT to transform to Fourier space
        # Input shape: (batch, channels, height, width)
        # Output shape: (batch, channels, height, width//2 + 1) complex
        # Even for irregular domains embedded in regular grids, FFT works!
        x_ft = torch.fft.rfft2(x)

        # Step 2: Multiply relevant Fourier modes with learnable weights
        # Initialize output Fourier coefficients (all zeros initially)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                            device=x.device, dtype=torch.cfloat)

        # Multiply low-frequency modes (positive frequencies in x)
        # Top-left corner of frequency domain
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Multiply low-frequency modes (negative frequencies in x)
        # Bottom-left corner of frequency domain
        # Due to conjugate symmetry in real FFT
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # High-frequency modes are implicitly set to zero (mode truncation)
        # This provides:
        # - Implicit regularization (smoothness prior)
        # - Computational efficiency (fewer parameters)
        # - Better generalization for smooth PDEs

        # Step 3: Apply inverse 2D Real FFT to return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
```

### FNO Architecture for Irregular Domains

```python
class SimpleBlock2d(nn.Module):
    """
    2D Fourier Neural Operator for irregular geometries.

    The architecture is the same as for regular domains, but the input/output
    represent irregular geometries embedded in regular grids.

    Input: (a(x,y), x, y) on embedded grid, shape (batch, s, s, 3)
           - a(x,y): permeability coefficient field
           - x, y: spatial coordinates (encode geometry)
    Output: u(x,y) pressure field on embedded grid, shape (batch, s, s, 1)

    Key insight: Including (x,y) coordinates as input channels helps the network
    understand where the physical domain is within the embedded grid.
    """
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1  # Fourier modes in x
        self.modes2 = modes2  # Fourier modes in y
        self.width = width    # Hidden channel dimension

        # Lifting layer: map 3 input channels to hidden representation
        self.fc0 = nn.Linear(3, self.width)  # (a, x, y) -> width channels

        # 4 Fourier layers with spectral convolutions
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Skip connections (local operations in physical space)
        # 1×1 convolutions to capture local geometric features
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Projection layers: map from hidden space to output
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        # Lift: (batch, s, s, 3) -> (batch, s, s, width)
        x = self.fc0(x)

        # Permute for convolution: (batch, width, s, s)
        x = x.permute(0, 3, 1, 2)

        # Fourier Layer 1: Spectral conv + Skip connection + Activation
        x1 = self.conv0(x)  # Global operation in Fourier space
        x2 = self.w0(x)     # Local operation in physical space
        x = F.gelu(x1 + x2) # GELU activation (smooth, often better for PDEs)

        # Fourier Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        # Fourier Layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        # Fourier Layer 4 (no activation on final layer)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Project to output: (batch, width, s, s) -> (batch, s, s, 1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x
```

### Training Configuration for Irregular Domains

```python
from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss, count_params
import numpy as np

# Data configuration
# For irregular domains, data typically comes from FEM/FVM solvers on unstructured meshes
# The data is interpolated/embedded onto regular grids for FNO

ntrain = 1000  # Training samples
ntest = 100    # Test samples

# Model hyperparameters
modes = 12      # Number of Fourier modes (low-frequency truncation)
width = 32      # Hidden channel dimension

# Training hyperparameters
batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 100  # LR decay step
gamma = 0.5      # LR decay factor

# Initialize model
model = SimpleBlock2d(modes, modes, width).cuda()
print(f"Model parameters: {count_params(model)}")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Loss function: Relative L2 norm (scale-invariant)
myloss = LpLoss(size_average=False)

# Training loop
for ep in range(epochs):
    model.train()
    train_l2 = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        # For irregular domains, loss is computed on full embedded grid
        # Or masking can be applied to compute loss only on physical domain
        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()
    train_l2 /= ntrain

    # Evaluation
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    test_l2 /= ntest

    if ep % 10 == 0:
        print(f"Epoch {ep}: Train L2: {train_l2:.4f}, Test L2: {test_l2:.4f}")
```

## Critical Parameters

### 1. Architecture Configuration
- **Fourier modes**: modes1 = modes2 = 12
  - Controls frequency content captured
  - Low-frequency modes sufficient for smooth Darcy flow
  - Typical range: 8-16 for irregular 2D domains
  - Trade-off: More modes = more parameters but may overfit
- **Hidden width**: 32 channels
  - Balance between model capacity and computational cost
  - Typical range: 20-64 for 2D problems
- **Number of layers**: 4 Fourier layers
  - Each layer refines the solution
  - Deeper networks capture more complex interactions
- **Activation**: GELU (Gaussian Error Linear Unit)
  - Smoother than ReLU, often better for function approximation
  - Alternative: ReLU, SiLU, tanh

### 2. Geometry Handling
- **Domain**: Triangular region with notch (irregular geometry)
- **Mesh points**: 2295 non-uniform points (from FEM solver)
- **Embedding strategy**:
  - Embed into bounding regular grid
  - Zero-pad or mask outside physical domain
  - Include (x,y) coordinates as input to encode geometry
- **Boundary conditions**: Dirichlet (u = 0 on irregular boundary ∂Ω)
- **Coordinate encoding**: Critical for network to understand domain shape

### 3. Training Configuration
- **Batch size**: 20
- **Learning rate**: 0.001 with StepLR decay (×0.5 every 100 epochs)
- **Weight decay**: 1e-4 (L2 regularization on parameters)
- **Epochs**: 500
- **Loss**: Relative L2 norm (scale-invariant, robust to magnitude variations)

### 4. Key Advantages for Irregular Geometries
- **Generalization**: Can handle various irregular shapes without architecture changes
- **Efficiency**: O(N log N) FFT on embedded grids vs. O(N²) for graph neural networks
- **Resolution invariance**: Train on one mesh density, evaluate on refined meshes
- **No meshing pipeline**: Avoids complex mesh generation required for FEM/FVM
- **Fast inference**: 1000× speedup over traditional PDE solvers

### 5. Comparison with Alternatives

| Method | Complexity | Resolution Inv. | Geometry Handling | Best For |
|--------|-----------|-----------------|-------------------|----------|
| FNO (embedded) | O(N log N) | ✓ Yes | Embedding + mask | Smooth PDEs, moderate complexity |
| Graph NN | O(N²) | Partial | Native | Very irregular meshes |
| Transformer | O(N²) | ✓ Yes | Native | Point clouds, small N |
| FEM/FVM | O(N^1.5-3) | ✗ No | Native | High accuracy, single solve |

### 6. Computational Performance
- **Training**: ~500 epochs, minutes to hours depending on hardware and resolution
- **Inference**: ~1-10 ms per sample (vs. seconds/minutes for FEM)
- **Speedup**: ~1000× faster than traditional PDE solvers
- **Hardware**: CUDA-enabled GPU recommended
- **Scalability**: FFT scales efficiently to large grids (up to millions of points)

### 7. Practical Considerations
- **Data generation**: Need ground-truth solutions from FEM/FVM for training
- **Embedding quality**: Better embeddings (tighter bounding box, adaptive grids) improve efficiency
- **Mask handling**: Can apply masks in loss computation or post-processing
- **Generalization**: Model generalizes to similar geometries in training distribution
- **Out-of-distribution**: Performance degrades for very different geometry types
