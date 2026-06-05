# DeepONet for 1D Burgers' Equation

**Keywords**: [PDE, parabolic, nonlinear, forward-problem, burgers, 1D, periodic, DeepONet, MLP, adam, mse, relative-l2, deepxde, tensorflow]

**Problem:** Learning the solution operator for the 1D viscous Burgers' equation:
```
∂u/∂t + u∂u/∂x = ν∂²u/∂x²,  x ∈ [0,1], periodic BC, ν = 0.1
```
The operator G maps initial conditions a(x) (from Gaussian Random Fields) to steady-state solutions u(x) at t=1, discretized on 128 points (subsampled from 8192). This nonlinear convection-diffusion PDE models shock formation. 1000 training, 200 test samples.

**Issues addressed:**
- Computational cost of solving nonlinear PDEs for many initial conditions
- Need for operator learning on periodic domains
- Handling nonlinear convection-diffusion dynamics
- Generalization across different initial conditions drawn from a distribution

## Key Method

DeepONet learns the solution operator G: a(x) → u(x) using:

1. **Branch Network**: Encodes discretized initial condition a ∈ ℝ^m into a latent representation
2. **Trunk Network**: Encodes spatial query locations x ∈ [0,1] with periodic feature transformation
3. **Output**: Inner product of branch and trunk outputs, with output standardization

**Architecture**: Branch [128, 128, 128, 128, 128] (4 hidden layers). Trunk input x undergoes periodic transform: x → [cos(2πx), sin(2πx), cos(4πx), sin(4πx)] (1D→4D), then [4, 128, 128, 128]. Output standardized by training mean/std for stability.

**Key Features**:
- **Periodic Feature Transform**: Captures periodic BC naturally in trunk network
- **Output Standardization**: Normalizes outputs using training data statistics for stable training
- **Subsampling**: Uses 128 spatial points (subsampled from 8192) for computational efficiency

**Data**:
- Initial conditions: a(x) sampled from Gaussian Random Field (GRF) with length scale 0.2
- Solutions: Steady-state u(x) obtained by solving Burgers' equation to t=1
- Spatial domain: [0,1] with periodic BC
- Viscosity: ν = 0.1 (Reynolds number R=10)

## Implementation

```python
import deepxde as dde
import numpy as np
from deepxde.backend import tf
from scipy import io
from sklearn.preprocessing import StandardScaler


def periodic(x):
    """
    Periodic feature transformation for trunk network.
    Maps x ∈ [0,1] to Fourier features capturing periodicity.

    Args:
        x: spatial coordinate, shape (n, 1)
    Returns:
        Fourier features [cos(2πx), sin(2πx), cos(4πx), sin(4πx)], shape (n, 4)
    """
    x *= 2 * np.pi
    return tf.concat(
        [tf.math.cos(x), tf.math.sin(x), tf.math.cos(2 * x), tf.math.sin(2 * x)], 1
    )


def get_data(ntrain, ntest):
    """
    Load and prepare Burgers' equation data.

    Args:
        ntrain: number of training samples
        ntest: number of test samples

    Returns:
        x_train: (branch_input, trunk_input) where:
            - branch_input: initial conditions a(x), shape (ntrain, m)
            - trunk_input: spatial grid, shape (s, 1)
        y_train: steady-state solutions u(x), shape (ntrain, s)
        x_test, y_test: similar for test set
    """
    sub_x = 2 ** 6  # Subsampling rate for input: 8192 / 64 = 128 points
    sub_y = 2 ** 6  # Subsampling rate for output: 8192 / 64 = 128 points

    # Load data: shape (2048 samples, 8192 grid points)
    data = io.loadmat("burgers_data_R10.mat")
    x_data = data["a"][:, ::sub_x].astype(np.float32)  # Initial conditions
    y_data = data["u"][:, ::sub_y].astype(np.float32)  # Steady-state solutions

    # Split into train and test
    x_branch_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_branch_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    # Create spatial grid: [0, 1] with 128 points
    s = 2 ** 13 // sub_y  # 8192 / 64 = 128
    grid = np.linspace(0, 1, num=2 ** 13)[::sub_y, None]

    x_train = (x_branch_train, grid)
    x_test = (x_branch_test, grid)
    return x_train, y_train, x_test, y_test


def train(model, lr, epochs):
    """Train model with Adam optimizer and inverse time decay."""
    decay = ("inverse time", epochs // 5, 0.5)
    model.compile(
        "adam",
        lr=lr,
        metrics=["mean l2 relative error"],
        decay=decay
    )
    losshistory, train_state = model.train(epochs=epochs, batch_size=None)
    print("\nTraining done ...\n")


# Load data: 1000 training, 200 test samples
x_train, y_train, x_test, y_test = get_data(1000, 200)

# DeepONet architecture
m = 2 ** 7  # 128 input points for branch network
net = dde.maps.DeepONetCartesianProd(
    [m, 128, 128, 128, 128],    # Branch: 128 → 128 → 128 → 128 → 128 (4 hidden layers)
    [1, 128, 128, 128],          # Trunk: 1 → 128 → 128 → 128 (3 hidden layers)
    "tanh",                      # Activation function
    "Glorot normal"              # Weight initialization
)

# Apply periodic feature transformation to trunk network input
# Trunk input: x (1D) → [cos(2πx), sin(2πx), cos(4πx), sin(4πx)] (4D)
net.apply_feature_transform(periodic)

# Standardize output using training data statistics
scaler = StandardScaler().fit(y_train)
std = np.sqrt(scaler.var_.astype(np.float32))

def output_transform(inputs, outputs):
    """
    Denormalize network output: y_pred = y_raw * std + mean
    Improves training stability by normalizing targets.
    """
    return outputs * std + scaler.mean_.astype(np.float32)

net.apply_output_transform(output_transform)

# Create dataset and model
data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
model = dde.Model(data, net)

# Train with Adam optimizer
lr = 0.001
epochs = 500000
train(model, lr, epochs)
```

## Critical Parameters

1. **Network Architecture**:
   - Branch network: [128, 128, 128, 128, 128]
     - Input: 128-point initial condition a(x)
     - 4 hidden layers with 128 neurons each
   - Trunk network: [1, 128, 128, 128]
     - Input: 1D spatial coordinate x
     - After periodic transform: 4D Fourier features
     - 3 hidden layers with 128 neurons each
   - Output dimension: 128 (matching branch and trunk)

2. **Feature Transformation**:
   - Periodic features: [cos(2πx), sin(2πx), cos(4πx), sin(4πx)]
   - Critical for capturing periodic boundary conditions
   - Extends trunk input from 1D to 4D

3. **Output Normalization**:
   - Standardization: zero mean, unit variance
   - Applied via output_transform
   - Improves gradient flow and training stability

4. **Data Configuration**:
   - Spatial resolution: 128 points (subsampled from 8192)
   - Training samples: 1000
   - Test samples: 200
   - Domain: [0, 1] with periodic BC
   - Viscosity: ν = 0.1 (Reynolds number R=10)

5. **Training Configuration**:
   - Optimizer: Adam
   - Learning rate: 0.001
   - Learning rate decay: inverse time with period=epochs//5, decay_rate=0.5
   - Epochs: 500,000
   - Batch size: None (full batch training)

6. **Activation Function**: tanh (hyperbolic tangent)
   - Smooth, bounded activation
   - Better gradient properties than ReLU for operator learning

7. **Weight Initialization**: Glorot normal (Xavier normal)

8. **Evaluation Metric**: Mean L2 relative error
