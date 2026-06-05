# DeepONet for 2D Advection Equations (Initial Conditions II & III)

**Keywords**: [PDE, hyperbolic, linear, forward-problem, advection, 2D, DeepONet, MLP, adam, mse, relative-l2, deepxde, tensorflow]

**Problem:** Learning the solution operator for the 1D linear advection equation:
```
∂u/∂t + ∂u/∂x = 0,  x ∈ [0,1], t ∈ [0,1], periodic BC
```
The operator G maps initial conditions u₀(x) to solutions u(x,t) = u₀(x-t) on a 40×40 grid. Initial conditions (IC2) are random combinations of rectangular pulse + smooth bump with varying heights, centers, widths, and steepness. 1000 training/test samples each.

**Issues addressed:**
- Traditional numerical solvers require re-solving PDEs for each new initial condition
- Difficulty in learning operators that map between infinite-dimensional function spaces
- Need for efficient surrogate models that can rapidly predict solutions for new initial conditions
- Generalization to unseen query points in the spatio-temporal domain

## Key Method

DeepONet (Deep Operator Network) learns nonlinear operators between function spaces using a two-network architecture:

1. **Branch Network**: Encodes the input function (initial condition u₀) into a fixed-dimensional representation
2. **Trunk Network**: Encodes the query locations (x, t) where the solution is evaluated

The output is computed as the dot product of branch and trunk outputs:
```
G(u₀)(x,t) ≈ Σᵢ (branch_net(u₀))ᵢ · (trunk_net(x,t))ᵢ
```

**Architecture**: Branch [40, 512, 512] processes 40-point IC. Trunk [2, 512, 512, 512] processes (x,t). Output: 512-dim dot product.

**POD-DeepONet**: Uses PCA to extract ~100-200 dominant modes from 1600-dim output. Branch predicts POD coefficients αᵢ. Reconstruction: u = ū + Σᵢ αᵢφᵢ. Drastically improves efficiency.

This architecture satisfies the universal approximation theorem for operators: any continuous operator can be approximated by DeepONet given sufficient basis functions.

**Data Format**:
- Input: u₀(x) sampled at nx locations (initial condition)
- Output locations: (x,t) pairs in a 40×40 grid
- Output: u(x,t) solution values

**POD-DeepONet Variant**:
POD-DeepONet enhances standard DeepONet by incorporating Proper Orthogonal Decomposition (POD) to extract dominant modes from training data:

1. Apply PCA to training outputs to extract POD basis functions φᵢ
2. Use POD basis as part of the trunk representation
3. Reconstruct solution: u(x,t) = ū + Σᵢ αᵢ(u₀) φᵢ(x,t)

where ū is the mean solution and αᵢ are coefficients predicted by the branch network. This reduces the effective dimensionality and improves training efficiency.

## Implementation

### Standard DeepONet

```python
import deepxde as dde
import numpy as np

def get_data(filename):
    """
    Load and prepare advection equation data.

    Returns:
        x_train: tuple of (u0, xt) where:
            - u0: initial conditions, shape (N, nx)
            - xt: query points (x,t), shape (nt*nx, 2)
        y_train: solutions u(x,t), shape (N, nt*nx)
    """
    nx = 40  # spatial grid points
    nt = 40  # temporal grid points
    data = np.load(filename)
    x = data["x"].astype(np.float32)  # spatial coordinates
    t = data["t"].astype(np.float32)  # temporal coordinates
    u = data["u"].astype(np.float32)  # solutions, shape: (N, nt, nx)

    u0 = u[:, 0, :]  # Extract initial conditions at t=0, shape: (N, nx)
    xt = np.vstack((np.ravel(x), np.ravel(t))).T  # Create (x,t) query grid
    u = u.reshape(-1, nt * nx)  # Flatten solutions
    return (u0, xt), u


# Load training and test data for IC type 2
x_train, y_train = get_data("train_IC2.npz")
x_test, y_test = get_data("test_IC2.npz")

# Create DeepONet dataset
data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

# Define DeepONet architecture
# Branch net: maps u0 (40 points) → 512 → 512 → output
# Trunk net: maps (x,t) → 512 → 512 → 512 → output
# Output dimension must match between branch and trunk
net = dde.maps.DeepONetCartesianProd(
    [nx, 512, 512],           # Branch network: [input_dim, hidden1, hidden2]
    [2, 512, 512, 512],       # Trunk network: [2 for (x,t), hidden1, hidden2, hidden3]
    "relu",                    # Activation function
    "Glorot normal"            # Weight initialization
)

# Compile model
model = dde.Model(data, net)
model.compile(
    "adam",                             # Optimizer
    lr=1e-3,                            # Learning rate
    decay=("inverse time", 1, 1e-4),   # Learning rate decay schedule
    metrics=["mean l2 relative error"], # Evaluation metric
)

# Train for IC2 (IC1 would use 100,000 epochs)
losshistory, train_state = model.train(epochs=250000, batch_size=None)
```

### POD-DeepONet with Dimensionality Reduction

```python
from sklearn.decomposition import PCA
import tensorflow as tf

class PODDeepONet(dde.maps.NN):
    """
    POD-enhanced DeepONet that uses PCA-extracted basis functions.
    Reduces output dimensionality by projecting onto dominant POD modes.
    """
    def __init__(
        self,
        pod_basis,              # POD basis functions from PCA, shape: (n_outputs, n_components)
        layer_sizes_branch,     # Branch network architecture
        layer_sizes_trunk,      # Trunk network architecture (None for POD-only)
        activation,
        kernel_initializer,
    ):
        super().__init__()
        activation_branch = dde.maps.activations.get(activation)
        self.activation_trunk = dde.maps.activations.get(activation)

        # Store POD basis as tensor
        self.pod_basis = tf.convert_to_tensor(pod_basis, dtype=tf.float32)

        # Branch network predicts coefficients for POD modes
        self.branch = dde.maps.FNN(
            layer_sizes_branch, activation_branch, kernel_initializer
        )

        # Trunk network (optional, None for pure POD)
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = dde.maps.FNN(
                layer_sizes_trunk, self.activation_trunk, kernel_initializer
            )
            self.b = tf.Variable(tf.zeros(1))

    def call(self, inputs, training=False):
        x_func = inputs[0]  # Initial condition u0
        x_loc = inputs[1]   # Query locations (x,t)

        # Branch network predicts POD coefficients
        x_func = self.branch(x_func)

        if self.trunk is None:
            # POD-only: reconstruct as linear combination of POD modes
            # output = Σᵢ αᵢ * φᵢ
            x = tf.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            # POD + trunk network enhancement
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = tf.einsum("bi,ni->bn", x_func, tf.concat((self.pod_basis, x_loc), 1))
            x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


# Extract POD basis using PCA
# Keep components explaining 99.99% of variance
pca = PCA(n_components=0.9999).fit(y_train)
print("# POD Components:", pca.n_components_)

# Create POD-DeepONet
# Branch output dimension = number of POD components (much smaller than nt*nx)
net = PODDeepONet(
    pca.components_.T * 40,           # POD basis functions, scaled
    [nx, 512, pca.n_components_],     # Branch: 40 → 512 → n_components
    None,                              # No trunk network (pure POD)
    "relu",
    "Glorot normal",
)

# Output transform: reconstruct from POD coefficients
def output_transform(inputs, outputs):
    """Add mean and normalize by number of components"""
    return outputs / pca.n_components_ + pca.mean_

net.apply_output_transform(output_transform)

# Compile and train
model = dde.Model(data, net)
model.compile(
    "adam",
    lr=1e-3,
    decay=("inverse time", 1, 1e-4),
    metrics=["mean l2 relative error"],
)
losshistory, train_state = model.train(epochs=100000, batch_size=None)
```

## Critical Parameters

1. **Network Architecture**:
   - Standard DeepONet:
     - Branch: [40, 512, 512] - processes 40-point initial condition
     - Trunk: [2, 512, 512, 512] - processes (x,t) coordinates
   - POD-DeepONet:
     - Branch: [40, 512, n_components] - outputs POD coefficients
     - No trunk network (pure POD reconstruction)

2. **POD Configuration**:
   - Variance threshold: 0.9999 (captures 99.99% of solution variance)
   - POD basis scaling factor: 40
   - Output transform: normalize by n_components and add mean

3. **Training Configuration**:
   - Optimizer: Adam
   - Learning rate: 1e-3
   - Decay: inverse time decay with period=1, decay_rate=1e-4
   - Batch size: None (full batch)
   - Epochs: 250,000 (standard), 100,000 (POD variant)

4. **Data Configuration**:
   - Spatial grid: 40 points
   - Temporal grid: 40 points
   - Total output dimension: 1600 (40×40)
   - Initial condition: sampled at 40 spatial locations

5. **Activation Function**: ReLU throughout

6. **Weight Initialization**: Glorot normal (Xavier normal)

7. **Evaluation Metric**: Mean L2 relative error
