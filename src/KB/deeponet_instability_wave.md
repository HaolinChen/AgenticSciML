# POD-DeepONet for Instability Wave Propagation

**Keywords**: [PDE, hyperbolic, nonlinear, forward-problem, 2D, DeepONet, MLP, PCA, adam, mse, relative-l2, deepxde, tensorflow]

**Problem:** Learning the solution operator for linear instability waves in high-speed boundary layers. Hyperbolic wave PDE with:
- **Input**: Initial wave state on 20×47 grid (940 DOF)
- **Output**: Evolved wave state on 111×47 grid (5217 DOF). POD-DeepONet addresses the high-dimensional output (5217) via dimensionality reduction using PCA.

**Issues addressed:**
- High-dimensional output spaces (111×47 = 5217 points)
- Training efficiency through dimensionality reduction using Proper Orthogonal Decomposition (POD)
- Non-uniform spatial grids (non-uniform in y-direction)
- Capturing dominant flow structures via POD modes

## Key Method

**POD-DeepONet** enhances standard DeepONet by:
1. **POD Basis Extraction**: Apply PCA to training outputs, retain modes explaining 99.99% variance
2. **Branch Network**: Predicts coefficients for POD modes instead of full output
3. **Reconstruction**: u(x,y) = ū + Σᵢ αᵢ(input) · φᵢ(x,y)

where ū is mean, φᵢ are POD modes, αᵢ are branch network outputs.

**Architecture**:
- PCA extracts ~100-200 dominant modes from 5217-dim output (99.99% variance)
- Branch [940, 512, n_components]: Maps flattened 20×47 input to POD coefficients
- Trunk: None (pure POD, no trunk network)
- Reconstruction: u = mean + Σᵢ αᵢφᵢ (αᵢ from branch, φᵢ from PCA)

**Advantage**: Output dimension reduced from 5217 to ~100-200 POD components, dramatically improving training efficiency and memory usage.

## Implementation

```python
import deepxde as dde
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

def get_data(path):
    """Load instability wave data with non-uniform grid"""
    x_train = np.load(path + "X_train.npy").astype(np.float32)  # (40800, 20, 47)
    y_train = np.load(path + "Y_train.npy").astype(np.float32)  # (40800, 111, 47)
    x_test = np.load(path + "X_valid.npy").astype(np.float32)  # (10000, 20, 47)
    y_test = np.load(path + "Y_valid.npy").astype(np.float32)  # (10000, 111, 47)

    ry = np.load(path + "ry.npy")  # (47,) non-uniform y-coordinates
    rx = np.load(path + "rx.npy")  # (111,) uniform x-coordinates
    xx, yy = np.meshgrid(rx, ry, indexing="ij")  # (111, 47)
    grid = np.vstack((xx.ravel(), yy.ravel())).T  # (5217, 2)

    # Flatten spatial dimensions
    x_train = (x_train.reshape(-1, 20 * 47), grid)
    y_train = y_train.reshape(-1, 111 * 47)
    x_test = (x_test.reshape(-1, 20 * 47), grid)
    y_test = y_test.reshape(-1, 111 * 47)
    return dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)


class PODDeepONet(dde.maps.NN):
    """DeepONet with POD basis for dimensionality reduction"""
    def __init__(self, pod_basis, layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer):
        super().__init__()
        activation_branch = dde.maps.activations.get(activation)
        self.activation_trunk = dde.maps.activations.get(activation)

        # Store POD basis
        self.pod_basis = tf.convert_to_tensor(pod_basis, dtype=tf.float32)

        # Branch network predicts POD coefficients
        self.branch = dde.maps.FNN(layer_sizes_branch, activation_branch, kernel_initializer)

        # Optional trunk network (None for pure POD)
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = dde.maps.FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
            self.b = tf.Variable(tf.zeros(1))

    def call(self, inputs, training=False):
        x_func = inputs[0]  # Branch input
        x_loc = inputs[1]   # Trunk input (query locations)

        x_func = self.branch(x_func)  # Predict POD coefficients

        if self.trunk is None:
            # POD-only reconstruction: u = Σ αᵢ φᵢ
            x = tf.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            # POD + trunk enhancement
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = tf.einsum("bi,ni->bn", x_func, tf.concat((self.pod_basis, x_loc), 1))
            x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


# Load data
data = get_data("path/to/data/")

# Extract POD basis using PCA
pca = PCA(n_components=0.9999).fit(y_train)  # Keep 99.99% variance
print(f"POD components: {pca.n_components_}")

# Build POD-DeepONet
net = PODDeepONet(
    pca.components_.T,           # POD basis functions
    [20*47, 512, pca.n_components_],  # Branch: 940 → 512 → n_components
    None,                        # No trunk (pure POD)
    "relu",
    "Glorot normal"
)

# Output transform: add mean
def output_transform(inputs, outputs):
    return outputs + pca.mean_

net.apply_output_transform(output_transform)

# Train
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
model.train(epochs=50000, batch_size=None)
```

## Critical Parameters

1. **POD Configuration**: Variance threshold = 0.9999 (captures ~100-200 modes)
2. **Architecture**: Branch=[940, 512, n_components], Trunk=None
3. **Grid**: Input 20×47=940, Output 111×47=5217
4. **Training**: epochs=50000, batch_size=None, lr=0.001
5. **Data**: 40800 training, 10000 test samples
