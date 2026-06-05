# U-FNO for Gas Saturation in CO2 Storage

**Keywords**: ["PDE", "parabolic", "nonlinear", "forward-problem", "navier-stokes", "3D", "irregular", "dirichlet", "heterogeneous", "multi-scale", "discontinuous", "turbulent", "FNO", "finite_difference", "U-Net", "adam", "relative-l2", "gpu", "pytorch"]

**Problem:** U-FNO (U-shaped Fourier Neural Operator) is an enhanced neural operator architecture for predicting multiphase flow dynamics in geological CO2 storage. It addresses the challenge of accurately predicting CO2 gas saturation evolution in heterogeneous porous media with varying permeability, porosity, reservoir thickness, injection configurations, and multiphase flow properties. The problem involves solving coupled nonlinear PDEs for CO2-water multiphase flow over 30 years of injection, requiring high spatial and temporal resolution to capture plume migration and front propagation.

**Issues addressed:**
- **Sharp fronts and discontinuities**: The U-FNO architecture combines Fourier layers with U-Net to capture both global patterns and sharp gas saturation fronts at the leading edge of the CO2 plume, achieving 2.7x better accuracy than CNN for front prediction.
- **Heterogeneous formations**: Significantly improves accuracy in highly heterogeneous geological formations (1.7x better than CNN), handling permeability variations from 0.001 mD to 10 Darcy with complex anisotropy patterns.
- **Data efficiency**: Requires only 1/3 of the training data compared to CNN benchmarks to achieve equivalent accuracy, reducing computational costs by 530 CPU hours for data generation.
- **Overfitting and generalization**: The Fourier layer provides inherent regularization, reducing overfitting by 70% compared to CNN (0.3% vs 1.0% MPE difference between training and testing).
- **Multi-scale features**: The hybrid architecture captures both low-frequency global flow patterns (via Fourier layers) and high-frequency local variations (via U-Net layers), addressing the multi-scale nature of multiphase flow.

## Key Method

U-FNO enhances the original Fourier Neural Operator (FNO) by introducing U-Fourier layers that combine:

1. **Fourier Layers** (first 3 layers): Perform integral kernel operations in Fourier space using Fast Fourier Transform, efficiently capturing global, low-frequency patterns with mesh-free properties.

2. **U-Fourier Layers** (last 3 layers): Augment Fourier operations with U-Net convolutional paths to enrich representation of high-frequency information and sharp features.

3. **Hybrid Architecture Benefits**:
   - Fourier path: Provides resolution invariance, regularization, and efficient global pattern learning
   - U-Net path: Captures local convolution features, sharp fronts, and high-frequency details
   - Combined: Lower training error than FNO while maintaining better generalization than CNN

The architecture processes 3D spatial-temporal inputs (96×200×24 grid with 12 input channels including kr, kz, porosity, injection location, injection rate, pressure, temperature, Swi, λ, grid_x, grid_y, grid_t) and outputs gas saturation evolution over 24 time snapshots.

**Training strategy**:
- Relative L2 loss with 1st derivative regularization: L(y,ŷ) = ||y-ŷ||₂/||y||₂ + 0.5·||dy/dr - dŷ/dr||₂/||dy/dr||₂
- Active cell masking for variable reservoir thickness
- Adam optimizer with step learning rate decay
- 100 epochs, batch size 4

## Implementation

```python
# SpectralConv3d: Core 3D Fourier layer performing FFT, linear transform in Fourier space, and inverse FFT
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        # modes: Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Fourier modes in r direction
        self.modes2 = modes2  # Fourier modes in z direction
        self.modes3 = modes3  # Fourier modes in t direction

        # Learnable weights in Fourier space (complex-valued)
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels,
                                                              self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels,
                                                              self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels,
                                                              self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels,
                                                              self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # Complex multiplication in Fourier space
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Forward FFT: transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes with learnable weights (truncated to low frequencies)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2),
                            x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        # Apply weights to four corners of Fourier spectrum
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Inverse FFT: return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
```

```python
# U_net: 3D U-Net for capturing local high-frequency features
class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        # Encoder path: downsampling with stride-2 convolutions
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

        # Decoder path: upsampling with transposed convolutions
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
        self.output_layer = self.output(input_channels*2, output_channels,
                                         kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):
        # Encoder with skip connections
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        # Decoder with concatenation of skip connections
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)
        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        # Convolutional block with BatchNorm, LeakyReLU, and Dropout
        return nn.Sequential(
            nn.Conv3d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        # Transposed convolution for upsampling
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)
```

```python
# SimpleBlock3d: Main U-FNO architecture with 3 Fourier layers + 3 U-Fourier layers
class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock3d, self).__init__()
        self.modes1 = modes1  # Fourier modes in each dimension
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width    # Channel width (hidden dimension)

        # Lifting: project 12 input channels to higher dimension
        self.fc0 = nn.Linear(12, self.width)

        # First 3 layers: Pure Fourier layers
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        # Last 3 layers: U-Fourier layers (Fourier + U-Net)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        # Bias terms (local linear transform in physical space)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)

        # U-Net components for last 3 layers
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.unet4 = U_net(self.width, self.width, 3, 0)
        self.unet5 = U_net(self.width, self.width, 3, 0)

        # Projection: map back to output dimension
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        # Lift input to higher dimension
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # (batch, channels, r, z, t)

        # Fourier layers 1-3: Fourier transform + bias + ReLU
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.relu(x)

        # U-Fourier layers 4-6: Fourier transform + bias + U-Net + ReLU
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet3(x)  # U-Net captures high-frequency features
        x = x1 + x2 + x3
        x = F.relu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet4(x)
        x = x1 + x2 + x3
        x = F.relu(x)

        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet5(x)
        x = x1 + x2 + x3
        x = F.relu(x)

        # Project back to output space
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Output: gas saturation at each grid point and time

        return x
```

```python
# Training configuration and loss function
# Hyperparameters
mode1 = mode2 = mode3 = 10  # Fourier modes in each dimension
width = 36                   # Channel width
batch_size = 4
epochs = 100
learning_rate = 0.001
scheduler_step = 2           # LR decay every 2 epochs
scheduler_gamma = 0.9        # LR decay factor

# Model initialization
model = Net3d(mode1, mode2, mode3, width)

# Optimizer with L2 weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# Custom loss: Relative L2 + first derivative regularization
# This helps capture sharp saturation fronts
for x, y in train_loader:
    # Compute finite difference derivative in r direction
    dy = (y[:,:,2:,:] - y[:,:,:-2,:])/grid_dx

    # Active cell mask for variable reservoir thickness
    mask = (x[:,:,:,0:1,0]!=0).repeat(1,1,1,24)

    # Forward pass
    pred = model(x).view(-1,96,200,24)
    dy_pred = (pred[:,:,2:,:] - pred[:,:,:-2,:])/grid_dx

    # Original loss (masked to active cells)
    ori_loss = 0
    for i in range(batch_size):
        ori_loss += myloss(pred[i,...][mask[i,...]].reshape(1, -1),
                          y[i,...][mask[i,...]].reshape(1, -1))

    # First derivative loss (helps with front sharpness)
    der_loss = 0
    mask_dy = mask[:,:,:198,:]
    for i in range(batch_size):
        der_loss += myloss(dy_pred[i,...][mask_dy[i,...]].reshape(1, -1),
                          dy[i,...][mask_dy[i,...]].view(1, -1))

    # Combined loss with derivative weight of 0.5
    loss = ori_loss + 0.5 * der_loss

    loss.backward()
    optimizer.step()
```

## Critical Parameters

**Architecture parameters**:
- `modes1, modes2, modes3 = 10`: Number of Fourier modes retained in each spatial-temporal dimension. Higher values capture more high-frequency content but increase computation.
- `width = 36`: Hidden channel dimension. Balances model capacity and computational cost.
- Number of layers: 3 Fourier layers + 3 U-Fourier layers. This 50/50 split achieves optimal performance for multiphase flow.

**Training parameters**:
- `batch_size = 4`: Limited by GPU memory for 3D data
- `epochs = 100`: Training stops when validation loss plateaus
- `learning_rate = 0.001`: Initial learning rate
- `scheduler_step = 2, scheduler_gamma = 0.9`: Learning rate decays by 0.9 every 2 epochs
- `weight_decay = 1e-4`: L2 regularization on model weights
- `derivative_weight = 0.5`: Weight on first derivative loss term

**Loss function parameters**:
- Relative L2 loss provides scale-invariance for varying saturation magnitudes
- First derivative regularization weight = 0.5 improves sharp front prediction
- Active cell masking accommodates variable reservoir thickness

**Input channels (12 total)**:
- kr, kz: horizontal and vertical permeability
- porosity: rock porosity
- inj_loc: injection location (perforation map)
- inj_rate: CO2 injection rate
- pressure: initial reservoir pressure
- temperature: reservoir temperature
- Swi: irreducible water saturation
- Lam (λ): Van Genuchten capillary pressure scaling factor
- grid_x, grid_y, grid_t: spatial and temporal grid coordinates
