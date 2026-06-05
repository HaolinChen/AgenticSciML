# Locally Adaptive Activation Functions with Slope Recovery

**Keywords**: [MLP, ResNet, self-adaptive, adam, sgd, mse, cross-entropy, pytorch]

**Problem:** Physics-informed neural networks (PINNs) and deep neural networks often suffer from slow convergence during training. While globally adaptive activation functions (with a single trainable slope parameter for the entire network) improve convergence, they do not exploit the different learning capacities of individual layers and neurons. Each hidden layer and neuron may benefit from having its own adaptive slope parameter, providing additional degrees of freedom to increase the network's learning capacity.

**Issues addressed:**
- Slow convergence in neural network training, especially in early training stages
- Suboptimal learning capacity when using a single global adaptive parameter
- Inefficient exploration of loss landscape across different network layers
- Limited flexibility in adapting activation functions to layer-specific features

## Key Method

The method introduces **locally adaptive activation functions** at two levels of granularity:

**1. Layer-wise Locally Adaptive Activation Functions (L-LAAF)**

Each hidden layer k has its own trainable slope parameter a^k:

**σ(na^k · L_k(z))**

where:
- a^k is a trainable scalar parameter for layer k
- n ≥ 1 is a fixed scaling factor
- L_k(z) is the affine transformation at layer k
- Adds D-1 parameters (one per hidden layer)

**2. Neuron-wise Locally Adaptive Activation Functions (N-LAAF)**

Each neuron i in hidden layer k has its own trainable slope parameter a^k_i:

**σ(na^k_i · (L_k(z))_i)**

where:
- a^k_i is a trainable scalar parameter for neuron i in layer k
- Adds Σ^(D-1)_(k=1) N_k parameters (one per neuron in all hidden layers)
- Acts as a vector activation function in each layer

**Slope Recovery Term**

To accelerate the increase of activation slopes during training, a slope recovery term S(a) is added to the loss function:

For L-LAAF:
S(a) = 1 / [1/(D-1) Σ^(D-1)_(k=1) exp(a^k)]

For N-LAAF:
S(a) = 1 / [1/(D-1) Σ^(D-1)_(k=1) exp(Σ^(N_k)_(i=1) a^k_i / N_k)]

This term forces the network to quickly increase activation slopes, thereby accelerating convergence.

**Key advantages:**
- Provides layer-specific or neuron-specific adaptive slopes
- Increases learning capacity by adding local degrees of freedom
- Accelerates convergence through slope recovery mechanism
- Negligible computational overhead (6.77% parameter increase for typical network)
- Modifies gradient dynamics by implicitly multiplying conditioning matrices

**Initialization:** na^k_i = 1 for all parameters to ensure stable initial conditions

## Implementation

### LeNet with Locally Adaptive Activation Functions

```python
class LeNet(nn.Module):
    def __init__(self, nc, nh, hw, num_classes, act_func, adapt=0, n=1.0):
        """
        LeNet architecture with adaptive activation support

        nc: number of input channels
        nh, hw: input height and width
        num_classes: number of output classes
        act_func: base activation function (e.g., ReLU, Sigmoid)
        adapt: adaptation mode (0=fixed, 1=global, 2=layer-wise, 3=neuron-wise)
        n: scaling factor for adaptive parameters
        """
        input_shape = (nc, nh, hw)
        super(LeNet, self).__init__()

        # Convolutional layers
        self.maxpool = nn.MaxPool2d((2, 2))
        self.conv1 = nn.Conv2d(nc, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)

        # Calculate flattened shape after convolutions
        self.flat_shape, shape1, shape2 = self.get_flat_shape(input_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_shape, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.act_func = act_func

        # Initialize adaptive parameters based on adaptation mode
        self.adapt = adapt
        self.n = n

        if self.adapt == 1:
            # Global adaptive: single parameter for entire network
            self.adaptive = nn.Parameter(torch.tensor(1.0/self.n))

        elif self.adapt == 2:
            # Layer-wise adaptive: one parameter per layer
            self.adaptive1 = nn.Parameter(torch.tensor(1.0/self.n))  # Conv1
            self.adaptive2 = nn.Parameter(torch.tensor(1.0/self.n))  # Conv2
            self.adaptive3 = nn.Parameter(torch.tensor(1.0/self.n))  # FC1

        elif self.adapt == 3:
            # Neuron-wise adaptive: one parameter per neuron/channel
            self.adaptive1 = nn.Parameter(torch.ones(shape1)/self.n)  # Conv1 output
            self.adaptive2 = nn.Parameter(torch.ones(shape2)/self.n)  # Conv2 output
            self.adaptive3 = nn.Parameter(torch.ones(1024)/self.n)    # FC1 output

    def get_flat_shape(self, input_shape):
        """Calculate output shapes after convolution and pooling operations"""
        dummy0 = Variable(torch.zeros(1, *input_shape))
        dummy1 = self.maxpool(self.conv1(dummy0))
        dummy2 = self.maxpool(self.conv2(dummy1))
        return dummy2.data.view(1, -1).size(1), dummy1.data.size(), dummy2.data.size()

    def forward(self, x_in):
        """
        Forward pass with adaptive activation
        Applies n*a*x before activation function where:
        - a is the adaptive parameter (global, layer-wise, or neuron-wise)
        - n is the fixed scaling factor
        """
        # Conv layer 1
        x = self.conv1(x_in)
        x = self.maxpool(x)

        # Apply adaptive scaling before activation
        if self.adapt == 1:
            x = self.n * self.adaptive * x
        elif self.adapt == 2 or self.adapt == 3:
            x = self.n * self.adaptive1 * x

        x = self.act_func(x)

        # Conv layer 2
        x = self.conv2(x)
        x = self.maxpool(x)

        if self.adapt == 1:
            x = self.n * self.adaptive * x
        elif self.adapt == 2 or self.adapt == 3:
            x = self.n * self.adaptive2 * x

        x = self.act_func(x)

        # Flatten
        x = x.view(-1, self.flat_shape)

        # FC layer 1
        x = self.fc1(x)

        if self.adapt == 1:
            x = self.n * self.adaptive * x
        elif self.adapt == 2 or self.adapt == 3:
            x = self.n * self.adaptive3 * x

        x = self.act_func(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # FC layer 2 (output)
        x_out = self.fc2(x)

        return x_out
```

### PreActResNet Block with Locally Adaptive Activation

```python
class PreActBlock(nn.Module):
    """Pre-activation ResNet block with adaptive activation support"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act_func=torch.nn.ReLU(),
                 adapt=0, adaptive=0, n=1.0, layer=2, counter=0):
        """
        in_planes: number of input channels
        planes: number of output channels
        stride: stride for convolution
        act_func: base activation function
        adapt: adaptation mode (0=fixed, 1=global, 2=layer-wise, 3=neuron-wise)
        adaptive: shared global adaptive parameter (for adapt=1)
        n: scaling factor
        layer: which layer group this block belongs to (1-4)
        counter: block index within the layer group
        """
        super(PreActBlock, self).__init__()

        # Batch normalization and convolution layers
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # Shortcut connection for residual
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                         kernel_size=1, stride=stride, bias=False)
            )

        self.act_func = act_func
        self.adapt = adapt
        self.n = n

        if self.adapt == 1:
            # Global adaptive: use shared parameter
            self.adaptive = adaptive

        elif self.adapt == 2:
            # Layer-wise adaptive: separate parameter for each activation in block
            self.adaptive1 = nn.Parameter(torch.tensor(1.0/self.n))
            self.adaptive2 = nn.Parameter(torch.tensor(1.0/self.n))

        elif self.adapt == 3:
            # Neuron-wise adaptive: parameters match spatial dimensions
            # Different sizes based on layer group and position
            if layer == 1:
                self.adaptive1 = nn.Parameter(torch.ones(64, 32, 32)/self.n)
                self.adaptive2 = nn.Parameter(torch.ones(64, 32, 32)/self.n)
            elif layer == 2:
                self.adaptive1 = nn.Parameter(torch.ones(128, 16, 16)/self.n)
                self.adaptive2 = nn.Parameter(torch.ones(128, 16, 16)/self.n)
                if counter == 0:  # First block has different input size
                    self.adaptive1 = nn.Parameter(torch.ones(64, 32, 32)/self.n)
            elif layer == 3:
                self.adaptive1 = nn.Parameter(torch.ones(256, 8, 8)/self.n)
                self.adaptive2 = nn.Parameter(torch.ones(256, 8, 8)/self.n)
                if counter == 0:
                    self.adaptive1 = nn.Parameter(torch.ones(128, 16, 16)/self.n)
            elif layer == 4:
                self.adaptive1 = nn.Parameter(torch.ones(512, 4, 4)/self.n)
                self.adaptive2 = nn.Parameter(torch.ones(512, 4, 4)/self.n)
                if counter == 0:
                    self.adaptive1 = nn.Parameter(torch.ones(256, 8, 8)/self.n)

    def forward(self, x):
        """Forward pass with pre-activation and adaptive slopes"""
        # First activation (pre-activation before first conv)
        if self.adapt == 1:
            out = self.act_func(self.adaptive * self.bn1(x))
        elif self.adapt == 2 or self.adapt == 3:
            out = self.act_func(self.adaptive1 * self.bn1(x))
        else:
            out = self.act_func(self.bn1(x))

        # Shortcut connection
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x

        # First convolution
        out = self.conv1(out)

        # Second activation and convolution
        if self.adapt == 1:
            out = self.conv2(self.act_func(self.adaptive * self.bn2(out)))
        elif self.adapt == 2 or self.adapt == 3:
            out = self.conv2(self.act_func(self.adaptive2 * self.bn2(out)))
        else:
            out = self.conv2(self.act_func(self.bn2(out)))

        # Add residual connection
        out += shortcut
        return out
```

### Training Loop with Slope Recovery

```python
def train(epoch):
    """
    Training function with slope recovery regularization
    """
    global optimizer
    model.train()

    # Learning rate scheduling
    optimizer = lr_scheduler(optimizer, epoch)

    for batch_idx, (x, y) in enumerate(train_loader):
        x = Variable(x.cuda())
        y = Variable(y.cuda())

        # Forward pass
        h1 = model(x)

        # Standard cross-entropy loss
        loss = F.cross_entropy(h1, y)

        # Add slope recovery regularization
        reg = 0
        counter = 0
        for name, param in model.named_parameters():
            if 'adaptive' in name:
                counter += 1
                # Compute average exponential of adaptive parameters
                reg += torch.exp(torch.sum(param) / param.numel())

        # Add inverse of exponential mean as regularization
        # This encourages adaptive parameters to increase
        if reg != 0 and counter != 0:
            loss += 1 / (reg / counter)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Main Training Script

```python
# Activation function selection
if args.activation == 'relu':
    act_func = torch.nn.ReLU()
elif args.activation == 'softplus':
    act_func = torch.nn.Softplus(beta=1000, threshold=20)
elif args.activation == 'sigmoid':
    act_func = torch.nn.Sigmoid()

# Model instantiation with adaptive activation
if args.model == 'LeNet':
    model = LeNet(nc, nh, nw, num_class, act_func,
                  adapt=args.method, n=args.n).cuda()
elif args.model == 'PreActResNet18':
    model = PreActResNet18(nc, num_class, act_func,
                           adapt=args.method, n=args.n).cuda()

# Optimizer setup
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

# Training loop
for epoch in range(end_epoch):
    train(epoch)

    # Evaluate on train and test sets
    train_loss, train_acc, _ = output(train_loader)
    test_loss, test_acc, extra_neuron = output(test_loader)
```

## Critical Parameters

1. **Adaptation mode (adapt)**
   - 0: Fixed activation (no adaptation)
   - 1: Global adaptive (single parameter for entire network)
   - 2: Layer-wise adaptive (one parameter per layer) - L-LAAF
   - 3: Neuron-wise adaptive (one parameter per neuron/channel) - N-LAAF
   - Effect: Controls granularity of adaptive activation

2. **Scaling factor (n)**
   - Typical values: 1.0 to 10.0
   - Purpose: Scales the adaptive parameters to control sensitivity
   - Initialization: na = 1 to ensure stable start
   - Note: Critical scaling factor n_crit exists above which optimization becomes sensitive

3. **Adaptive parameter initialization**
   - Initial value: a = 1.0/n (so that na = 1 initially)
   - Trainable: Yes, optimized alongside weights and biases
   - Typical learned values vary by layer/neuron
   - Effect: Dynamically adjusts activation function slope during training

4. **Learning rate**
   - Initial value: 0.01
   - Momentum: 0.9
   - Schedule: Divided by 10 at epoch 10 (without data augmentation) or epochs 10, 100 (with data augmentation)
   - Critical for stable optimization of adaptive parameters

5. **Slope recovery weight**
   - Implicit weight in loss: 1 / (average_exponential_of_adaptive_parameters)
   - Effect: Encourages rapid increase of adaptive parameters
   - Accelerates convergence in early training stages

6. **Network architecture**
   - LeNet: 2 conv layers (64 filters each) + 2 FC layers (1024, num_classes)
   - PreActResNet18: [2,2,2,2] blocks with 64, 128, 256, 512 channels
   - Activation: ReLU, Softplus, or Sigmoid
   - Dropout: 0.5 after first FC layer (LeNet)

7. **Training parameters**
   - Batch size: 64
   - Test batch size: 100
   - Weight decay: 0.0
   - Epochs: 50 (without augmentation), 200 (with augmentation)

8. **Parameter overhead**
   - L-LAAF: D-1 additional parameters (D = number of layers)
   - N-LAAF: Σ^(D-1)_(k=1) N_k additional parameters
   - Example: 3-layer network with 20 neurons each → 6.77% increase
   - Overhead decreases with deeper/wider networks
