# hp-VPINN for Advection-Diffusion Equation (Inverse Problem)

**Keywords**: [PDE, parabolic, linear, inverse-problem, parameter-estimation, advection-diffusion, 1D, dirichlet, PINN, weak-form, domain-decomposition, MLP, adam, mse, tensorflow]

**Problem:** Solving time-dependent advection-diffusion equations with unknown diffusivity coefficients using sparse observational data. The challenge is to simultaneously learn the PDE solution and identify unknown physical parameters (diffusivity κ) from limited measurements at sensor locations, requiring efficient parameter estimation in the presence of data sparsity.

**Issues addressed:**
- Parameter identification in inverse problems with limited observational data
- Efficient training for time-dependent PDEs through domain decomposition
- Balancing accuracy between interior solution and boundary conditions

## Key Method

hp-VPINN (hp-Variational Physics-Informed Neural Networks) combines **variational formulation** with **domain decomposition** for solving PDEs. The method uses:

1. **Global Neural Network Trial Function**: A single neural network u_NN(x,t) approximates the solution over the entire space-time domain
2. **Local Polynomial Test Functions**: Piecewise orthogonal polynomials (Legendre-based) defined on non-overlapping sub-domains
3. **Petrov-Galerkin Formulation**: Projects PDE residuals onto test functions via integration, forming a variational loss

**Key Innovation - hp-Refinement**:
- **h-refinement**: Domain decomposition into multiple elements for localized learning
- **p-refinement**: High-order polynomial test functions in each element for accuracy

**Variational Loss for Advection-Diffusion**:
The method minimizes: L = L_boundary + L_variational + L_data

where the variational residual in each element e is:
```
R^(e) = ∫∫_{Ω_e} (∂u_NN/∂t + v·∂u_NN/∂x - κ·∂²u_NN/∂x²) v_k(x,t) dx dt
```

For inverse problems, an additional data term enforces sparse measurements:
```
L_data = (1/N*) Σ |u_NN(x*_i, t*_i) - u*_i|²
```

The unknown diffusivity κ is a trainable parameter optimized alongside network weights.

**Advantages**:
- Localized learning through domain decomposition reduces training complexity
- High-order accuracy from polynomial test functions
- Natural handling of variational forms through integration by parts
- Parameter identification capability for inverse problems

## Implementation

### Core VPINN Class with Domain Decomposition

```python
class VPINN:
    def __init__(self, XT_u_train, u_train, XT_f_train, XT_quad, W_quad,
                 T_quad, WT_quad, grid_x, grid_t, N_testfcn, XT_test, u_test, layers, lb, ub):

        # Trainable diffusivity parameter for inverse problem
        self.epsilon = tf.Variable(1*tf.ones([1], dtype=tf.float64), dtype=tf.float64)

        # Training data
        self.x = XT_u_train[:,0:1]  # Boundary/initial x coordinates
        self.t = XT_u_train[:,1:2]  # Boundary/initial t coordinates
        self.u = u_train             # Boundary/initial conditions

        # Quadrature points and weights for integration
        self.xquad = XT_quad[:,0:1]
        self.tquad = XT_quad[:,1:2]
        self.wquad = W_quad

        # Element information
        self.Nelementx = np.size(N_testfcn[0])  # Number of elements in x
        self.Nelementt = np.size(N_testfcn[1])  # Number of elements in t

        # Initialize neural network
        self.weights, self.biases, self.a = self.initialize_NN(layers)

        # Variational loss computation over all elements
        self.varloss_total = 0
        for ex in range(self.Nelementx):
            for et in range(self.Nelementt):
                Ntest_elementx = N_testfcn[0][ex]
                Ntest_elementt = N_testfcn[1][et]

                # Jacobian for coordinate transformation
                jacobian = (grid_t[et+1]-grid_t[et])/2 * (grid_x[ex+1]-grid_x[ex])/2
                jacobian_x = (grid_x[ex+1]-grid_x[ex])/2

                # Map quadrature points to physical element
                x_quad_element = tf.constant(grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(self.xquad+1),
                                           dtype=tf.float64)
                t_quad_element = tf.constant(grid_t[et] + (grid_t[et+1]-grid_t[et])/2*(self.tquad+1),
                                           dtype=tf.float64)

                # Neural network solution and derivatives at quadrature points
                u_NN_quad_element = self.net_u(x_quad_element, t_quad_element)
                d1xu_NN_quad_element, d2xu_NN_quad_element = self.net_dxu(x_quad_element, t_quad_element)
                d1tu_NN_quad_element = self.net_dtu(x_quad_element, t_quad_element)

                # Test functions at quadrature points
                testx_quad_element = self.Test_fcn(Ntest_elementx, self.xquad)
                testt_quad_element = self.Test_fcn(Ntest_elementt, self.tquad)

                # Variational form: strong form (no integration by parts)
                if var_form == 0:
                    U_NN_element = tf.convert_to_tensor([[
                        jacobian*tf.reduce_sum(
                            self.wquad[:,0:1]*testx_quad_element[r]*
                            self.wquad[:,1:2]*testt_quad_element[k]*
                            (d1tu_NN_quad_element + V*d1xu_NN_quad_element - self.epsilon*d2xu_NN_quad_element)
                        ) for r in range(Ntest_elementx)] for k in range(Ntest_elementt)], dtype=tf.float64)

                # Variational form: weak form (once integration by parts)
                if var_form == 1:
                    d1testx_quad_element, d2testx_quad_element = self.dTest_fcn(Ntest_elementx, self.xquad)
                    U_NN_element = tf.convert_to_tensor([[
                        jacobian*tf.reduce_sum(self.wquad[:,0:1]*testx_quad_element[r]*
                                             self.wquad[:,1:2]*testt_quad_element[k]*
                                             (d1tu_NN_quad_element + V*d1xu_NN_quad_element))
                        + self.epsilon*jacobian/jacobian_x*tf.reduce_sum(
                            self.wquad[:,0:1]*d1testx_quad_element[r]*
                            self.wquad[:,1:2]*testt_quad_element[k]*d1xu_NN_quad_element)
                        for r in range(Ntest_elementx)] for k in range(Ntest_elementt)], dtype=tf.float64)

                # Flatten and compute element loss
                Res_NN_element = tf.reshape(U_NN_element, [1,-1])
                loss_element = tf.reduce_mean(tf.square(Res_NN_element))
                self.varloss_total = self.varloss_total + loss_element

        # Total loss: boundary + variational
        self.lossb = 10*tf.reduce_mean(tf.square(self.u_tf - self.u_NN_pred))
        self.lossv = self.varloss_total
        self.loss = self.lossb + self.lossv
```

### Test Functions (Modified Legendre Polynomials)

```python
def Test_fcn(self, N_test, x):
    """
    Construct test functions using Jacobi polynomials.
    Test functions satisfy homogeneous boundary conditions:
    v_k(x) = P_{k+1}(x) - P_{k-1}(x)
    where P_n is the Legendre polynomial of degree n
    """
    test_total = []
    for n in range(1, N_test+1):
        # Modified Legendre polynomials vanishing at boundaries
        test = Jacobi(n+1, 0, 0, x) - Jacobi(n-1, 0, 0, x)
        test_total.append(test)
    return np.asarray(test_total)

def dTest_fcn(self, N_test, x):
    """
    Compute first and second derivatives of test functions.
    Required for weak formulations with integration by parts.
    """
    d1test_total = []
    d2test_total = []
    for n in range(1, N_test+1):
        if n == 1:
            d1test = ((n+2)/2)*Jacobi(n, 1, 1, x)
            d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x)
        elif n == 2:
            d1test = ((n+2)/2)*Jacobi(n, 1, 1, x) - ((n)/2)*Jacobi(n-2, 1, 1, x)
            d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x)
        else:
            d1test = ((n+2)/2)*Jacobi(n, 1, 1, x) - ((n)/2)*Jacobi(n-2, 1, 1, x)
            d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x) - ((n)*(n+1)/(2*2))*Jacobi(n-3, 2, 2, x)
        d1test_total.append(d1test)
        d2test_total.append(d2test)
    return np.asarray(d1test_total), np.asarray(d2test_total)
```

### Neural Network Architecture

```python
def neural_net(self, X, weights, biases, a):
    """
    Standard fully connected neural network with tanh activation.
    X: input features [x, t]
    """
    num_layers = len(weights) + 1
    H = X
    # Forward pass through hidden layers
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    # Output layer (linear)
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y
```

### Inverse Problem Setup

```python
# Interior sensor measurements (sparse observations)
NPu_inter = 5  # Number of measurements per sensor
# Three sensor locations: x = -0.5, 0.0, 0.5
x_inter_1 = np.empty(NPu_inter)[:,None]
x_inter_1.fill(-0.5)
t_inter_1 = T*lhs(1, NPu_inter)  # Random times at sensor 1

x_inter_2 = np.empty(NPu_inter)[:,None]
x_inter_2.fill(0.0)
t_inter_2 = T*lhs(1, NPu_inter)  # Random times at sensor 2

x_inter_3 = np.empty(NPu_inter)[:,None]
x_inter_3.fill(0.5)
t_inter_3 = T*lhs(1, NPu_inter)  # Random times at sensor 3

# Concatenate all sensor data
xu_inter = np.concatenate([x_inter_1, x_inter_2, x_inter_3])
timeu_inter = np.concatenate([t_inter_1, t_inter_2, t_inter_3])
XT_u_inter_train = np.hstack((xu_inter, timeu_inter))

# Compute exact solution at sensor locations
u_inter_train = np.asarray([u_ext(XT_u_inter_train[i,0], XT_u_inter_train[i,1])
                           for i in range(XT_u_inter_train.shape[0])]).flatten()[:,None]

# Combine boundary/initial conditions with interior measurements
XT_u_train = np.concatenate((x_up_train, x_lo_train, x_in_train, XT_u_inter_train))
u_train = np.concatenate((u_up_train, u_lo_train, u_in_train, u_inter_train))
```

### Training Loop

```python
def train(self, nIter, tresh):
    tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
               self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
               self.x_quad: self.xquad, self.t_quad: self.tquad,
               self.x_test: self.xtest, self.t_test: self.ttest}

    for it in range(nIter):
        self.sess.run(self.train_op_Adam, tf_dict)

        if it % 10 == 0:
            loss_value = self.sess.run(self.loss, tf_dict)
            epsilon_value = self.sess.run(self.epsilon, tf_dict)  # Current diffusivity estimate

            if loss_value < tresh:
                print('It: %d, Loss: %.3e' % (it, loss_value))
                break

        if it % 100 == 0:
            print('It: %d, Lossv: %.3e, Lossb: %.3e, epsilon: %.4f' %
                  (it, loss_valuev, loss_valueb, epsilon_value))
```

## Critical Parameters

1. **Network architecture**
   - Layers: [2, 5, 5, 5, 1] (2 inputs → 3 hidden layers with 5 neurons → 1 output)
   - Activation: tanh
   - Input: (x, t) coordinates
   - Output: u(x, t) solution

2. **Domain decomposition**
   - N_el_x: 1 (number of elements in spatial direction)
   - N_el_t: 1 (number of elements in temporal direction)
   - Domain: x ∈ [-1, 1], t ∈ [0, 1]
   - Can be increased for h-refinement to handle sharp gradients

3. **Test functions (p-refinement)**
   - N_test_x: 5 test functions per element in x-direction
   - N_test_t: 5 test functions per element in t-direction
   - Type: Modified Legendre polynomials satisfying homogeneous BCs
   - Higher orders increase accuracy but also computational cost

4. **Quadrature**
   - N_quad: 10 Gauss-Lobatto quadrature points per direction per element
   - Total integration points per element: 10 × 10 = 100
   - Critical for accurate evaluation of variational integrals

5. **Variational formulation**
   - var_form = 0: Strong form (no integration by parts)
   - var_form = 1: Weak form (once integration by parts in spatial derivative)
   - Weak form reduces regularity requirements on neural network

6. **Training parameters**
   - Optimizer: Adam with learning rate 0.001
   - Iterations: 1500
   - Boundary loss weight: 10
   - Ensures accurate satisfaction of boundary/initial conditions

7. **Inverse problem specifics**
   - Unknown parameter: κ (diffusivity coefficient)
   - Initialization: κ = 1.0
   - True value: κ = 0.1/π ≈ 0.0318
   - Sensor measurements: 15 total (5 measurements × 3 sensors)
   - Sensor locations: x = -0.5, 0.0, 0.5

8. **Physical parameters**
   - Advection velocity (known): v = 1.0
   - Initial condition: u(x, 0) = -sin(πx)
   - Boundary conditions: u(±1, t) = 0 (homogeneous Dirichlet)
