# Control Theory Notation Reference

This reference sheet provides an overview of common mathematical symbols, operators, and
notational conventions used in control theory, systems engineering, and estimation. It is
intended for graduate students and researchers as a quick reference during study and
analysis. The notation is largely consistent with seminal texts in classical, modern,
optimal, and robust control.

---

```rust,editable
fn adder<T: core::ops::Add<Output = T>>(a: T, b: T) -> T { a + b }

fn main() {
    let sum = adder(1, 2);
    println!("{sum}")
}
```

## Sets and Spaces

This section covers the fundamental sets and function spaces used to define signals,
systems, and their properties.

| Symbol | Meaning / Description | Context / Example |
| :--- | :--- | :--- |
| $\mathbb{R}, \mathbb{C}$ | The set of real and complex numbers, respectively. | A pole is located at $s = -a + jb$, where $s \in \mathbb{C}$. |
| $\mathbb{R}^n$ | The space of $n$-dimensional vectors with real entries. | The state vector $x$ belongs to $\mathbb{R}^n$, written as $x \in \mathbb{R}^n$. |
| $\mathbb{R}^{n \times m}$ | The space of $n \times m$ matrices with real entries. | The system matrix $A$ is in $\mathbb{R}^{n \times n}$, written as $A \in \mathbb{R}^{n \times n}$. |
| $s, z$ | The complex variables for the Laplace and Z-transforms. | Transfer function $G(s)$, pulse transfer function $G(z)$. |
| RHP, LHP | Right-Half Plane and Left-Half Plane of the complex plane. | For stability, poles must be in the LHP: $\Re(s) < 0$. |
| $\mathcal{L}_2, \mathcal{L}_\infty$ | Lebesgue spaces. $\mathcal{L}_2$ is the space of finite-energy signals. $\mathcal{L}_\infty$ is the space of bounded-amplitude signals. | Signal energy is the $\mathcal{L}_2$ norm: $\|u(t)\|_2^2 = \int_0^\infty \|u(t)\|^2 dt < \infty$. |
| $\mathcal{H}_2, \mathcal{H}_\infty$ | Hardy spaces. $\mathcal{H}_2$ and $\mathcal{H}_\infty$ are spaces of stable, proper transfer functions. | The $\mathcal{H}_\infty$ norm of a system $G(s)$ is its maximum gain over all frequencies: $\|G(s)\|_\infty = \sup_\omega \|G(j\omega)\|$. |

---

## Vectors and Matrices

Notation for linear algebra is central to the state-space representation of systems.

| Symbol | Meaning / Description | Context / Example |
| :--- | :--- | :--- |
| $x, u, y$ | Lowercase letters for column vectors (state, input, output). | The state vector is $x = [x_1, x_2, \dots, x_n]^T$. |
| $A, B, C, D$ | Uppercase letters for matrices. | The state-space representation: $\dot{x} = Ax + Bu$. |
| $A^T, A^*$ | Transpose and conjugate transpose (Hermitian) of matrix $A$. | The solution to the LQR ARE involves $A^T P$. |
| $A^{-1}, A^\dagger$ | Inverse and Moore-Penrose pseudoinverse of matrix $A$. | The input for a desired state is $u = B^\dagger(\dot{x}-Ax)$. |
| $\det(A), \text{tr}(A)$ | Determinant and trace of matrix $A$. | The characteristic equation is $\det(sI - A) = 0$. |
| $\lambda_i(A)$ | The $i$-th eigenvalue of matrix $A$. | The system poles are the eigenvalues of the state matrix $A$. |
| $\sigma_i(A)$ | The $i$-th singular value of matrix $A$. | Robustness is often analyzed using singular values. |
| $\bar{\sigma}(A), \underline{\sigma}(A)$ | Maximum and minimum singular values of $A$. | The $H_\infty$ norm is the peak singular value: $\|G(s)\|_\infty = \sup_\omega \bar{\sigma}(G(j\omega))$. |
| $A \succ 0, A \succeq 0$ | $A$ is a symmetric positive definite / positive semidefinite matrix. | A Lyapunov function candidate $V(x)=x^T P x$ requires $P \succ 0$. |
| $I$ | The identity matrix. | Sized according to context, e.g., $Ix = x$. |
| $\|x\|_p$ | The $p$-norm of a vector $x$. | Euclidean norm: $\|x\|_2 = \sqrt{x^T x}$. |
| $\|A\|$ | A matrix norm (e.g., induced 2-norm, Frobenius norm). | Used to quantify the "gain" of a linear system. |
| $\otimes$ | The Kronecker product. | Used in solving Sylvester equations, e.g., $AX+XB=C$. |

---

## System Dynamics and Signals

These symbols describe the behavior of dynamical systems and the signals that they process.

| Symbol | Meaning / Description | Context / Example |
| :--- | :--- | :--- |
| $t, k$ | Continuous time variable and discrete time index. | $x(t)$ for continuous-time state, $x[k]$ or $x_k$ for discrete-time. |
| $\dot{x}, \ddot{x}$ | First and second time derivatives of $x$. | Newton's law: $F = m\ddot{x}$. State-space: $\dot{x} = Ax+Bu$. |
| $x(t), u(t), y(t)$ | Continuous-time state, input, and output signals. | The output is a function of the state and input: $y(t) = Cx(t)+Du(t)$. |
| $G(s), P(s)$ | Transfer function or matrix (often for the plant). | $Y(s) = G(s)U(s)$. |
| $K(s), C(s)$ | Transfer function for the controller. | The control law is $U(s) = K(s)E(s)$. |
| $L(s)$ | Loop transfer function. | $L(s) = P(s)K(s)$. |
| $S(s), T(s)$ | Sensitivity and complementary sensitivity functions. | $S = (I+L)^{-1}$, $T = (I+L)^{-1}L$. Note that $S+T=I$. |
| $\mathcal{C}, \mathcal{O}$ | Controllability and observability matrices. | $\mathcal{C} = [B \ AB \ A^2B \ \dots \ A^{n-1}B]$. System is controllable if $\text{rank}(\mathcal{C}) = n$. |
| $\zeta, \omega_n$ | Damping ratio and natural frequency. | Second-order system poles: $s^2 + 2\zeta\omega_n s + \omega_n^2 = 0$. |
| $GM, PM$ | Gain Margin and Phase Margin. | Key frequency-domain measures of relative stability. |
| $\ast$ | Convolution operator. | The output is the convolution of input and impulse response: $y(t) = h(t) \ast u(t)$. |
| $\mathcal{L}\{\cdot\}, \mathcal{Z}\{\cdot\}$ | Laplace transform and Z-transform operators. | $\mathcal{L}\{\dot{f}(t)\} = sF(s) - f(0)$. |

---

## Optimization and Optimal Control

This section covers notation common in optimization problems, particularly within optimal
control frameworks like LQR and MPC.

| Symbol | Meaning / Description | Context / Example |
| :--- | :--- | :--- |
| $J$ | Cost functional or objective function to be minimized. | LQR cost: $J = \int_0^\infty (x^T Q x + u^T R u) dt$. |
| $Q, R, P$ | Weighting matrices in cost functions; $P$ is the solution to a Riccati equation. | $Q \succeq 0$ penalizes state deviation, $R \succ 0$ penalizes control effort. |
| $\mathcal{H}$ | The Hamiltonian, used in applying Pontryagin's Minimum Principle. | $\mathcal{H}(x, u, p) = L(x,u) + p^T f(x,u)$. |
| $p(t), \lambda(t)$ | Costate (or adjoint) vector. | Adjoint dynamics: $\dot{p} = -\frac{\partial \mathcal{H}}{\partial x}$. |
| ARE, DRE | Algebraic Riccati Equation and Differential Riccati Equation. | The ARE for LQR is $A^T P + PA - PBR^{-1}B^T P + Q = 0$. |
| $\nabla f, \nabla^2 f$ | Gradient and Hessian of a scalar function $f$. | The optimal control satisfies $\frac{\partial \mathcal{H}}{\partial u} = 0$. |
| LMI | Linear Matrix Inequality. | Many control problems can be convexly formulated as LMIs: $F(x) = F_0 + \sum_{i=1}^m x_i F_i \succ 0$. |

---

## Probability and Estimation

This notation is fundamental to stochastic systems, filtering, and state estimation.

| Symbol | Meaning / Description | Context / Example |
| :--- | :--- | :--- |
| $\hat{x}$ | An estimate of the state vector $x$. | The output of a Kalman filter is the state estimate $\hat{x}$. |
| $\tilde{x}$ | The estimation error. | $\tilde{x} = x - \hat{x}$. |
| $w(t), v(t)$ | Process noise and measurement noise. | Stochastic system: $\dot{x} = Ax + Bu + w$; $y = Cx + v$. |
| $Q, R$ | Covariance matrices of process noise ($w$) and measurement noise ($v$). | These are design parameters in a Kalman filter, representing noise strength. |
| $P, \Sigma$ | State error covariance matrix. | The Kalman filter propagates $P_k = E[\tilde{x}_k \tilde{x}_k^T]$. |
| $K$ or $K_f$ | Kalman gain. | The state update equation is $\hat{x}_{k\|k} = \hat{x}_{k\|k-1} + K_k(y_k - C\hat{x}_{k\|k-1})$. |
| $E[\cdot]$ or $\mathbb{E}[\cdot]$ | Expected value operator. | Noise is often assumed to be zero-mean: $E[w(t)] = 0$. |
| $\mathcal{N}(\mu, \Sigma)$ | A Gaussian (normal) probability distribution with mean $\mu$ and covariance $\Sigma$. | It is often assumed that $w \sim \mathcal{N}(0, Q)$ and $v \sim \mathcal{N}(0, R)$. |
| $p(x \| y)$ | Conditional probability density function of $x$ given $y$. | Bayesian estimation aims to find $p(x_k \| y_{1:k})$. |

---

## Robust Control

These symbols are specific to the analysis and design of controllers for systems with uncertainty.

| Symbol | Meaning / Description | Context / Example |
| :--- | :--- | :--- |
| $\Delta$ | A norm-bounded operator representing system uncertainty. | True plant model $P = P_0(I + W\Delta)$, where $\|\Delta\|_\infty \leq 1$. |
| $W_p, W_u, W_T$ | Weighting functions used to shape performance and robustness objectives. | To minimize sensitivity, we make $\|W_p S\|_\infty$ small. |
| $\mathcal{F}_u(M, \Delta), \mathcal{F}_l(M, \Delta)$ | Upper and Lower Linear Fractional Transformations (LFTs). | Used to pull out uncertainty $\Delta$ from an interconnected system $M$.  |
| $\gamma$ | The performance level in $H_\infty$ synthesis. | The goal is to find a stabilizing controller $K$ such that $\|\mathcal{F}_l(P,K)\|_\infty < \gamma$. |
| $\mu(M)$ | The structured singular value (SSV), a tool for analyzing robustness to structured uncertainty. | The robust stability condition is $\mu_{\Delta}(M) < 1$. |
