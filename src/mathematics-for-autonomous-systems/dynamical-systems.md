# Dynamical Systems

> &emsp;&emsp;_"In short, a system is any process or entity that has one or more
> well-defined inputs and one or more well-defined outputs. Examples of systems
> include a simple physical object obeying Newtonian mechanics, and the US
> economy! Systems can be physical, or we may talk about a mathematical
> description of a system. The point of modeling is to capture in a mathematical
> representation the behavior of a physical system. As we will see, such
> representation lends itself to analysis and design, and certain restrictions
> such as linearity and time-invariance open a huge set of available tools."_ >
> [hover_system_2022]

&emsp;&emsp;Dynamical systems theory concerns the evolution of systems in time,
where the state of a system can be described by a finite number of parameters.
This framework applies to both continuous-time systems modeled by ordinary
differential equations (ODEs) and discrete-time systems modeled by state
machines or difference equations.

The focus of dynamical systems is to answer these central questions: What are
the equilibrium or time-periodic solutions of the dynamics? Are these solutions
stable controllable or observable? What is the long-time asymptotic behavior of
the solutions?

## 1. Modeling Fundamentals

**System State**: \\(\mathbf{x}(t) \in \mathbb{R}^n\\)

- Minimal set of independent variables fully characterizing the system at time
  \\(t\\).
- Given \\(\mathbf{x}(t_0)\\) and the system dynamics, the future and past
  behavior \\(\mathbf{x}(t)\\) can be predicted.
- The choice of state variables is **not unique**: multiple valid
  representations exist.
- Variables may be physical quantities measured directly (position, velocity,
  temperature) or abstract constructs that have no physical interpretation.

### Ordinary Differential Equations

#### Linear ODEs

\\[ \dot{\mathbf{x}} = A\mathbf{x} \\]

where \\(A \in \mathbb{R}^{n \times n}\\) is the system matrix and \\(\mathbf{x}
\in \mathbb{R}^n\\) is the state vector.

**Superposition**: Let \\(f\\) be a linear function, then \\[ f(x_1) + f(x_2) =
f(x_1 + x_2) \\] \\[ \alpha \cdot f(x) = f(\alpha \cdot x) \\]

##### Representing Higher-Order ODEs as First-Order Systems

Any higher-order ODE can be reformulated as a system of first-order ODEs by
introducing additional state variables. For the second-order equation: \\(
\ddot{y} + 5\dot{y} + 6y = u(t) \\) introduce state variables \\(x_1 = y\\) and
\\(x_2 = \dot{y}\\), yielding: \\[ \begin{bmatrix} \dot{x}_1 \\\\ \dot{x}_2
\end{bmatrix} = \begin{bmatrix} 0 & 1 \\\\ -6 & -5 \end{bmatrix} \begin{bmatrix}
x_1 \\\\ x_2 \end{bmatrix} + \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} u(t) \\]

_Note that \\(x_2=\dot{x_1}\\)_

#### Nonlinear ODEs

\\[\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})\\]

where \\(\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^n\\) is a nonlinear function.

### Equilibrium Points: Static Operating Modes

\\[ \mathbf{x}^* \in \mathbb{R}^n \\] \\[ \dot{\mathbf{x}}^* =
\mathbf{f}(\mathbf{x}^\*) = \mathbf{0} \\]

Linear systems have only the origin as equilibrium when \\(\det(A) \neq 0\\).
Nonlinear systems may have multiple equilibrium points.

#### Classification of Equilibrium Points

The stability of an equilibrium point can be determined by the eigenvalues of
the Jacobian matrix at that point. The classification is:

| Type          | Condition                                                                     | Behavior                                     |
| ------------- | ----------------------------------------------------------------------------- | -------------------------------------------- |
| Stable Node   | All \\(\text{Re}(\lambda_i) < 0\\)                                            | Trajectories converge to equilibrium         |
| Unstable Node | All \\(\text{Re}(\lambda_i) > 0\\)                                            | Trajectories diverge from equilibrium        |
| Saddle Point  | Mixed signs \\(\text{Re}(\lambda_i)\\)                                        | Stable in one direction, unstable in another |
| Focus/Spiral  | Complex \\(\lambda_i = \alpha \pm j\beta\\) with \\(\text{Re}(\lambda) < 0\\) | Spiral convergence with oscillation          |
| Center        | Purely imaginary \\(\lambda_i = \pm j\omega\\)                                | Closed orbits (conservative systems)         |

> # TODO
>
> Add time response vs pole zero map.

### Phase Portraits: Visual Analysis of System Trajectories

A **phase portrait** is a geometric representation of all possible trajectories
in the state space (phase space). For two-dimensional systems, the phase plane
shows the complete qualitative behavior without solving the ODEs explicitly.

#### Vector Field and Integral Curves

At each point \\(\mathbf{x}\\) in the phase space, the vector
\\(\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})\\) indicates the instantaneous
velocity of the trajectory. By plotting these velocity vectors at a grid of
points, we obtain the **vector field**. The integral curves (solution
trajectories) are curves that follow the vector field everywhere tangentially.

> # TODO
>
> add plot showing example phase portrait of a system.

## 2. Representations

### State-Space Form (Time Domain)

The **state-space representation** is the standard form for modern control
systems. It consists of two equations:

**State Equation (Dynamics):** \\( \dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}
\\)

**Output Equation (Measurement):** \\( \mathbf{y} = C\mathbf{x} + D\mathbf{u}
\\)

- \\(\mathbf{x} \in \mathbb{R}^n\\): state vector
- \\(\mathbf{u} \in \mathbb{R}^m\\): input vector (\\(m\\) inputs)
- \\(\mathbf{y} \in \mathbb{R}^p\\): output vector (\\(p\\) outputs)
- \\(A \in \mathbb{R}^{n \times n}\\): system matrix (determines open-loop
  dynamics and stability)
- \\(B \in \mathbb{R}^{n \times m}\\): input matrix (how controls influence each
  state)
- \\(C \in \mathbb{R}^{p \times n}\\): output matrix (which states are measured)
- \\(D \in \mathbb{R}^{p \times m}\\): feed-through matrix (direct coupling from
  input to output)

#### Advantages Over Transfer Functions

1. **Initial Conditions**: State-space naturally incorporates non-zero initial
   conditions \\(\mathbf{x}(0) = \mathbf{x}\_0\\).
2. **MIMO Systems**: Handles multiple inputs and outputs directly; transfer
   functions require transfer function matrices.
3. **State Feedback Control**: Enables direct state feedback design:
   \\(\mathbf{u} = -K\mathbf{x}\\).

### Transfer Functions (Frequency Domain)

The **transfer function** relates a system's output to the input through the
Laplace transform, assuming zero initial conditions:

\\[ G(s) = \frac{Y(s)}{U(s)} = \mathcal{L}\\{y(t)\\} / \mathcal{L}\\{u(t)\\} \\]

where \\(s \in \mathbb{C}\\) is the complex frequency variable.

#### Rational Function Form

Transfer functions are typically expressed as ratios of polynomials: \\[ G(s) =
\frac{b_m s^m + b_{m-1} s^{m-1} + \cdots + b_1 s + b_0}{a_n s^n + a_{n-1}
s^{n-1} + \cdots + a_1 s + a_0} \\]

| Relationship | Classification      | Limit \\(\omega \to \infty\\) | Physical Realizability                                                                                       |
| :----------- | :------------------ | :---------------------------- | :----------------------------------------------------------------------------------------------------------- |
| \\(n > m\\)  | **Strictly Proper** | Decay to 0                    | **Realizable.** Most physical systems fall here (inertial damping prevents instantaneous change).            |
| \\(n = m\\)  | **proper**          | Constant \\(k \neq 0\\)       | **Realizable.** Indicates a direct feed-through from input to output (e.g., a resistor network).             |
| \\(n < m\\)  | **Improper**        | Growth to \\(\infty\\)        | **Not Realizable.** Requires differentiation, which is non-causal in theory and amplifies noise in practice. |

> **Note on "Proper":** In general control theory, a system is considered
> **Proper** if \\(n \geq m\\) (covering both the first two rows). This is the
> condition required for a system to have a valid State-Space representation.

### Poles and Zeros: Frequency Domain Structure

\[ G(s) = \frac{N(s)}{D(s)} = K \frac{\prod*{i=1}^m (s - z_i)}{\prod*{j=1}^n
(s - p_j)} \]

\[ \mathcal{Z} = \{ s \in \mathbb{C} \mid N(s) = 0 \} \implies \forall z \in
\mathcal{Z}, G(z) = 0 \]

\[ \mathcal{P} = \{ s \in \mathbb{C} \mid D(s) = 0 \} \implies \forall p \in
\mathcal{P}, \lim\_{s \to p} |G(s)| = \infty \]

### Example

\[ G(s) = \frac{2s + 1}{(s+3)(s+2)} = \frac{2(s + 0.5)}{(s+3)(s+2)} \]

\[ \mathcal{P} = \{ -3, -2 \} \]

\[ \mathcal{Z} = \{ -0.5 \} \]

### Time Domain Response Mappings (\(\mathcal{L}^{-1}\))

**1. Real Pole (\(p_i = -\sigma, \sigma > 0\)):** \[ G(s) = \frac{1}{s + \sigma}
\xrightarrow{\mathcal{L}^{-1}} g(t) = e^{-\sigma t}u(t) \] \[ \tau =
\frac{1}{\sigma} \]

**2. Complex Conjugate Poles (\(p\_{1,2} = -\alpha \pm j\omega_d\)):** \[ G(s) =
\frac{\omega_d}{(s + \alpha)^2 + \omega_d^2} \xrightarrow{\mathcal{L}^{-1}} g(t)
= e^{-\alpha t} \sin(\omega_d t)u(t) \]

**3. Zeros and Derivative Action:** Let \(H(s) = \frac{1}{D(s)}\) and \(G(s) =
(s - z)H(s)\). \[ \mathcal{L}^{-1}\{G(s)\} = \frac{d}{dt}h(t) - z \cdot h(t) \]

**4. RHP Zero (Non-Minimum Phase) (\(z > 0\)):** Step response \(Y(s) =
\frac{1}{s} \frac{K(s-z)}{D(s)}\) \[ \dot{y}(0^+) = \lim\_{s \to \infty} s
[sY(s) - y(0^+)] \propto -K \quad (\text{Initial undershoot}) \]

### Stability Criteria

\[ \text{BIBO Stable} \iff \forall p_j \in \mathcal{P}, \quad \text{Re}(p_j) < 0
\]

\[ \text{Unstable} \iff \exists p_j \in \mathcal{P}, \quad \text{Re}(p_j) \geq 0
\]

#### Pole-Zero Plots

The pole-zero diagram plots poles (marked as ×) and zeros (marked as ○) on the
complex \\(s\\)-plane. The real axis represents damping (leftward = more
damping), and the imaginary axis represents oscillation frequency. This
visualization immediately reveals stability (all poles must be in the left-half
plane for BIBO stability) and transient characteristics.

### Discrete-Time Systems and Z-Transform

For digital implementations, continuous-time systems must be converted to
discrete-time systems. This requires sampling and the **Z-transform**.

#### Sampling and Difference Equations

A continuous-time signal \\(x(t)\\) is sampled at rate \\(f_s = 1/T\\): \\[ x[k]
= x(kT), \quad k = 0, 1, 2, \ldots \\]

where \\(T\\) is the sampling period. Differential equations become difference
equations: \\[ \dot{x}(t) \approx \frac{x[k] - x[k-1]}{T} \\]

#### Z-Transform Definition

The **Z-transform** is the discrete-time counterpart of the Laplace transform:
\\[ X(z) = \sum\_{k=0}^{\infty} x[k] z^{-k} \\]

It converts difference equations into algebraic equations in the \\(z\\)-domain,
enabling algebraic analysis of discrete systems.

#### Discrete Transfer Function

For a discrete-time LTI system, the transfer function is: \\[ H(z) =
\frac{Y(z)}{U(z)} = \frac{b_0 + b_1 z^{-1} + \cdots + b_m z^{-m}}{1 + a_1 z^{-1}

- \cdots + a_n z^{-n}} \\]

The impulse response at samples is obtained by inverse Z-transforming
\\(H(z)\\).

#### Continuous-to-Discrete Conversion Methods

Several methods map continuous transfer functions \\(G(s)\\) to discrete
transfer functions \\(G(z)\\):

| Method                | Mapping                                      | Characteristics                                            |
| --------------------- | -------------------------------------------- | ---------------------------------------------------------- |
| Forward Euler         | \\(z = 1 + sT\\)                             | Simple, but unstable for fast dynamics (large eigenvalues) |
| Backward Euler        | \\(z = \frac{1}{1-sT}\\)                     | Always stable, but lower accuracy                          |
| **Tustin (Bilinear)** | \\(s = \frac{2}{T} \frac{z-1}{z+1}\\)        | Best frequency matching; preserves stability               |
| Zero-Pole Match       | \\(z_i = e^{s_i T}\\), \\(p_i = e^{p_i T}\\) | Exact pole-zero mapping; good for control design           |

The **Tustin (bilinear) transform** is the industry standard because it
preserves the stability property: continuous-stable poles map to discrete-stable
poles inside the unit circle.

---

## 3. System Analysis

### Stability Criteria: Predicting Long-Term Behavior

Stability analysis determines whether the system remains bounded and whether it
converges to equilibrium over time. Different criteria apply to continuous and
discrete systems.

#### Continuous-Time Stability

For a linear continuous-time system \\(\dot{\mathbf{x}} = A\mathbf{x}\\),
stability is determined by the eigenvalues of the system matrix \\(A\\):

- **Asymptotically Stable (AS)**: All eigenvalues have negative real parts:
  \\(\text{Re}(\lambda_i) < 0\\)

  - Equivalently: all poles lie in the **Left-Half Plane (LHP)** of the
    \\(s\\)-plane
  - Behavior: every trajectory converges to the origin as \\(t \to \infty\\)
  - General solution: \\(\mathbf{x}(t) = e^{At}\mathbf{x}(0) \to \mathbf{0}\\)
    as \\(t \to \infty\\)

- **Marginally Stable (MS)**: Some eigenvalues on the imaginary axis
  (\\(\text{Re}(\lambda) = 0\\)), rest in LHP

  - Example: undamped oscillator has poles at \\(s = \pm j\omega\\)
  - Behavior: bounded oscillation without decay; limit cycles

- **Unstable**: At least one eigenvalue with positive real part:
  \\(\text{Re}(\lambda_i) > 0\\)
  - Behavior: exponential divergence from equilibrium
  - Solutions grow unboundedly

#### Discrete-Time Stability

For discrete-time systems, stability is determined by pole locations in the
\\(z\\)-plane:

- **Stable**: All poles satisfy \\(|z_i| < 1\\) (inside unit circle)

  - Behavior: state decays exponentially to zero
  - Time-domain decay: \\(|x[k]| \sim |z_i|^k |x[0]|\\)
  - System response bounded for bounded inputs

- **Marginally Stable**: Poles on unit circle \\(|z| = 1\\)

  - Behavior: bounded oscillation (no decay)
  - Exception: multiple poles at the same location on unit circle cause
    instability

- **Unstable**: Poles outside unit circle \\(|z_i| > 1\\)
  - Behavior: exponential growth of state

#### Lyapunov Stability Theory for Nonlinear Systems

For nonlinear systems \\(\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})\\), Lyapunov
theory provides a powerful framework that does not require solving the ODEs
explicitly.

A scalar function \\(V(\mathbf{x})\\) is a **Lyapunov function** for an
equilibrium \\(\mathbf{x}^\*\\) if:

1. \\(V(\mathbf{x}^\*) = 0\\) (the equilibrium has zero "energy")
2. \\(V(\mathbf{x}) > 0\\) for all \\(\mathbf{x} \neq \mathbf{x}^\*\\) (positive
   definite)
3. \\(\dot{V}(\mathbf{x}) = \nabla V \cdot \mathbf{f}(\mathbf{x}) < 0\\) along
   trajectories (energy dissipation)

If a Lyapunov function exists, then:

- The equilibrium is **globally asymptotically stable (GAS)**
- All trajectories converge to \\(\mathbf{x}^\*\\) regardless of initial
  condition

[stanford-lyapunov],[washington-lyapunov]

The Lyapunov function represents generalized "energy" that is not necessarily
mechanical energy. If energy always decreases (except at equilibrium), the
system must dissipate and settle at the equilibrium. This is a fundamental
principle: systems naturally evolve toward states of lowest energy
\cite{stanford-lyapunov,washington-lyapunov}.

### Frequency Response: Magnitude and Phase vs. Frequency

Frequency response characterizes how a system responds to sinusoidal inputs of
varying frequencies. This is essential for understanding bandwidth, resonance,
and control design.

When a linear system is driven by a sinusoidal input \\(u(t) = U_0 \sin(\omega
t)\\), the steady-state output is also sinusoidal: \\[ y(t) = Y_0 \sin(\omega t

- \phi) \\]

The frequency response describes:

- **Magnitude (Gain)**: \\(|G(j\omega)| = Y_0 / U_0\\) (amplitude scaling)
- **Phase Shift**: \\(\phi(\omega) = \angle G(j\omega)\\) (time delay expressed
  as phase)

A **Bode plot** consists of two semi-logarithmic graphs:

1. **Magnitude Plot**:

   - Vertical axis: Gain in decibels: \\(\text{Gain (dB)} = 20
     \log\_{10}|G(j\omega)|\\)
   - Horizontal axis: Frequency in log scale: \\(\log\_{10}(\omega)\\) [rad/s]
   - Key frequency: -3 dB point marks the bandwidth

2. **Phase Plot**:
   - Vertical axis: Phase angle in degrees: \\(\phi(\omega) = \angle
     G(j\omega)\\)
   - Horizontal axis: Frequency in log scale

The logarithmic scale makes it easy to see the system response over multiple
decades of frequency, and the dB scale makes multiplying gains (cascading
systems) equivalent to adding them on the plot.

#### Asymptotic Approximation

- Each **zero** at \\(s = -z_i\\) contributes:

  - \\(+20\\) dB/decade slope above the corner frequency \\(\omega = z_i\\)
  - \\(+90°\\) phase shift

- Each **pole** at \\(s = -p_i\\) contributes:

  - \\(-20\\) dB/decade slope above the corner frequency \\(\omega = p_i\\)
  - \\(-90°\\) phase shift

- A **pole at the origin** (integrator) contributes:
  - Initial slope of \\(-20\\) dB/decade
  - Phase shift of \\(-90°\\)

This asymptotic method enables rapid sketching and visualization of frequency
response without detailed computation.

### Gain Margin and Phase Margin: Stability Robustness

Gain and phase margins quantify how much uncertainty or perturbation a system
can tolerate before becoming unstable. They are critical for robust controller
design.

#### Gain Margin (GM)

_The maximum multiplicative increase in system gain before the closed-loop
system becomes unstable._

Measured at the phase crossover frequency \\(\omega*{pc}\\) (where phase =
\\(-180°\\)): \\[ \text{GM (dB)} = 0 - |G(j\omega*{pc})| \text{ (in dB)} =
-20\log*{10}|G(j\omega*{pc})| \\]

- **GM > 0 dB**: System is stable
- **GM = 0 dB**: System is marginally stable (critically damped)
- **GM < 0 dB**: System is unstable

A typical design specification is \\(\text{GM} > 6\\) dB, which allows the
system to tolerate a gain increase by a factor of 2 before instability.

#### Phase Margin (PM)

**Definition**: The maximum phase lag increase before the closed-loop system
becomes unstable.

Measured at the gain crossover frequency \\(\omega*{gc}\\) (where
\\(|G(j\omega)| = 0\\) dB, i.e., unity gain): \\[ \text{PM} = 180° - |\angle
G(j\omega*{gc})| \\]

or equivalently: \\[ \text{PM} = -180° - \angle G(j\omega_{gc}) \\]

when using the convention that the phase of the open-loop transfer function is
negative.

- **PM > 0°**: System is stable
- **PM = 0°**: System is marginally stable
- **PM < 0°**: System is unstable

A typical design specification is \\(\text{PM} > 45°\\), which means the system
can tolerate up to 45° of additional phase lag from unmodeled dynamics.

#### Nyquist Criterion

In the Nyquist plot, the critical point is \\((-1, 0j)\\). Gain and phase
margins are measures of distance from this critical point:

- Gain margin: how far the magnitude can increase before the plot passes through
  \\(-1\\)
- Phase margin: how much additional phase lag is tolerable before crossing
  \\(-1\\)

### Controllability

Controllability determines whether the control input can drive the system to any
desired state in finite time. This is essential for control system design.

#### LTI Continuous-Time Controllability

A system \\(\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}\\) is **controllable**
if, for any initial state \\(\mathbf{x}(0) = \mathbf{x}\_0\\) and any desired
final state \\(\mathbf{x}\_f\\), there exists a finite time \\(t_f > 0\\) and
control input \\(\mathbf{u}(t)\\), \\(t \in [0, t_f]\\), such that
\\(\mathbf{x}(t_f) = \mathbf{x}\_f\\).

#### Kalman Controllability Test

A system is controllable if and only if the **controllability matrix** has full
rank: \\[ \text{rank}(\mathcal{C}) = n \\]

where the controllability matrix is: \\[ \mathcal{C} = \begin{bmatrix} B & AB &
A^2B & \cdots & A^{n-1}B \end{bmatrix} \in \mathbb{R}^{n \times nm} \\]

This is an \\(n \times nm\\) matrix formed by stacking the \\(n \times m\\)
matrices. The rank must equal the system order \\(n\\).

- \\(B\\) column: direct influence of input on the states
- \\(AB\\) column: influence of input through one state transition (dynamics)
- \\(A^{k-1}B\\): influence through longer paths in the state space

If rank\\((\mathcal{C}) < n\\), some state direction is unreachable: the control
input cannot affect that mode, making it impossible to steer the system to
arbitrary states.

### Observability

> _Observability determines whether we can uniquely estimate the internal_ >
> _state from measurements of the system output._ > _This is essential for
> observer design and state feedback implementation._

#### LTI Continuous-Time Observability

A system with state dynamics \\(\dot{\mathbf{x}} = A\mathbf{x}\\) and
measurement \\(\mathbf{y} = C\mathbf{x}\\) is **observable** if, given the
measurement \\(\mathbf{y}(t)\\) over any finite time interval \\([0, t_f]\\),
the initial state \\(\mathbf{x}(0)\\) can be uniquely determined.

#### Kalman Observability Test

A system is observable if and only if the **observability matrix** has full
rank: \\[ \text{rank}(\mathcal{O}) = n \\]

where the observability matrix is: \\[ \mathcal{O} = \begin{bmatrix} C \\ CA \\
CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix} \in \mathbb{R}^{pn \times n} \\]

This is an \\(pn \times n\\) matrix formed by stacking \\(p \times n\\)
matrices. The rank must equal the system order \\(n\\).

- \\(C\\) row: direct measurement of states
- \\(CA\\) row: information about states after one state transition
- \\(CA^{k-1}\\) row: information over longer time horizons

If rank\\((\mathcal{O}) < n\\), some state component is unobservable: it affects
the dynamics but never influences the measurements, making it impossible to
determine that state from output alone.

---

## 4. Simulation

### Initial Value Problems (IVP): Problem Formulation

An **Initial Value Problem (IVP)** is an ODE together with an initial condition.
Numerical methods solve IVPs to simulate dynamical systems.

#### Standard Form

\\[ \dot{\mathbf{x}} = \mathbf{f}(t, \mathbf{x}), \quad \mathbf{x}(t_0) =
\mathbf{x}_0 \\]

Given the function \\(\mathbf{f}\\), the initial state \\(\mathbf{x}\_0\\), and
the initial time \\(t_0\\), we seek the solution \\(\mathbf{x}(t)\\) for \\(t
\geq t_0\\) \cite{frontiers-rk4}.

#### Existence and Uniqueness (Picard-Lindelöf Theorem)

A unique solution exists in a neighborhood of \\(t_0\\) if:

1. \\(\mathbf{f}(t, \mathbf{x})\\) is continuous in both arguments
2. \\(\mathbf{f}\\) is Lipschitz continuous in \\(\mathbf{x}\\):
   \\(\|\mathbf{f}(t, \mathbf{x}\_1) - \mathbf{f}(t, \mathbf{x}\_2)\| \leq L
   \|\mathbf{x}\_1 - \mathbf{x}\_2\|\\) for some constant \\(L\\)

Most practical systems satisfy these conditions, ensuring that numerical
solutions are meaningful.

### Euler Integration: Forward Stepping

The **Forward Euler method** (also called RK1) is the simplest explicit
numerical integrator. Given step size \\(h\\):

\\[ \mathbf{x}_{k+1} = \mathbf{x}_k + h \cdot \mathbf{f}(t_k, \mathbf{x}_k) \\]

where \\(t\_{k+1} = t_k + h\\).

From the Taylor series expansion: \\[ \mathbf{x}(t + h) = \mathbf{x}(t) +
h\dot{\mathbf{x}}(t) + \frac{h^2}{2}\ddot{\mathbf{x}}(\xi) \quad \xi \in (t,
t+h) \\]

The Euler method truncates after the first-order term, replacing the derivative
with its value at the current time: \\[ \mathbf{x}(t + h) \approx \mathbf{x}(t)

- h \mathbf{f}(t, \mathbf{x}(t)) \\]

#### Error Analysis

- **Local Truncation Error (LTE)**: \\(\mathcal{O}(h^2)\\) per step (from
  truncating the Taylor series)
- **Global Error**: \\(\mathcal{O}(h)\\) over a fixed time interval (accumulated
  over multiple steps)
- **Stability Requirement**: For numerical stability of linear systems, the step
  size must satisfy \\(|1 + h\lambda| < 1\\), where \\(\lambda\\) is an
  eigenvalue

The stability constraint is severe for stiff systems (systems with wide range of
timescales), requiring tiny step sizes \cite{frontiers-rk4}.

#### Disadvantages

- Only first-order accurate (slow convergence with step size)
- Unstable for stiff systems without prohibitively small time steps
- Rarely used in practice for engineering applications

## References

```bibtex
@book{hover_system_2022,
  title = {System {Design} for {Uncertainty}},
  url = {<https://eng.libretexts.org/Bookshelves/Mechanical_Engineering/System_Design_for_Uncertainty_(Hover_and_Triantafyllou)>},
  abstract = {This text covers the design, construction, and testing of field robotic systems, through team projects with each student responsible for a specific subsystem. Projects focus on electronics, instrumentation, and machine elements. Design for operation in uncertain conditions is a focus point, with ocean waves and marine structures as a central theme. Topics include basic statistics, linear systems, Fourier transforms, random processes, spectra, ethics in engineering practice, and extreme events with applications in design.},
  language = {en},
  urldate = {2025-12-24},
  publisher = {Engineering LibreTexts},
  author = {Hover, Franz S. and Triantafyllou, Michael S.},
  month = mar,
  year = {2022},
  note = {License: CC BY-NC-SA 4.0. Original source: MIT OpenCourseWare (Fall 2009)},
}


@article{aero_students_phase,
  title = {Phase {Portraits} and {Stability}},
  url = {http://www.aerostudents.com/courses/differential-equations/phasePortraitsAndStability.pdf},
  author = {{AeroStudents}},
}
```
