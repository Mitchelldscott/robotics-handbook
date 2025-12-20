# Chapter 1: Linear Systems and Data Representations

## I. Motivating Linear Systems for Roboticists

In autonomous systems, nearly every computation reduces to linear algebra operations:

- **State representation**: Pose of a robot, configuration of a manipulator arm
- **Sensor models**: Linear mappings from state to measurements (cameras, lidar)
- **Dynamics linearization**: Control design via feedback gains requires linearized models
- **Estimation**: Kalman filters, least-squares parameter fitting
- **Coordinate transformations**: Rotations, homogeneous transforms
- **System identification**: Learning dynamics models from data

This chapter develops the mathematical machinery—vectors, matrices, subspaces, eigenvalues—
  with robotics applications in mind. Unlike pure mathematics texts, we emphasize
  **computational implementation** and **numerical stability** from the start.

---

## II. Data Representations

### 1. Vectors and Vector Spaces

#### Definition: Vector

A **vector** is an ordered list of numbers, denoted \(\mathbf{v} \in \mathbb{R}^n\):
\[\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}\]

Geometrically: a point or direction in \(n\)-dimensional space. In robotics, vectors
  represent positions, velocities, forces, and sensor readings.

#### Vector Spaces and Axioms

A **vector space** \(V\) over a field \(\mathbb{F}\) (usually \(\mathbb{R}\) or
  \(\mathbb{C}\)) is a set closed under two operations:

1. **Vector Addition** \(\mathbf{u} + \mathbf{v}: V \times V \to V\)
   - Commutative: \(\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}\)
   - Associative: \((\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})\)
   - Identity: \(\exists \, \mathbf{0} \in V : \mathbf{u} + \mathbf{0} = \mathbf{u}\)
   - Inverses: \(\forall \mathbf{u} \, \exists -\mathbf{u} : \mathbf{u} + (-\mathbf{u}) = \mathbf{0}\)

2. **Scalar Multiplication** \(\alpha \mathbf{v}: \mathbb{F} \times V \to V\)
   - Associativity: \(\alpha(\beta \mathbf{v}) = (\alpha\beta) \mathbf{v}\)
   - Left distributive: \(\alpha(\mathbf{u} + \mathbf{v}) = \alpha\mathbf{u} + \alpha\mathbf{v}\)
   - Right distributive: \((\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}\)
   - Unit scaling: \(1 \cdot \mathbf{v} = \mathbf{v}\)

**Key Examples:**

- \(\mathbb{R}^n\): \(n\)-dimensional real Euclidean space (most robotics applications)
- \(\mathbb{C}^n\): complex vectors (frequency-domain analysis, eigenvalues)
- Function spaces: \(\mathcal{C}[0,T]\) = continuous functions on \([0,T]\) (state trajectories)

#### Linear Functions (Functionals)

**Theorem (Riesz Representation)**: If \(f: \mathbb{R}^n \to \mathbb{R}\) is linear, then
\[f(\mathbf{x}) = \mathbf{a}^T \mathbf{x}\]
for some unique \(\mathbf{a} \in \mathbb{R}^n\).

**Proof Sketch**: For the standard basis \(\mathbf{e}_i = (0, \ldots, 1, \ldots, 0)^T\),
\[f(\mathbf{x}) = f\left(\sum_{i=1}^n x_i \mathbf{e}_i\right) = \sum_{i=1}^n x_i
  f(\mathbf{e}_i) = \mathbf{a}^T \mathbf{x}\]
where \(a_i := f(\mathbf{e}_i)\).

**Superposition Principle**: Linear functions satisfy
\[f(\alpha \mathbf{u} + \beta \mathbf{v}) = \alpha f(\mathbf{u}) + \beta f(\mathbf{v})\]

**Affine Functions**: A function is **affine** if it is linear plus a constant:
\[f(\mathbf{x}) = \mathbf{a}^T \mathbf{x} + b\]

Equivalently: \(f(\alpha \mathbf{u} + \beta \mathbf{v}) = \alpha f(\mathbf{u}) + \beta
  f(\mathbf{v})\) whenever \(\alpha + \beta = 1\).

_Application_: Robot dynamics after feedback \(u = -K x + r\) is affine in state.

#### Norms: Measuring Vector "Size"

A **norm** \(\|\cdot\|: V \to \mathbb{R}_{\geq 0}\) satisfies:

1. **Positive Definiteness**: \(\|\mathbf{v}\| > 0\) unless \(\mathbf{v} = \mathbf{0}\)
2. **Homogeneity**: \(\|\alpha \mathbf{v}\| = |\alpha| \|\mathbf{v}\|\)
3. **Triangle Inequality**: \(\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|\)

##### \(L_p\) Norms

For \(p \geq 1\), the **\(L_p\) norm** is:
\[\|\mathbf{v}\|_p = \left(\sum_{i=1}^{n} |v_i|^p\right)^{1/p}\]

| \(p\) | Name | Formula | Use in Robotics |
|---|---|---|---|
| 1 | Manhattan / Taxicab | \(\sum_i \|v_i\|\) | Path cost on grids |
| 2 | Euclidean | \(\sqrt{\sum_i v_i^2}\) | Distance, state tracking |
| \(\infty\) | Maximum | \(\max_i \|v_i\|\) | Component-wise bounds, constraints |

##### Euclidean Norm (Most Common)

\[\|\mathbf{v}\|_2 = \sqrt{\mathbf{v}^T \mathbf{v}} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}\]

Geometric interpretation: Euclidean distance from origin to point \(\mathbf{v}\).

**Inner Product Definition**:
\[\mathbf{u}^T \mathbf{v} = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos\theta\]
where \(\theta\) is the angle between vectors. Orthogonal vectors satisfy \(\mathbf{u}^T
  \mathbf{v} = 0\).

---

### 2. Matrices and Linear Transformations

#### Definition: Matrix

An **\(m \times n\) matrix** is a rectangular array:
\[A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots &
 a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}\]

- **Rows**: \(m\) (vertical dimension)
- **Columns**: \(n\) (horizontal dimension)
- Entry: \(A_{ij}\) or \(a_{ij}\) is the element at row \(i\), column \(j\)

**Special Cases**:

- **Column vector**: \(n \times 1\) matrix
- **Row vector**: \(1 \times n\) matrix
- **Square matrix**: \(m = n\)
- **Tall matrix**: \(m > n\) (more rows than columns; often overdetermined systems)
- **Wide matrix**: \(m < n\) (more columns than rows; underdetermined systems)

_Application_: In sensor fusion, an \(m \times n\) Jacobian matrix relates joint
  velocities (\(n\) DOF) to end-effector velocities (\(m\) task dimensions).

#### Linear Transformations

A **linear transformation** \(T: \mathbb{R}^n \to \mathbb{R}^m\) satisfies:

1. \(T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})\)
2. \(T(\alpha \mathbf{v}) = \alpha T(\mathbf{v})\)

**Key Theorem**: Every linear transformation \(T: \mathbb{R}^n \to \mathbb{R}^m\) can be represented by a unique \(m \times n\) matrix \(A\) such that
\[T(\mathbf{x}) = A\mathbf{x}\]

The columns of \(A\) are the images of the standard basis:
\[A = \begin{bmatrix} | & & | \\ T(\mathbf{e}_1) & \cdots & T(\mathbf{e}_n) \\ | & & | \end{bmatrix}\]

#### Matrix Operations

**Addition** (only for same size):
\[(A + B)_{ij} = A_{ij} + B_{ij}\]

**Scalar Multiplication**:
\[(cA)_{ij} = c \cdot A_{ij}\]

**Matrix Multiplication** (compatible dimensions: \(A\) is \(m \times n\), \(B\) is \(n \times p\)):
\[(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}\]

**Critical Note**: Matrix multiplication is **not commutative**: \(AB \neq BA\) in general.

**Matrix Inverse** (for square \(A\), if it exists):
\[A^{-1} : A A^{-1} = A^{-1} A = I_n\]

Exists if and only if \(\text{rank}(A) = n\) (full rank).

**Properties**:

- \((AB)^{-1} = B^{-1} A^{-1}\)
- \((A^T)^{-1} = (A^{-1})^T\)

#### Transpose

\[(A^T)_{ij} = A_{ji}\]

**Properties**:

- \((A^T)^T = A\)
- \((A + B)^T = A^T + B^T\)
- \((AB)^T = B^T A^T\)
- \((A^{-1})^T = (A^T)^{-1}\)

_Application_: In control, the **dual system** \(\dot{x} = A^T x\) characterizes observability.

#### Span, Basis, and Dimension

**Linear Combination**: \(\mathbf{v}\) is a linear combination of \(\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}\) if
\[\mathbf{v} = c_1 \mathbf{v}_1 + \cdots + c_k \mathbf{v}_k\]
for some scalars \(c_i\).

**Span**: The set of all linear combinations:
\[\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \left\{ \sum_{i=1}^k c_i \mathbf{v}_i : c_i \in \mathbb{R} \right\}\]

**Linear Independence**: Vectors \(\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}\) are linearly independent if
\[c_1 \mathbf{v}_1 + \cdots + c_k \mathbf{v}_k = \mathbf{0} \quad \Rightarrow \quad c_i = 0 \text{ for all } i\]

Otherwise, they are **linearly dependent**.

**Basis**: A set \(B\) is a basis for vector space \(V\) if:

1. \(B\) is linearly independent
2. \(\text{span}(B) = V\)

**Dimension**: The number of vectors in any basis for \(V\):
\[\dim(V) = |B|\]

_Application_: In a robot with 6 DOF in \(\text{SE}(3)\), any valid configuration space basis has dimension 6.

#### Rank and Nullity

For a matrix \(A \in \mathbb{R}^{m \times n}\):

- **Rank** \(\text{rank}(A) = r\) = dimension of the column space
- **Nullity** \(\text{nullity}(A) = n - r\) = dimension of the null space

**Rank-Nullity Theorem**:
\[\text{rank}(A) + \text{nullity}(A) = n\]

_Application_: For a robot Jacobian \(J \in \mathbb{R}^{6 \times n}\):

- \(\text{rank}(J) = 6\) everywhere except singularities (full rank = fully controllable end-effector)
- \(\text{nullity}(J) > 0\) at singularities (null space = redundant DOF)

#### Determinants

The **determinant** \(\det(A)\) of a square \(n \times n\) matrix encodes crucial information: invertibility, volume scaling, and eigenvalues.

**Leibniz Formula**:
\[\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} A_{i, \sigma(i)}\]

where \(S_n\) is all permutations of \(\{1, \ldots, n\}\) and \(\text{sgn}(\sigma) = \pm 1\).

**Cofactor Expansion** (easier to compute):
\[\det(A) = \sum_{j=1}^{n} (-1)^{1+j} A_{1j} \det(A_{1j})\]
where \(A_{1j}\) is the \((n-1) \times (n-1)\) submatrix obtained by deleting row 1 and column \(j\).

**Key Properties**:

1. \(\det(AB) = \det(A)\det(B)\)
2. \(\det(A^T) = \det(A)\)
3. \(\det(cA) = c^n \det(A)\)
4. \(\det(A) = 0 \iff A\) is singular (non-invertible)
5. \(\det(A) = \prod_{i=1}^{n} \lambda_i\) where \(\lambda_i\) are eigenvalues

---

## III. Solving Linear Systems and Subspaces

### The Four Fundamental Subspaces

For any \(A \in \mathbb{R}^{m \times n}\) with rank \(r\):

| Subspace | Definition | Dimension | Role |
|---|---|---|---|
| **Column Space** \(C(A)\) | \(\{\mathbf{y} = A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}\) | \(r\) | Range of \(A\); reachable outputs |
| **Row Space** \(C(A^T)\) | Span of rows of \(A\) | \(r\) | Span of gradients of output functions |
| **Nullspace** \(N(A)\) | \(\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}\) | \(n - r\) | Uncontrollable directions |
| **Left Nullspace** \(N(A^T)\) | \(\{\mathbf{y} : A^T\mathbf{y} = \mathbf{0}\}\) | \(m - r\) | Unobservable directions |

**Orthogonality Relations**:

- \(C(A^T) \perp N(A)\): row space orthogonal to null space
- \(C(A) \perp N(A^T)\): column space orthogonal to left null space

**Direct Sum Decomposition**:

- \(\mathbb{R}^n = C(A^T) \oplus N(A)\) (every vector = controllable part + uncontrollable part)
- \(\mathbb{R}^m = C(A) \oplus N(A^T)\)

_Application_: In observability analysis, the left null space of the observability matrix characterizes unobservable modes.

### Solving \(A\mathbf{x} = \mathbf{b}\)

**Three cases**:

1. **Square, full rank** (\(m = n\), \(\det(A) \neq 0\)): Unique solution \(\mathbf{x} = A^{-1}\mathbf{b}\)
2. **Overdetermined** (\(m > n\)): Usually no exact solution; find least-squares \(\mathbf{\hat{x}}\)
3. **Underdetermined** (\(m < n\)): Infinitely many solutions; find minimum-norm solution

**Gaussian Elimination** (classical approach):

1. Augment: \([A | \mathbf{b}]\)
2. Forward elimination to row echelon form (REF)
3. Back substitution

**Numerically Robust**: Use \(A = QR\) decomposition (QR method) or \(A = U\Sigma V^T\) (SVD).

---

## IV. Eigenvalues, Eigenvectors, and Spectral Analysis

### Eigenvalue Equation

For a square matrix \(A\), the equation
\[A\mathbf{v} = \lambda \mathbf{v}\]
has a non-trivial solution if and only if \(\det(A - \lambda I) = 0\).

- **Eigenvalue** \(\lambda\): scalar
- **Eigenvector** \(\mathbf{v} \neq \mathbf{0}\): vector

**Geometric Meaning**: \(A\) only **scales** the eigenvector by factor \(\lambda\); the direction is unchanged.

#### Characteristic Polynomial

\[p(\lambda) = \det(A - \lambda I) = (-1)^n \lambda^n + (\text{lower order terms})\]

- **Degree**: \(n\) (an \(n \times n\) matrix has \(n\) eigenvalues, counting multiplicities and complex values)
- **Roots**: eigenvalues \(\lambda_1, \ldots, \lambda_n\)

_Application_: Stability of continuous-time LTI system \(\dot{\mathbf{x}} = A\mathbf{x}\) is determined by Re\((\lambda_i) < 0\) for all \(i\).

#### Diagonalization

If \(A\) has \(n\) linearly independent eigenvectors (collected as columns of \(S\)), then
\[A = S\Lambda S^{-1}\]
where \(\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)\).

**When is this possible?**

- All eigenvalues are distinct: always diagonalizable
- Repeated eigenvalues: diagonalizable if geometric multiplicity = algebraic multiplicity (i.e., enough independent eigenvectors)
- **Defective matrices**: Cannot be diagonalized; use Jordan normal form instead

**Power of \(A\)**:
\[A^k = S\Lambda^k S^{-1}\]

_Application_: Discrete-time system \(\mathbf{x}_{k+1} = A\mathbf{x}_k\) solutions: \(\mathbf{x}_k = A^k \mathbf{x}_0 = S\Lambda^k S^{-1} \mathbf{x}_0\).

#### Spectral Theorem (Symmetric Matrices)

If \(A = A^T\) (symmetric), then:

1. All eigenvalues are **real**
2. Eigenvectors form an **orthonormal** basis
3. \(A = Q\Lambda Q^T\) where \(Q\) is orthogonal (\(Q^T Q = I\))

**Consequence**: Symmetric matrices are always diagonalizable.

_Application_: Covariance matrices in Kalman filters are symmetric positive semidefinite; eigendecomposition is numerically stable.

#### Spectral Radius

\[\rho(A) = \max_i |\lambda_i|\]

- **Stability**: Continuous-time \(\dot{\mathbf{x}} = A\mathbf{x}\) is stable if \(\rho(A) < 0\) (all Re\((\lambda_i) < 0\))
- **Discrete-time**: \(\mathbf{x}_{k+1} = A\mathbf{x}_k\) is stable if \(\rho(A) < 1\)

---

## V. Orthogonality and Least-Squares

### Orthogonal Vectors and Matrices

**Orthogonality**: \(\mathbf{u} \perp \mathbf{v}\) if \(\mathbf{u}^T \mathbf{v} = 0\).

**Orthogonal Set**: Vectors \(\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}\) are mutually orthogonal if \(\mathbf{v}_i^T \mathbf{v}_j = 0\) for \(i \neq j\).

**Orthonormal Set**: Orthogonal + unit norm: \(\|\mathbf{v}_i\|_2 = 1\) and \(\mathbf{v}_i^T \mathbf{v}_j = \delta_{ij}\).

#### Orthogonal Matrices

A square matrix \(Q \in \mathbb{R}^{n \times n}\) is **orthogonal** if its columns are orthonormal:
\[Q^T Q = Q Q^T = I_n\]

Equivalently: \(Q^T = Q^{-1}\).

**Properties**:

- \(\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2\) (preserves Euclidean norm)
- \(\det(Q) = \pm 1\)
- Numerically stable: no amplification of rounding errors

_Application_: Rotation matrices are orthogonal. Coordinate transformations via \(\mathbf{x}_{\text{new}} = R\mathbf{x}_{\text{old}}\) preserve distances.

#### Gram–Schmidt Orthogonalization

Given linearly independent vectors \(\{\mathbf{a}_1, \ldots, \mathbf{a}_n\}\), produce orthonormal vectors \(\{\mathbf{q}_1, \ldots, \mathbf{q}_n\}\):

**Algorithm**:

```
for j = 1 to n:
    v_j = a_j
    for i = 1 to j-1:
        v_j = v_j - (q_i^T a_j) * q_i  // subtract projection onto q_i
    q_j = v_j / ||v_j||_2             // normalize
```

**Foundation of QR decomposition**: \(A = QR\) where \(Q\) has orthonormal columns and \(R\) is upper triangular.

### Least-Squares and Projection

**Problem**: Solve overdetermined system \(A\mathbf{x} = \mathbf{b}\) where \(m > n\) and \(\mathbf{b} \notin C(A)\).

**Least-Squares Solution**: Find \(\mathbf{\hat{x}}\) minimizing
\[\|\mathbf{e}\|_2^2 = \|A\mathbf{x} - \mathbf{b}\|_2^2\]

**Normal Equations**:
\[A^T A \mathbf{\hat{x}} = A^T \mathbf{b}\]

(Solvable if \(A\) has full column rank.)

**Projection Matrix** (for full column rank):
\[P = A(A^T A)^{-1}A^T\]

Projects any vector onto \(C(A)\):

- \(\mathbf{p} = P\mathbf{b}\) = best approximation to \(\mathbf{b}\) in \(C(A)\)
- \(\mathbf{e} = \mathbf{b} - \mathbf{p} = P^{\perp}\mathbf{b}\) where \(P^{\perp} = I - P\)

**Residual Vector**: \(\mathbf{e} = \mathbf{b} - A\mathbf{\hat{x}}\) satisfies \(A^T \mathbf{e} = \mathbf{0}\) (residual is orthogonal to column space).

_Application_: Robot sensor calibration via least-squares parameter estimation from noisy measurements.

#### Pseudo-Inverse

For \(A \in \mathbb{R}^{m \times n}\) with full column rank:
\[A^{\dagger} = (A^T A)^{-1}A^T\]

The **least-squares solution** is \(\mathbf{\hat{x}} = A^{\dagger} \mathbf{b}\).

**General pseudo-inverse** (via SVD, works for all ranks):
\[A^{\dagger} = V \Sigma^{\dagger} U^T\]
where \(A = U\Sigma V^T\) and \(\Sigma^{\dagger}\) inverts non-zero singular values.

---

## VI. Matrix Factorizations

### QR Decomposition

**Form**: \(A = QR\)

- \(Q\): \(m \times n\) matrix with orthonormal columns
- \(R\): \(n \times n\) upper triangular matrix

**When to use**:

- Numerically stable least-squares: solve \(R\mathbf{x} = Q^T\mathbf{b}\) by back-substitution
- Gram–Schmidt orthogonalization

**Computational cost**: \(\sim 2mn^2\) flops (Householder reflections or Givens rotations).

### Singular Value Decomposition (SVD)

**Form**: \(A = U\Sigma V^T\)

- \(U\): \(m \times m\) orthogonal matrix
- \(\Sigma\): \(m \times n\) diagonal matrix with non-negative singular values \(\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0\)
- \(V\): \(n \times n\) orthogonal matrix

**Key properties**:

- \(\text{rank}(A) = \#\) of non-zero singular values
- \(\text{cond}(A) = \sigma_1 / \sigma_n\) (condition number)
- Null space: columns of \(V\) corresponding to zero singular values
- Column space: columns of \(U\) corresponding to non-zero singular values

**Applications**:

- Computing pseudo-inverse: \(A^{\dagger} = V\Sigma^{\dagger}U^T\)
- Condition number assessment (numerical stability)
- Rank detection and truncation (data compression)
- Principal Component Analysis (PCA)

_Application_: In feature-based SLAM, SVD of measurement-to-state Jacobian detects observable directions.

### LU Decomposition (with Partial Pivoting)

**Form**: \(PA = LU\)

- \(P\): permutation matrix (row swaps for numerical stability)
- \(L\): \(n \times n\) lower triangular with 1's on diagonal
- \(U\): \(n \times n\) upper triangular

**When to use**:

- Solving multiple systems with same \(A\): \(PA = LU\) once, then solve \(L(U\mathbf{x}) = P\mathbf{b}\) for each \(\mathbf{b}\)
- Efficient for sparse matrices

**Computational cost**: \(\sim n^3/3\) flops.

### Cholesky Decomposition

**Form**: \(A = LL^T\)

- \(L\): lower triangular with positive diagonal entries
- \(A\) must be **symmetric positive definite** (SPD)

**When to use**:

- Fastest factorization for SPD matrices (half the flops of LU)
- Solving \(A\mathbf{x} = \mathbf{b}\) with SPD \(A\): forward/back-substitution

**Computational cost**: \(\sim n^3/6\) flops.

_Application_: Covariance matrix factorization in Kalman filters. Cholesky is numerically stable and efficient.

**Numerical Note**: If Cholesky fails (negative diagonal pivot), matrix is not SPD—check for rounding errors or ill-conditioning.

### Schur and Jordan Normal Forms

**Schur Decomposition** (for any square matrix):
\[A = QTQ^T\]
where \(Q\) is orthogonal and \(T\) is upper triangular (real Schur form uses block-triangular).

**Jordan Normal Form** (non-symmetric):
\[A = SJS^{-1}\]
where \(J\) is block-diagonal with Jordan blocks. Used to handle repeated eigenvalues and defective matrices.

---

## VII. Jacobians and Taylor Approximations

### Jacobian Matrix

For a function \(\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m\), the **Jacobian** is:
\[J_{\mathbf{f}}(\mathbf{x}) = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}\]

**Size**: \(m \times n\)

**Meaning**: \(J_{\mathbf{f}}(\mathbf{x}_0)\) is the best **linear approximation** of \(\mathbf{f}\) near \(\mathbf{x}_0\).

_Application_: Robot Jacobian \(J(\mathbf{q})\) relates joint velocities to end-effector velocities:
\[\mathbf{v}_{\text{EE}} = J(\mathbf{q}) \dot{\mathbf{q}}\]

### First-Order Taylor Approximation

For \(\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m\) differentiable near \(\mathbf{x}_0\):
\[\hat{\mathbf{f}}(\mathbf{x}) = \mathbf{f}(\mathbf{x}_0) + J_{\mathbf{f}}(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)\]

**Accuracy**: Excellent when all \(x_i - x_{0i}\) are small.

_Application_: **Linearization** for control design. For nonlinear dynamics \(\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})\), linearize around equilibrium to design LTI feedback.

---

## VIII. Practical Example: Robot Sensor Calibration via Least-Squares

**Problem**: A 6-axis force/torque sensor has unknown biases \(\mathbf{b}\) and a scale matrix \(M\). Collect \(N\) measurements under known loads and estimate \([\mathbf{b}, M]\).

**Model**:
\[\mathbf{y}_i = M \mathbf{f}_i + \mathbf{b} + \boldsymbol{\epsilon}_i\]

where \(\mathbf{y}_i\) = measured force/torque, \(\mathbf{f}_i\) = true load, \(\boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \Sigma)\) noise.

**Rewritten as linear system**:
Stack vertically to get \(A\mathbf{x} = \mathbf{y} + \boldsymbol{\epsilon}\) where \(\mathbf{x}\) contains the unknown parameters (36 scale elements + 6 biases = 42 parameters).

**Solution**:

```julia
# Collect N measurements (typically N > 50 for robustness)
A = [ones(N, 1)  f_data]  # Design matrix: [1 f_x f_y f_z ...]
y = measurements         # Stacked sensor readings

# Normal equations
x_hat = (A' * A) \ (A' * y)

# Extract bias and scale
bias_hat = x_hat[1:6]
scale_hat = reshape(x_hat[7:end], 6, 6)

# Check residuals
residuals = y - A * x_hat
rmse = sqrt(mean(residuals.^2))
```

**Numerical Considerations**:

- If \(\text{cond}(A^T A)\) is large, use SVD: \(\text{x_hat} = (V \Sigma^{\dagger} U^T) \mathbf{y}\)
- Add regularization if underdetermined: \((A^T A + \lambda I)\mathbf{x} = A^T \mathbf{y}\)
- Validate on held-out test loads

---

## Summary

Linear algebra is the **mathematical foundation** of autonomous systems. Key takeaways:

| Concept | What It Does | Robotics Use |
|---|---|---|
| Vectors, norms | Represent state, measure distances | Position, velocity, quaternions |
| Matrices, transformations | Linear mappings, coordinate changes | Sensor models, feedback gains |
| Rank, nullspace | Characterize solvability, structure | Controllability, observability |
| Eigenvalues, eigenvectors | Stability, resonant modes | Pole placement, modal analysis |
| SVD, QR, Cholesky | Numerically robust factorizations | Least-squares, filtering, optimization |
| Jacobian, Taylor | Linearization, first-order models | Feedback linearization, MPC |

**Next Steps**: These tools are used throughout autonomous systems—in dynamics (rigid-body models), estimation (Kalman filters), and control (feedback design). The following chapters build on this foundation.

---

# Chapter 2: Probability and Information Theory

## I. The Role of Uncertainty in Robotics

Real robots operate in **noisy, uncertain environments**. Unlike laboratory simulations, autonomous systems must:

1. **Sense** with imperfect sensors (GPS drift, camera noise, lidar artifacts)
2. **Act** with noisy actuators (motor slip, compliance, delays)
3. **Reason** about incomplete information (partial observations, dynamic environments)

**Probabilistic Robotics** replaces point estimates with **probability distributions**, allowing systems to:

- Quantify uncertainty explicitly
- Make principled decisions under risk
- Fuse information from multiple sensors
- Adapt as uncertainty decreases

### Sources of Uncertainty

| Source | Mechanism | Impact |
|---|---|---|
| **Environment** | Unstructured, dynamic, stochastic behavior (e.g., pedestrians, weather) | Plan with contingencies, adapt online |
| **Sensors** | Physical limits (resolution, range), measurement noise, calibration errors | Estimate state from noisy observations |
| **Actuators** | Motor saturation, friction, compliance, timing delays | Control accounting for execution errors |
| **Model Mismatch** | Approximate dynamics, unmodeled effects | Learn models from data, use robust control |

---

## II. Foundations of Probability

### Probability Spaces and Axioms

A **probability space** \((\Omega, \mathcal{F}, P)\) consists of:

- **Sample space** \(\Omega\): set of all possible outcomes
- **\(\sigma\)-algebra** \(\mathcal{F}\): collection of events (measurable subsets), closed under countable unions and complements
- **Probability measure** \(P: \mathcal{F} \to [0, 1]\): assigns probability to events

**Axioms**:

1. \(P(\Omega) = 1\) (certainty)
2. \(P(\emptyset) = 0\) (impossibility)
3. Countable additivity: if events are disjoint, \(P(\cup_i A_i) = \sum_i P(A_i)\)

### Random Variables

A **random variable** \(X: \Omega \to \mathbb{R}\) is a function assigning real values to outcomes.

**Cumulative Distribution Function (CDF)**:
\[F_X(x) = P(X \leq x)\]

**Probability Mass Function (PMF)** (discrete):
\[p_X(x) = P(X = x)\]

**Probability Density Function (PDF)** (continuous):
\[f_X(x) = \frac{dF_X}{dx}\]

with \(\int_{-\infty}^{\infty} f_X(x) dx = 1\).

### Conditional Probability and Bayes' Rule

**Conditional Probability**:
\[P(A | B) = \frac{P(A \cap B)}{P(B)}\]

assuming \(P(B) > 0\).

**Bayes' Rule** (foundation of probabilistic inference):
\[P(\theta | z) = \frac{P(z | \theta) P(\theta)}{P(z)}\]

where:

- \(P(\theta | z)\): **posterior** (updated belief after observing \(z\))
- \(P(z | \theta)\): **likelihood** (how well \(\theta\) explains the data)
- \(P(\theta)\): **prior** (pre-data belief)
- \(P(z) = \int P(z | \theta) P(\theta) d\theta\): **evidence** (normalizing constant)

_Application_: In robot localization, Bayes' rule updates the belief over robot pose given range measurements.

### Independence and Conditional Independence

**Independence**: Events \(A\) and \(B\) are independent if \(P(A \cap B) = P(A)P(B)\), written \(A \perp B\).

**Conditional Independence**: \(A \perp B | C\) if \(P(A \cap B | C) = P(A | C)P(B | C)\).

**Factorization**: Conditional independence enables decomposing complex joint distributions:
\[P(X, Y, Z) = P(X | Y, Z) P(Y | Z) P(Z)\]

_Application_: Markov assumption in particle filters: current state conditionally independent of history given previous state.

---

## III. Expectation, Variance, and Covariance

### Expectation

The **expectation** (mean, expected value) is:

- **Discrete**: \(\mathbb{E}[X] = \sum_x x \, p_X(x)\)
- **Continuous**: \(\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x) dx\)

**Linearity of Expectation** (even for dependent variables):
\[\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]\]

### Variance

**Variance** measures spread:
\[\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2\]

**Properties**:

- \(\text{Var}(aX + b) = a^2 \text{Var}(X)\)
- For independent: \(\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)\)

### Covariance and Correlation

**Covariance**:
\[\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]\]

**Correlation**:
\[\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}} \in [-1, 1]\]

**Covariance Matrix** (for random vector \(\mathbf{X} \in \mathbb{R}^n\)):
\[\Sigma = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]\]

where \(\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]\). Properties:

- Symmetric: \(\Sigma^T = \Sigma\)
- Positive semidefinite: \(\Sigma \succeq 0\)

_Application_: In Kalman filters, the covariance matrix \(\Sigma\) quantifies estimation uncertainty; its eigenvalues reveal principal axes of uncertainty.

---

## IV. Key Probability Distributions

### Gaussian (Normal) Distribution

**PDF**:
\[f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\]

denoted \(\mathcal{N}(\mu, \sigma^2)\).

**Properties**:

- Symmetric around \(\mu\)
- \(68.3\%\) of mass within \(1\sigma\), \(95.4\%\) within \(2\sigma\), \(99.7\%\) within \(3\sigma\) (68–95–99.7 rule)
- Preserved under linear transformations: if \(X \sim \mathcal{N}(\mu, \Sigma)\), then \(AX + b \sim \mathcal{N}(A\mu + b, A\Sigma A^T)\)

**Multivariate Gaussian** (\(\mathbf{X} \in \mathbb{R}^n\)):
\[f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n \det(\Sigma)}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)\]

denoted \(\mathcal{N}(\boldsymbol{\mu}, \Sigma)\).

**Why Gaussian?**

- **Central Limit Theorem**: Sum of many independent random variables \(\approx\) Gaussian
- **Tractable**: Closed-form Bayesian inference, linear transformations
- **Sensor noise**: Often well-modeled as Gaussian
- **Computational**: Maximum entropy distribution with specified mean and covariance

_Application_: Kalman filter assumes Gaussian distributions; nonlinear systems use Extended Kalman Filter (EKF) with local Gaussian approximations.

### Other Important Distributions

| Distribution | PDF/PMF | Mean | Variance | Use Case |
|---|---|---|---|---|
| **Uniform** \(U(a,b)\) | \(\frac{1}{b-a}\) on \([a,b]\) | \(\frac{a+b}{2}\) | \(\frac{(b-a)^2}{12}\) | Unknown disturbances with known bounds |
| **Exponential** \(\text{Exp}(\lambda)\) | \(\lambda e^{-\lambda x}\) on \([0,\infty)\) | \(\frac{1}{\lambda}\) | \(\frac{1}{\lambda^2}\) | Waiting times, failure rates |
| **Chi-squared** \(\chi^2_k\) | (related to Gaussian) | \(k\) | \(2k\) | Quadratic forms \(\sum Z_i^2\) with \(Z_i \sim \mathcal{N}(0,1)\) |
| **Laplace** | \(\frac{1}{2b} e^{-\|x-\mu\|/b}\) | \(\mu\) | \(2b^2\) | Robust estimation (heavier tails than Gaussian) |

---

## V. Fundamental Theorems

### Law of Large Numbers (LLN)

**Weak LLN**: For i.i.d. samples \(X_1, X_2, \ldots\) with mean \(\mu\),
\[\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{p} \mu \quad \text{as } n \to \infty\]

(convergence in probability)

**Justification for Monte Carlo**: empirical average converges to true expectation.

### Central Limit Theorem (CLT)

For i.i.d. samples \(X_i\) with mean \(\mu\) and variance \(\sigma^2\),
\[\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)\]

(convergence in distribution to standard normal)

**Consequence**: Confidence interval for \(\mu\):
\[\bar{X}_n \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\]

where \(z_{\alpha/2}\) is the critical value (e.g., \(1.96\) for \(95\%\)).

**Why CLT matters in robotics**: Aggregate measurement noise (from many sources) is often approximately Gaussian even if individual sources are not.

### Markov and Chebyshev Inequalities

**Markov Inequality** (for \(X \geq 0\)):
\[P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}\]

**Chebyshev Inequality**:
\[P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}\]

**Use**: Model-free bounds when distribution is unknown. Conservative but guaranteed.

---

## VI. Bayesian Inference and Estimation

### Maximum Likelihood Estimation (MLE)

Given observations \(\mathbf{z} = \{z_1, \ldots, z_n\}\) and model with unknown parameters \(\boldsymbol{\theta}\), the **MLE** is:
\[\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} P(\mathbf{z} | \boldsymbol{\theta})\]

or equivalently (log-likelihood):
\[\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \sum_{i=1}^n \log P(z_i | \boldsymbol{\theta})\]

**Properties**:

- Asymptotically unbiased and efficient
- No prior information needed
- Often has closed-form solution (e.g., Gaussian likelihood \(\Rightarrow\) least-squares)

_Application_: Sensor calibration (estimate bias, scale).

### Maximum A Posteriori (MAP) Estimation

Incorporates **prior** belief:
\[\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} P(\boldsymbol{\theta} | \mathbf{z}) = \arg\max_{\boldsymbol{\theta}} P(\mathbf{z} | \boldsymbol{\theta}) P(\boldsymbol{\theta})\]

**Relationship to MLE**: \(\text{MAP} = \text{MLE}\) when prior is uniform.

**Regularization**: Informative prior acts as regularization (penalizes unlikely parameters).

### Conjugate Priors

If prior \(P(\boldsymbol{\theta})\) and likelihood \(P(\mathbf{z} | \boldsymbol{\theta})\) are **conjugate**, the posterior has the same functional form as the prior. Enables **sequential Bayesian updates**.

**Example**: Beta-Binomial conjugacy

- Prior: \(P(p) = \text{Beta}(\alpha, \beta)\)
- Likelihood: \(P(z | p) = \text{Binomial}(n, p)\)
- Posterior: \(P(p | z) = \text{Beta}(\alpha + \text{successes}, \beta + \text{failures})\)

_Application_: Sequential robot learning with minimal re-computation.

---

## VII. Information Theory

### Entropy

**Entropy** quantifies average information (or uncertainty):
\[H[X] = -\sum_x P(X=x) \log P(X=x)\]

(continuous: \(H[X] = -\int f(x) \log f(x) dx\))

**Interpretation**:

- High entropy: uniform distribution, high uncertainty
- Low entropy: concentrated distribution, low uncertainty
- **Minimum**: \(H = 0\) for deterministic variable (one outcome has probability 1)

**Units**: bits (log base 2) or nats (natural log).

### Mutual Information

**Mutual Information** measures dependence:
\[I[X; Y] = H[X] + H[Y] - H[X, Y]\]

or equivalently:
\[I[X; Y] = \mathbb{E}_{X,Y}\left[\log \frac{P(X,Y)}{P(X)P(Y)}\right]\]

**Interpretation**:

- \(I[X; Y] = 0\): \(X\) and \(Y\) are independent
- \(I[X; Y] > 0\): learning \(Y\) reduces uncertainty about \(X\)

_Application_: **Information gain** in active sensing; choose measurements that maximize information about unobserved state.

### Kullback–Leibler Divergence

**KL divergence** measures distribution distance:
\[D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_P[\log P(X) - \log Q(X)]\]

**Properties**:

- Non-symmetric: \(D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)\)
- Non-negative: \(D_{\text{KL}}(P \| Q) \geq 0\), with equality iff \(P = Q\)
- Not a true distance (fails triangle inequality)

_Application_: Variational inference; approximate intractable posterior \(P(\boldsymbol{\theta} | \mathbf{z})\) with tractable \(Q(\boldsymbol{\theta})\) by minimizing \(D_{\text{KL}}(Q \| P)\).

---

## VIII. Practical Example: Estimating Sensor Noise from Stationary Data

**Problem**: Characterize noise statistics of a gyroscope when robot is stationary.

**Setup**: Collect \(n = 1000\) samples over time \(T = 100\) s. True angular velocity \(\omega_{\text{true}} = 0\) (stationary).

**Assume Gaussian noise**: \(z_i \sim \mathcal{N}(0, \sigma^2)\).

**Estimators**:
\[\hat{\mu} = \frac{1}{n}\sum_i z_i, \quad \hat{\sigma}^2 = \frac{1}{n-1}\sum_i (z_i - \hat{\mu})^2\]

**Confidence interval** on variance (using chi-squared distribution):
\[\frac{(n-1)\hat{\sigma}^2}{\chi^2_{1-\alpha/2, n-1}} \leq \sigma^2 \leq \frac{(n-1)\hat{\sigma}^2}{\chi^2_{\alpha/2, n-1}}\]

**Implementation** (Python):

```python
import numpy as np
from scipy.stats import chi2

# Simulate gyro data
true_variance = 0.01  # rad/s)^2
gyro_data = np.random.normal(0, np.sqrt(true_variance), n=1000)

# Estimate mean and variance
mu_hat = np.mean(gyro_data)
sigma_hat = np.std(gyro_data, ddof=1)

# 95% confidence interval
n = len(gyro_data)
chi2_lower = chi2.ppf(0.025, n-1)
chi2_upper = chi2.ppf(0.975, n-1)

var_ci_lower = (n-1) * sigma_hat**2 / chi2_upper
var_ci_upper = (n-1) * sigma_hat**2 / chi2_lower

print(f"Estimated variance: {sigma_hat**2:.6f}")
print(f"95% CI: [{var_ci_lower:.6f}, {var_ci_upper:.6f}]")

# Check if true value is in CI
if var_ci_lower <= true_variance <= var_ci_upper:
    print("✓ True variance is in confidence interval")
else:
    print("✗ True variance outside CI (check for systematic bias)")
```

---

## Summary

Probability and information theory are essential for handling uncertainty in autonomous systems:

| Concept | Purpose | Application |
|---|---|---|
| **Probability spaces, random variables** | Rigorous mathematical framework | Formal modeling of uncertainty |
| **Bayes' rule** | Update beliefs given data | Sequential state estimation |
| **Expectation, variance, covariance** | Summarize distributions | Filter tuning, uncertainty quantification |
| **Gaussian distributions** | Natural noise model, tractable inference | Kalman filters, optimal control |
| **MLE, MAP** | Parameter estimation | Sensor calibration, learning |
| **Entropy, mutual information** | Quantify information | Active sensing, exploration |
| **Bayesian inference** | Integrate prior + likelihood | Robust decision-making under uncertainty |

**Next Chapter**: We apply these probability foundations to **state estimation**—the core task of perception in autonomous systems.
