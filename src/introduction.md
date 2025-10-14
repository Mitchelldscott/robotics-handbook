# Data-driven Design of Autonomous Systems

## Introduction

This book is a consolidation of foundations and algorithms for data-driven
autonomous systems. The material is distilled from key reference texts in
Machine Learning and Probabilistic Robotics. The goal is to provide a model-
based design handbook for robotics.

### Learning Methodology

This first part of the book provides an overview of probability, linear algebra
and differential equations, to get a deeper understanding of these topics
readers should use a more in-depth resource. The second part of this book is
meant to be like [rust-by-example](https://doc.rust-lang.org/rust-by-example/)
and provide hands on mini-projects.

## Book Organization

The book is divided into two primary parts: **Foundations** (focusing on
Applied Math and Machine Learning Basics) and **Algorithms** (focusing on
specialized applications in Robotics and AI).

### Part I: Foundations (Applied Mathematics & Machine Learning Basics)

This section lays the mathematical and statistical groundwork necessary to
comprehend the probabilistic nature of modern robotics and deep learning.
Topics covered include:

1. **Mathematical Core:**
    * **Linear Algebra**.
    * **Probability and Information Theory**. Probabilistic robotics
    explicitly models uncertainty using probability theory.
    * **Numerical Computation** and **Optimization:** Including the notions of
    gradient, gradient descent, stochastic gradient descent, Momentum
    techniques (RMSPROP, Nesterov’s acceleration), and the ADAM/NADAM
    algorithms.
2. **Core Machine Learning Concepts:**
    * **Machine Learning Basics** (including supervised and unsupervised
    learning).
    * **Bayes Filters:** The recursive algorithm for state estimation that
    forms the basis for virtually every technique presented in probabilistic
    robotics.

### Part II: Algorithms and Specialized Applications

This part focuses on applying the foundational math to complex robotics and AI
problems, detailing algorithms used for estimation, mapping, control, and deep
learning architectures.

#### A. Probabilistic Robotics (Localization and Mapping) @@citation Thrun2005ProbabilisticRobotics

This section focuses on algorithms, typically derived from Bayes filters, for
position estimation and map acquisition.

1. **Recursive State Estimation Techniques:**
    * **Gaussian Filters:** The Kalman Filter (KF) and Extended Kalman Filter
    (EKF), often applied to feature-based maps.
    * **Nonparametric Filters:** Histogram Filters (Grid Localization) and
    the **Particle Filter (PF)** (Monte Carlo Localization - MCL). PF/MCL are
    well-suited for complex multimodal beliefs and global uncertainty.
2. **Core Robotics Models:**
    * **Motion Models:** Including the odometry motion model, derived from
    kinematics, which describes the outcome of a control probabilistically.
    * **Measurement Models:** Describing how sensor measurements are
    generated, including feature-based sensor models for landmarks.
3. **Simultaneous Localization and Mapping (SLAM):**
    * **Feature-Based SLAM:** Algorithms like **FastSLAM 2.0** (a particle
    filter-based SLAM) and **GraphSLAM** (a graph-based SLAM algorithm) will
    be implemented and compared. The aim is often to determine which algorithm
    produces more accurate maps or has smaller computational latency.
    * **Graph Optimization:** Utilizing concepts like the Extended Information
    Filter (EIF) SLAM, recognizing the sparse structure of information
    constraints between poses and features.

#### B. Deep Learning and Reinforcement Learning

@@citation Goodfellow2016DeepLearning Sutton2018ReinforcementLearning

This section covers modern learning paradigms that often rely on function
approximation to deal with large state spaces.

1. **Deep Learning Architectures:**
    * **Deep Feedforward Networks**.
    * **Convolutional Networks (CNNs)** and **Sequence Modeling** (Recurrent
    and Recursive Nets, including LSTMs).
    * **Generative Models:** Autoencoders, Variational Autoencoders (VAEs),
    and Generative Adversarial Networks (GANs).
2. **Reinforcement Learning (RL):**
    * **Formal Frameworks:** Markov Decision Processes (MDPs) and Partially
    Observable Markov Decision Processes (POMDPs).
    * **Value Function Methods (Q and V):** Covering key concepts like
    Temporal-Difference (TD) learning, Monte Carlo (MC) methods, and the
    combination of the two.
    * **Control Algorithms:** Details on algorithms like **Q-learning**
    (off-policy) and **SARSA** (on-policy).
    * **Advanced Concepts:** Including n-step methods, **Eligibility Traces**
    (the $\lambda$-return and the equivalence between forward and
    backward views), and **Policy Gradient Methods**.
