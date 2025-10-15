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

***

## Outline

The book is divided into two primary parts: **Foundations** (focusing on
Applied Math and Machine Learning Basics) and **Algorithms** (focusing on
specialized applications in Robotics and AI).

### Part I: Foundations (Applied Mathematics & Machine Learning Basics)

This section lays the mathematical groundwork for th

1. **Mathematics Core**
    * **Linear Algebra**
    * **Probability and Information Theory**
    * **Numerical Computation**
    * **Mechanics**
    * **Machine Learning Basics**
2. **System Modeling Core**
    * **Multi-armed Bandits**
    * **Finite Markov Descion Process**
    * **Dynamic Programming**
    * **Monte Carlo Methods**
3. **Deep Learning Core**
    * **Deep Feedforward Networks**
    * **Regularization for Deep Learning**
    * **Optimization for Training Deep Learning**
    * **Convolutional Networks**
    * **Sequence Modeling: Recurrent and Recursive**

### Part II: Algorithms and Specialized Applications

This part focuses on applying the foundational math to complex robotics and AI
problems, detailing algorithms used for estimation, mapping, control, and deep
learning architectures.

#### A. Navigation and Localization @@Thrun2005ProbabilisticRobotics

This section focuses on algorithms, typically derived from Bayes filters, for position estimation and map acquisition.

1. **Recursive State Estimation Techniques**
    * **Gaussian Filters**
    * **Nonparametric Filters**
2. **Mechanics of Robots**
    * **Motion**
    * **Perception**
3. **Localization**
    * **Markov**
    * **Gaussian**
4. **Mapping**
    * **Simulataneous Localization and Mapping (SLAM)**
    * **GraphSLAM**
    * **Sparse Extended Information Filter**
    * **FastSLAM**
5. **Descion Making**
    * **Markov Descion Processes (MDP)**
    * **Partially Observable Markov Descion Processes (POMDP)**
    * **Approximate POMDP**
    * **Exploration**

#### B. Advanced Deep Learning @@Goodfellow2016DeepLearning

This section covers modern learning paradigms that use hierarchical representations, often learned via neural networks with multiple layers, to model complex patterns in data.

1. **Linear Factor Models**
2. **Autoencoders**
3. **Representation Learning**
4. **Structured Probablistic Models**
5. **Monte-Carlo Methods**
6. **Deep Generative Models**

#### C. Reinforcement Learning @@Sutton2018ReinforcementLearning

This section details methods for learning how an agent should take actions in an environment to maximize a cumulative reward signal.

1. **On-policy Prediction/Control with Approximation**
2. **Off-policy Methods with Approximation**
3. **Eligibility Traces**
4. **Policy Gradient Methods**
