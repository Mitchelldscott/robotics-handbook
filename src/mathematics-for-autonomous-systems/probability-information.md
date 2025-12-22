# Probability and Information Theory

This crash course/cheat sheet summarizes the fundamentals of probability and
information theory as they apply to robotics, focusing on core concepts
essential for state estimation, perception, and planning under uncertainty.

## I. The Role of Uncertainty in Robotics

Probability theory is **very important** to robotics, forming the core of
robotic perception. Robots operate in the physical world, where uncertainty is
inherent and unavoidable.

| Source of Uncertainty       | Description                                                                                                                                      |
| :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Environments**            | Unstructured, dynamic, and unpredictable worlds (e.g., highways, private homes).                                                                 |
| **Sensors (Perception)**    | Sensors are inherently limited by physical laws (range, resolution) and are subject to **noise**, perturbing measurements in unpredictable ways. |
| **Robots (Action/Control)** | Actuators (motors) are imperfect and subject to control noise, wear-and-tear, and unpredictable effects (e.g., wheel slip or motion drift).      |

The goal of **Probabilistic Robotics** is to represent this uncertainty
explicitly using the calculus of probability theory. Instead of relying on a
single "best guess," information is represented by **probability distributions**
over possible hypotheses, enabling robots to accommodate various sources of
uncertainty and operate robustly.

## II. Core Probabilistic Concepts

| Concept                                | Definition/Use in Robotics                                                                                                                                                                                                         | Source(s) |
| :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------- |
| **State (\(x_t\))**                    | Represents the configuration of the robot and its environment (e.g., pose, joint configuration, object velocities). A state is **complete** if it is the best predictor of the future; otherwise, it is an **incomplete state**.   |           |
| **Belief Distribution (\(bel(x_t)\))** | The robot's internal knowledge about the true state, represented by a **conditional probability distribution** over state variables, conditioned on all available data (measurements \(z*{1:t}\) and controls \(u*{1:t}\)).        |           |
| **Prior (\(P(X)\))**                   | The probability distribution summarizing knowledge before incorporating new data.                                                                                                                                                  |           |
| **Likelihood (\(P(Z\mid X)\))**        | The probability of receiving sensor data \(Z\) given a certain state \(X\). Often referred to in robotics as the "generative model".                                                                                               |           |
| **Posterior (\(P(X\mid Z)\))**         | The updated belief about the state \(X\) after incorporating the observation \(Z\).                                                                                                                                                |           |
| **Bayes' Rule**                        | The mathematical foundation for incrementally digesting new sensor information: \[P(X\mid Z) = \frac{P(Z\mid X)P(X)}{P(Z)}\].                                                                                                      |           |
| **Markov Assumption**                  | Postulates that the current state \(x*t\) is a **complete summary of the past**; past data and future data are independent if the current state is known (\(P(x_t\mid x*{0:t-1}, z*{1:t-1}, u*{1:t}) = P(x*t\mid x*{t-1}, u_t)\)). |           |

## III. Foundational Estimation Algorithms (Bayes Filters)

**Bayes Filters (Recursive State Estimation)** are the principal algorithms for
calculating the belief recursively, where the belief at time \(t\) is calculated
from the belief at time \(t-1\).

The Bayes filter alternates two steps:

1. **Prediction Step (Motion Update):** Modifies the belief in accordance with
   the executed action (control \(u*t\)). This typically **increases
   uncertainty** due to motion noise. Requires the **State Transition
   Probability** \(P(x_t|u_t, x*{t-1})\).
2. **Measurement Update Step:** Integrates sensor data (\(z_t\)) into the
   present belief. This typically **decreases uncertainty**. Requires the
   **Measurement Probability** \(P(z_t|x_t)\) or \(P(z_t|x_t, m)\).

## IV. Key State Estimation Implementations

### A. Gaussian Filters (Parametric)

Gaussian filters rely on approximating the belief distribution \(bel(x_t)\)
using its first and second moments (mean \(\mu\) and covariance \(\Sigma\)).
They are confined to **unimodal Gaussian distributions** and are usually
appropriate for tracking applications where uncertainty is local.

-   **Kalman Filter (KF):** A classic algorithm for state estimation, originally
    developed for **linear systems**.
-   **Extended Kalman Filter (EKF):** Solves state estimation for **nonlinear
    systems** by **linearizing the nonlinear model** at each time step and then
    applying the Kalman filter to the linearized model. EKF localization is
    often applied to **feature-based maps**.
-   **Unscented Kalman Filter (UKF):** Can provide a more appropriate solution
    for localization problems in nonlinear systems by **avoiding the
    linearization step** used in the EKF.
-   **Extended Information Filter (EIF):** Uses canonical parameters
    (\(\Omega\), \(\xi\)) to represent the probability in logarithmic form.
    **Information integration is achieved by summing up information** from
    multiple robots, making it suitable for decentralized integration. EIF SLAM
    is particularly useful when the information matrix is sparse.

### B. Non-Parametric Filters

These filters are suitable for representing **complex multi-modal beliefs** and
are the method of choice when dealing with global uncertainty or ambiguous
situations.

-   **Particle Filters (PF) / Monte Carlo Localization (MCL):** A recursive
    Bayesian filter that represents the posterior distribution using a weighted
    set of samples (or "particles"). MCL has become one of the most popular
    localization algorithms in robotics, capable of solving **global
    localization** and the **kidnapped robot problem**.

    -   **Core MCL Steps (Iterative):**

    1. **Motion Prediction:** Shift the particle distribution, allowing it to
       spread out (increases uncertainty) by applying the motion model with
       added zero-mean noise terms (e.g., Gaussian distribution).
    2. **Measurement Update:** Determine the **importance weight**
       (\(w^{[m]}\_t\)) of each particle based on how well the particle's
       predicted state matches the actual sensor measurement using the
       measurement model (likelihood function).
    3. **Resampling:** Draw new particles from the temporary set with
       probability proportional to their weights. This focuses the particle
       distribution on high-likelihood regions.

    -   **Handling Failures:** Adding **random particles** (e.g., in proportion
        to the decay of short-term measurement likelihood relative to the
        long-term average) allows MCL to recover from localization failures (the
        kidnapped robot problem).

-   **Grid Localization:** Approximates the posterior using a **histogram
    filter** over a grid decomposition of the pose space. It can represent
    multi-modal distributions. The resolution of the grid involves a trade-off
    between accuracy and computational efficiency.

## V. Information Theory and Metrics

Information theory provides a mathematical framework to study complex systems
and quantify uncertainty.

| Concept                                            | Definition/Interpretation                                                                                                                                                                                                         | Relevance to Robotics                                                                                                                                                                                                                    | Source(s) |
| :------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------- |
| **Entropy (\(H[X]\))**                             | Quantifies the amount of **uncertainty** involved in the value of a random variable \(X\). It measures how random, variable, or uncertain one should be about \(X\). It is the minimum mean description length.                   | Negative entropy can be used as a **belief-dependent reward function** for search tasks, where the objective is to **reduce uncertainty** (maximizing information gain).                                                                 |           |
| **Mutual Information (\(I[X; Y]\))**               | Measures the **reduction in description length** from using dependencies between variables \(X\) and \(Y\). It quantifies how much can be learned about what was sent (\(X\)) from what is received (\(Y\)) over a noisy channel. | Used for inference of community structure, identifying correlations between brain stimuli/responses, and maximizing information flow in deep learning.                                                                                   |           |
| **Relative Entropy (Kullback-Leibler Divergence)** | Measures the **excess description length** from guessing the wrong distribution. It is a measure of distance between two probability density functions (pdfs).                                                                    | Used to track propagation of political discourses, understand social disruption, and in machine learning models.                                                                                                                         |           |
| **Fisher Information Matrix (FIM)**                | Relates to the curvature of the log-likelihood function around the maximum. Provides a metric for **quantifying the potential uncertainty reduction** of a measurement configuration.                                             | The inverse of the FIM is the **Cramér-Rao lower bound (CRLB)**, which provides the minimum variance achievable for any unbiased estimator. Used in Information-rich Rapidly-exploring Random Tree (IRRT) algorithm for motion planning. |           |

## VI. Probabilistic Decision Making and Planning

Action selection in robotics requires algorithms to cope with uncertainty, often
necessitating trade-offs between yielding immediate rewards and gathering
information for long-term success. This is modeled through decision-making
frameworks.

### A. Markov Decision Processes (MDPs)

-   **Assumption:** State is **fully observable** at all times.
-   **Focus:** Addresses uncertainty in action effects (stochastic action
    outcomes \(P(s'|s, a)\)).
-   **Solution:** **Value Iteration** computes an optimal policy (a mapping from
    states to actions) defined over the entire state space, allowing the robot
    to accommodate non-determinism by reacting appropriately to its observed
    state.

### B. Partially Observable Markov Decision Processes (POMDPs)

-   **Assumption:** The state is **not fully observable**; measurements are
    noisy projections of the state.
-   **Focus:** Deals with both stochastic action effects and noisy sensing. The
    robot must plan in **belief space** (the space of all possible belief
    distributions \(b\)).
-   **Definition:** A POMDP framework is defined by the tuple \(\langle S, A, T,
    O, R, \gamma, b_0 \rangle\):
    -   \(S\): Set of states.
    -   \(A\): Set of actions.
    -   \(T(s', a, s)\): **Transition model**, \(P(s'|s, a)\), the probability
        of reaching state \(s'\) given current state \(s\) and action \(a\).
    -   \(O(o', s', a)\): **Observation model**, \(P(o'|s', a)\), the
        probability of perceiving observation \(o'\) if state \(s'\) was reached
        after action \(a\).
    -   \(R\): Bounded **reward function**.
    -   \(\gamma\): Discount factor.
    -   \(b_0\): Initial probability mass function (pmf) over states.
-   **Key Advantage:** POMDP planning allows the robot to **actively pursue
    information gathering** (exploration) while maximizing expected utility
    (exploitation).
-   **Applications:** Localization and navigation, autonomous driving, search
    and tracking, manipulation, and human-robot interaction.

### C. Information-Theoretic Motion Planning

This field leverages information theory principles (like FIM and entropy) to
drive robot trajectories.

-   **Objective:** Maximize the information content gathered while minimizing
    resource costs.
-   **Adaptive Sampling:** A sequential decision problem where sensing
    configurations are selected to best reduce uncertainty about an unknown
    quantity.
-   **IRRT Algorithm:** **Information-rich Rapidly-exploring Random Tree**
    algorithm, which extends the RRT algorithm by embedding metrics on
    uncertainty reduction (predicted using FIM) at the tree growth and path
    selection levels. IRRT generates dynamically feasible paths and is suited
    for constrained sensing platforms.
