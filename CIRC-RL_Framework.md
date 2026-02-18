# CIRC-RL: A Causal-Invariant Framework for Scientific Policy Discovery

## Abstract

We propose **Causal Invariant Regularized Constrained Reinforcement Learning (CIRC-RL)**, a methodological framework that replaces the standard optimize-and-hope paradigm of reinforcement learning with a process of **scientific discovery applied to sequential decision-making**. Rather than training neural networks to approximate unknown policy functions, CIRC-RL discovers explicit, falsifiable hypotheses about the structure of environment dynamics and reward mechanisms, deduces optimal policies as logical consequences of validated hypotheses, and reserves function approximation exclusively for residual components that resist analytic characterization.

The framework proceeds through a cycle inspired by the scientific method: (i) causal structure discovery via conditional independence testing, (ii) invariance verification across environment families, (iii) explicit hypothesis generation on functional forms via symbolic regression, (iv) systematic falsification of candidate hypotheses, (v) analytic derivation of optimal policies from validated hypotheses, and (vi) bounded residual learning for unexplained variance. Each component produces **falsifiable, interpretable, and diagnosticable** outputs -- when the framework fails, the failure is localized to a specific hypothesis, enabling targeted correction rather than opaque retraining.

We formalize each component mathematically, establish theoretical guarantees where possible, acknowledge fundamental limitations, and position this framework as a structural alternative to function-approximation-based RL for domains where the underlying mechanisms are at least partially discoverable.

---

## 1. Introduction

### 1.1 The Epistemological Problem in RL

Reinforcement learning suffers from a particularly severe form of overfitting due to four structural properties:

**Non-stationarity.** The agent's policy modifies the state distribution during training, violating the i.i.d. assumption fundamental to statistical learning theory.

**Reward hacking.** Agents exploit correlations in the reward function rather than learning the designer's intent, leading to solutions that achieve high training reward through unintended mechanisms.

**Temporal credit assignment.** The causal chain from action to reward spans multiple timesteps, enabling spurious correlations to dominate learning signals.

**Train-deployment mismatch.** The training environment inevitably differs from deployment conditions, and standard RL provides no structural guarantees of robustness to this shift.

Standard RL optimizes:

$$\pi^* = \arg\max_{\pi} \mathbb{E}_{\tau \sim \rho_\pi}[R(\tau)]$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory sampled under policy $\pi$ from the induced state-action distribution $\rho_\pi$. This objective is **distribution-specific**: the optimal policy depends critically on $\rho_\pi$, which in turn depends on the training environment dynamics, initial state distribution, and the policy itself.

However, the problems above are symptoms of a deeper issue. The standard RL pipeline -- collect data, optimize a parameterized function approximator, evaluate performance -- is not a process of **knowledge acquisition**. It is curve fitting with a reinforcement signal. The agent does not form hypotheses about why actions lead to rewards, does not deduce consequences of its assumptions, and cannot diagnose the source of its failures. It discovers nothing; it merely fits.

### 1.2 The Deeper Problem: RL is Not Science

Consider the contrast with the scientific method:

| Aspect | Scientific Method | Standard RL | CIRC-RL |
|--------|------------------|-------------|---------|
| **Starting point** | Formulate a hypothesis | Choose a network architecture | Discover causal structure |
| **Process** | Deduce consequences, test them | Optimize loss function | Generate hypotheses, falsify, deduce policy |
| **Output** | Falsifiable theory | Opaque weight vector | Explicit functional form + bounded residual |
| **On failure** | Identify which hypothesis failed | Retrain (no diagnosis) | Localize failure to specific hypothesis |
| **Generalization** | Theory predicts novel cases | Hope | Deduction from validated structure |

The core insight of this framework is: **the optimal policy for a well-understood system is not an object to be learned -- it is a consequence to be derived.** Learning enters only where understanding ends.

### 1.3 Core Principles

CIRC-RL is built on four principles, in lexicographic order:

1. **Safety** (Constraints): Encode domain knowledge as hard constraints that no policy may violate.
2. **Understanding** (Causal Discovery + Hypothesis Generation): Discover the causal and functional structure of the environment before attempting to act in it.
3. **Derivation over Approximation** (Analytic Policy): Derive optimal policies from validated structural knowledge; approximate only what cannot be derived.
4. **Parsimony** (MDL): Among equivalent explanations, prefer the simplest.

---

## 2. Foundations

### 2.1 Causal Framework for RL

We model the RL environment as a **Structural Causal Model (SCM)** consisting of:

**State Dynamics:**
$$S_{t+1} = f_s(S_t, A_t, U_s^t)$$

**Reward Generation:**
$$R_t = f_r(S_t, A_t, S_{t+1}, U_r^t)$$

where:
- $f_s: \mathcal{S} \times \mathcal{A} \times \mathcal{U}_s \to \mathcal{S}$ is the state transition function
- $f_r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{U}_r \to \mathbb{R}$ is the reward function
- $U_s^t, U_r^t$ are unobserved exogenous variables (noise, hidden confounders)

**Crucially, $f_s$ and $f_r$ are not arbitrary functions to be approximated -- they are mechanisms to be discovered.** The central epistemological shift of CIRC-RL is treating these functions as objects of scientific inquiry rather than targets of function approximation.

**Causal Graph.** We denote the causal graph as $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where vertices $\mathcal{V}$ include states, actions, rewards, and unobserved variables, and edges $\mathcal{E}$ represent direct causal influence.

**Interventional Distribution.** Following Pearl's do-calculus, we distinguish:
- **Observational distribution** $P(R_t | A_t = a)$: reward when we observe action $a$ was taken
- **Interventional distribution** $P(R_t | do(A_t = a))$: reward when we force action $a$, breaking incoming causal edges to $A_t$

The interventional distribution eliminates confounding and reveals true causal effects.

### 2.2 Environment Families and Invariance

**Definition 2.1 (Environment Family).** An environment family $\mathcal{E}$ is a set of MDPs $\lbrace M_e = (\mathcal{S}, \mathcal{A}, P_e, R_e, \gamma) : e \in \mathcal{E}\rbrace$ that share the same state space, action space, and causal graph $\mathcal{G}$, but may differ in the functional forms of $f_s$ and $f_r$ or the distribution of exogenous variables.

**Definition 2.2 (Causal Invariance).** A mechanism $m: \mathcal{X} \to \mathcal{Y}$ is causally invariant across environment family $\mathcal{E}$ if:

$$\forall e, e' \in \mathcal{E}: \quad P_e(Y | do(X)) = P_{e'}(Y | do(X))$$

That is, the interventional distribution is identical across environments.

**Definition 2.3 (Transition Mechanism Invariance).** State feature $s_i$ has an **invariant transition mechanism** if:

$$P_e(s_i' \mid s, a) = P_{e'}(s_i' \mid s, a) \quad \forall\, e, e' \in \mathcal{E}$$

This is the dual of reward mechanism invariance: it identifies which dimensions of the state space have dynamics that vary across environments.

**Assumption 2.1 (Shared Causal Structure).** Environments in $\mathcal{E}$ differ only in parametric variations or exogenous noise distributions, not in fundamental causal relationships. Formally, the causal graph $\mathcal{G}$ is invariant: $\mathcal{G}_e = \mathcal{G}_{e'}$ for all $e, e' \in \mathcal{E}$.

This assumption is **not verifiable a priori** but is empirically testable: if the causal graph varies, conditional independence patterns will differ across environments.

### 2.3 Complexity and Description Length

**Definition 2.4 (Kolmogorov Complexity).** The Kolmogorov complexity $K(\pi)$ of a policy $\pi$ is the length of the shortest program (in bits) that implements $\pi$ on a universal Turing machine.

Since $K(\pi)$ is uncomputable, we use tractable approximations:

**Symbolic Complexity:**
$$C_{\text{sym}}(h) = |h|_{\text{nodes}}$$
the number of nodes in the expression tree of a symbolic hypothesis $h$. This is the natural complexity measure for analytic functional forms and directly computable.

**Parametric Complexity:**
$$C_{\text{param}}(\pi_\theta) = |\theta|$$
the number of free parameters. For analytic policies, this is typically small (the constants in a closed-form expression). For residual neural components, it measures the approximator size.

**Functional Complexity:**
$$C_{\text{func}}(\pi) = \mathbb{E}_{s \sim \rho}[\mathcal{H}(\pi(\cdot|s))]$$
the expected entropy of the policy distribution.

**Assumption 2.2 (MDL Principle).** Among hypotheses achieving similar predictive accuracy, prefer those with lower description length. This encodes Occam's razor: simpler explanations are more likely to generalize because they have fewer degrees of freedom available for overfitting.

### 2.4 Safety Constraints

**Definition 2.5 (Constraint Function).** A constraint function $C_i: \mathcal{T} \to \mathbb{R}$ maps trajectories to real values representing cost or risk. A constraint is satisfied if:

$$\mathbb{E}_{\tau \sim \rho_\pi}[C_i(\tau)] \leq \delta_i$$

for some threshold $\delta_i \in \mathbb{R}$.

Constraints encode domain knowledge about unacceptable behaviors (e.g., collision probability, energy consumption, regulatory violations, maximum drawdown).

**Assumption 2.3 (Constraint Completeness).** The specified constraints $\lbrace C_i\rbrace$ adequately capture safety requirements for the domain. This assumption is pragmatic: we cannot formalize unknown unknowns, but we can formalize known risks.

---

## 3. The CIRC-RL Framework

### 3.1 Formal Objective

The CIRC-RL optimization problem is defined as a **lexicographic multi-objective program** with the following priority ordering:

**Priority 1 (Constraints).** Ensure all safety constraints are satisfied:

$$\forall i: \quad \mathbb{E}_{\tau \sim \rho_\pi}[C_i(\tau)] \leq \delta_i$$

**Priority 2 (Causal Invariance).** Maximize worst-case causal return across environment family:

$$\max_\pi \quad \min_{e \in \mathcal{E}} \mathbb{E}_{\tau \sim \rho_\pi^e}[R | do(\pi), e]$$

**Priority 3 (Stability).** Minimize variance in performance across environments:

$$\min_\pi \quad \text{Var}_{e \in \mathcal{E}}\left[\mathbb{E}_{\tau \sim \rho_\pi^e}[R(\tau)]\right]$$

**Priority 4 (Simplicity).** Among policies satisfying priorities 1-3, prefer those with minimal complexity:

$$\min_\pi \quad \alpha_1 C_{\text{sym}}(\pi) + \alpha_2 C_{\text{param}}(\pi) + \alpha_3 C_{\text{func}}(\pi)$$

Lexicographic ordering means: we only consider priority $k+1$ among policies that are Pareto-optimal for priorities $1, \ldots, k$.

### 3.2 Phase 1: Causal Structure Identification

1. Collect exploratory data from multiple environments $e \in \mathcal{E}$
2. Infer causal graph $\mathcal{G}$ via conditional independence testing (PC algorithm), score-based methods (GES with BIC/MDL), or algorithms handling latent confounders (FCI)
3. Identify causal parents of reward: $\text{Pa}_{\mathcal{G}}(R)$
4. Validate invariance of causal mechanisms across environments

**Domain Knowledge Integration.** Leverage expert knowledge to constrain the causal graph search: encode known causal relationships as hard constraints (e.g., in physical systems: force causes acceleration, not vice versa; in markets: order flow causes price impact, not vice versa).

**Hybrid Approach.** Combine domain knowledge (hard constraints on graph structure) with data-driven discovery (learn parameters and residual structure).

### 3.3 Phase 2: Feature Selection and Invariance Testing

**Reward Mechanism Invariance.** For each feature $f \in \mathcal{F}$:
1. Test whether $f \in \text{Anc}_{\mathcal{G}}(R)$ (ancestors of reward in causal graph)
2. Test stability of $P(R | do(f))$ across environments using Leave-One-Environment-Out (LOEO) $R^2$
3. Retain only features with stable causal effects: $\mathcal{F}_{\text{robust}} = \lbrace f : \text{Var}_e[P_e(R|do(f))] < \epsilon\rbrace$

**Transition Mechanism Invariance.** For each state dimension $s_i$:
1. Compute LOEO $R^2$ of predicting $s_i'$ from $(s, a)$
2. State dimensions with low $R^2$ have **variant dynamics** -- their transition mechanism changes across environments
3. State dimensions with high $R^2$ have **invariant dynamics** -- no normalization needed

**Dynamics Scale Estimation.** For each environment $e$ with variant dynamics:
1. Fit linear model $\Delta s \sim s + a$ and extract action-coefficient matrix $B_e \in \mathbb{R}^{d_s \times d_a}$
2. Compute dynamics scale $D_e = \|B_e\|_F$
3. Compute reference scale $D_{\text{ref}} = \frac{1}{|\mathcal{E}|} \sum_e D_e$

The dynamics scale $D_e$ quantifies how much one unit of action affects state transitions in environment $e$. The ratio $r_e = D_e / D_{\text{ref}}$ will be used in the analytic policy derivation (Phase 5) to normalize actions across environments.

### 3.3b Phase 2b: Observation Analysis and Canonical Reparametrization

Before hypothesis generation, the raw observation space may contain **algebraic constraints** that reduce the effective dimensionality. Detecting and exploiting these constraints is critical for symbolic regression: redundant coordinates inflate the search space, and trigonometric identities (e.g., $\cos^2\theta + \sin^2\theta = 1$) create degenerate features that confuse SR.

**Constraint Detection.** For each pair of observation dimensions $(s_i, s_j)$:
1. Compute $r_{ij} = s_i^2 + s_j^2$ across all samples
2. If $\text{std}(r_{ij}) / \text{mean}(r_{ij}) < \epsilon$ (e.g., $\epsilon = 0.001$), identify a **circle constraint**: $(s_i, s_j)$ are the cosine and sine of some underlying angle

**Canonical Reparametrization.** When a circle constraint is detected on $(s_i, s_j)$:
1. Compute the canonical angle: $\phi = \text{atan2}(s_j, s_i)$
2. Replace the two constrained dimensions with a single angular dimension $\phi$
3. Build a canonical dataset where states are represented as $(\phi, s_{\text{remaining}})$

This reduces the state dimensionality (e.g., from 3 to 2 for the pendulum: $(\cos\theta, \sin\theta, \dot\theta) \to (\phi_0, s_2)$) and provides physically meaningful coordinates for symbolic regression.

**Coordinate-Aware Pipeline.** When canonical coordinates are available:
- Dynamics hypotheses are generated and tested in **canonical space** (Phase 3-4)
- Reward hypotheses are generated in **observation space** (since the reward function is defined over raw observations)
- The analytic policy (Phase 5) plans in canonical space, with bidirectional mappings between canonical and observation spaces

**Derived Features.** Observation-space features may be pre-computed from canonical coordinates for use in reward hypothesis generation. For example, $\theta = \text{atan2}(\sin\theta, \cos\theta)$ is a derived feature enabling compact reward expressions like $R = -(\theta^2 + 0.1\dot\theta^2 + 0.001 a^2)$.

### 3.4 Phase 3: Structural Hypothesis Generation

**This phase replaces the neural policy optimization of classical RL with scientific hypothesis generation.** The goal is to discover explicit, analytic functional forms for the environment dynamics and reward mechanism.

#### 3.4.1 Dynamics Hypotheses

For each state dimension $s_i$ identified as having **variant dynamics** in Phase 2, we seek the functional form:

$$\Delta s_i = h_i(s, a; \theta_e)$$

where $\theta_e$ are the environment parameters. The tool is **symbolic regression** (e.g., PySR, SINDy), which takes as input $(s, a, \theta_e)$ and target $\Delta s_i$, and proposes candidate analytic expressions ordered by complexity and accuracy -- a Pareto front of explicit hypotheses.

**Procedure:**
1. Pool exploration data from all training environments (using canonical coordinates when available from Phase 2b)
2. Run symbolic regression with complexity penalty across **multiple random seeds** (e.g., 3 runs with different seeds), generating a Pareto front of candidate expressions $\lbrace h_i^{(1)}, h_i^{(2)}, \ldots, h_i^{(K)}\rbrace$ ordered from simplest to most complex. Multi-seed runs mitigate the stochasticity of SR search and increase Pareto front coverage.
3. Each candidate is an **explicit, falsifiable hypothesis** about the functional form of the dynamics
4. **Operator constraints** prevent degenerate expressions: nested constraints (e.g., $\sin(\sin(\cdot))$ forbidden) and complexity caps on operator arguments keep the search space tractable

**Example (Inverted Pendulum).** Symbolic regression on angular velocity data might return:
- $h^{(1)}$: $\Delta\omega \approx c_1 \cdot a$ (too simple, ignores gravity)
- $h^{(2)}$: $\Delta\omega \approx c_1 \cdot \sin(\theta) / l + c_2 \cdot a / (ml^2)$ (structurally correct)
- $h^{(3)}$: 15-term expression (overfitting)

**Example (Market Microstructure).** Symbolic regression on mid-price changes might return:
- $h^{(1)}$: $\Delta p \approx \lambda \cdot q$ (pure linear impact)
- $h^{(2)}$: $\Delta p \approx \kappa(\mu - p)\Delta t + \lambda \cdot q + \sigma\epsilon$ (mean-reverting + linear impact)
- $h^{(3)}$: $\Delta p \approx \kappa(\mu - p)\Delta t + \lambda \cdot \text{sign}(q)|q|^{0.5} + \sigma\epsilon$ (mean-reverting + square-root impact)

#### 3.4.2 Reward Hypotheses

If the reward mechanism is invariant (verified in Phase 2), seek:

$$R = g(s, a)$$

with no dependence on $\theta_e$. If not invariant, seek $R = g(s, a; \theta_e)$ with explicit parametric dependence.

The same symbolic regression approach applies, but reward hypotheses are generated in **observation space** (not canonical space), since the reward function is typically defined over raw observations. Derived features from Phase 2b (e.g., $\theta = \text{atan2}(\sin\theta, \cos\theta)$) are included as additional SR features to enable compact reward expressions.

In many domains, the reward function is **designer-specified** (not learned), in which case this step is trivial: the reward hypothesis is the reward function itself.

#### 3.4.3 The Hypothesis Register

Each candidate hypothesis is formally registered with:
- **Expression**: the analytic functional form
- **Complexity**: number of nodes in expression tree (symbolic complexity $C_{\text{sym}}$)
- **Training fit**: $R^2$ on pooled training data
- **Status**: untested / validated / falsified

### 3.5 Phase 4: Hypothesis Falsification

**This is the core Popperian step.** Each hypothesis from Phase 3 is subjected to systematic falsification. A hypothesis that survives all tests is provisionally accepted; one that fails any test is rejected.

**Coordinate-Aware Falsification.** When canonical coordinates are used (Phase 2b), the falsification engine runs in two passes:
1. **Dynamics pass**: tests dynamics hypotheses with canonical variable names and canonical data
2. **Reward pass**: tests reward hypotheses with observation-space variable names and original data

This separation is critical because dynamics and reward hypotheses live in different coordinate systems. Without it, reward hypotheses using observation-space names (e.g., $\theta$) would be incorrectly falsified against canonical names (e.g., $\phi_0$).

#### 3.5.1 Cross-Environment Structural Consistency

If the hypothesis specifies a parametric relationship between environment parameters and dynamics coefficients, this relationship must hold independently in each environment.

**Procedure:**
1. For hypothesis $h_i$, estimate its coefficients independently in each environment $e$, yielding $\hat{\alpha}_e, \hat{\beta}_e, \ldots$
2. If $h_i$ predicts that $\hat{\alpha}_e \propto g(\theta_e)$ for some function $g$, verify that the observed coefficients are consistent with this prediction across all training environments
3. Quantify consistency via a statistical test (e.g., F-test on the residuals of the predicted relationship)

**Example:** If $h^{(2)}$ for the pendulum predicts $\beta_e \propto 1/(m_e l_e^2)$, estimate $\beta_e$ independently in each environment and test whether $\beta_e \cdot m_e l_e^2 = \text{const}$ within statistical uncertainty.

**Falsification criterion:** If the predicted structural relationship fails with $p < 0.01$, the hypothesis is falsified.

#### 3.5.2 Out-of-Distribution Prediction

Hold out a subset of environments $\mathcal{E}_{\text{held-out}} \subset \mathcal{E}$. The hypothesis, calibrated on $\mathcal{E}_{\text{train}} = \mathcal{E} \setminus \mathcal{E}_{\text{held-out}}$, must produce **quantitative predictions** for the dynamics in $\mathcal{E}_{\text{held-out}}$.

**Procedure:**
1. From the validated hypothesis and the known environment parameters $\theta_{e'}$ of a held-out environment $e'$, predict the dynamics coefficients $\hat{\alpha}_{e'}, \hat{\beta}_{e'}, \ldots$
2. Estimate the true coefficients from data in $e'$
3. Compare: predicted vs. observed, with confidence intervals

**Falsification criterion:** If the predicted coefficient falls outside the 99% confidence interval of the observed value, the hypothesis is falsified for that environment. If falsified in $> 20\%$ of held-out environments, the hypothesis is globally rejected.

#### 3.5.3 Trajectory Prediction

The most stringent test: use the hypothesis to simulate trajectories forward in time, and compare predicted state sequences to observed ones.

**Procedure:**
1. From initial state $s_0$ and a sequence of actions $(a_0, \ldots, a_{T-1})$, use the hypothesis to predict $(s_1, \ldots, s_T)$
2. Compare predicted trajectory to observed trajectory
3. Quantify divergence as a function of horizon

**Falsification criterion:** If the predicted trajectory diverges from observed by more than a threshold $\epsilon_T$ (which grows with horizon $T$) in more than 20% of test trajectories, the hypothesis is falsified.

#### 3.5.4 Selection Among Surviving Hypotheses

Among hypotheses that survive all falsification tests, select using the **Pareto front with $R^2$ threshold** method:

1. Construct the Pareto front of validated hypotheses in (complexity, $R^2$) space, sorted from simplest to most complex
2. Walk the front from simplest to most complex; return the **first hypothesis whose training $R^2 \geq \tau$** (default $\tau = 0.999$)
3. If no hypothesis meets the threshold, fall back to the highest-$R^2$ hypothesis on the Pareto front

This replaces pure MDL/BIC scoring, which exhibits a pathology on large datasets: the $O(n)$ data-fit term dominates the $O(\log n)$ complexity penalty, causing selection of overly complex expressions that overfit to noise.

**Best-effort fallback.** When no validated hypothesis exists for a target variable (all were falsified), the system selects the hypothesis with the highest training $R^2$ regardless of validation status, logged as a best-effort fallback. This preserves usability even when falsification is overly strict, while clearly marking the hypothesis as unvalidated.

For reference, the MDL score $\text{MDL}(h) = -\log P(D | h) + C_{\text{sym}}(h) \cdot \log(n)$ is still computed and stored in the hypothesis register for diagnostic purposes.

### 3.6 Phase 5: Analytic Policy Derivation

**Given validated hypotheses for dynamics and reward, derive the optimal policy as a logical consequence -- do not learn it.**

The form of the validated dynamics hypothesis determines the derivation method:

#### 3.6.1 Linear Dynamics + Quadratic Reward

If validated dynamics are linear: $s_{t+1} = As_t + Ba_t + c$ and the reward is quadratic: $R_t = -s_t^T Q s_t - a_t^T R a_t$, the optimal policy is a **Linear-Quadratic Regulator** (LQR):

$$a^*_t = -K \cdot s_t$$

where $K = (R + B^T P B)^{-1} B^T P A$ and $P$ solves the Discrete Algebraic Riccati Equation (DARE):

$$P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A$$

**Zero free parameters. Zero training. Zero overfitting.** The matrix $K$ is a consequence of the validated hypothesis.

**Cross-environment normalization.** If the dynamics matrix $B$ varies across environments as $B_e$, the policy in environment $e$ is:

$$a^*_t(e) = -K_e \cdot s_t, \quad K_e = (R + B_e^T P_e B_e)^{-1} B_e^T P_e A$$

Each $K_e$ is computed analytically from the known $B_e$. No "dynamics predictor" neural network is needed -- the relationship between environment parameters $\theta_e$ and $B_e$ is given by the validated hypothesis.

#### 3.6.2 Nonlinear Known Dynamics + Known Reward

When the validated dynamics $s_{t+1} = h(s_t, a_t; \theta_e)$ are nonlinear but fully specified, use optimal control methods:

**Pontryagin Maximum Principle.** Derive necessary conditions for optimality via the Hamiltonian:

$$H(s, a, \lambda, t) = R(s, a) + \lambda^T h(s, a; \theta_e)$$

where $\lambda$ is the costate variable. The optimal action satisfies $\partial H / \partial a = 0$, and the costate evolves as $\dot{\lambda} = -\partial H / \partial s$.

**Hamilton-Jacobi-Bellman.** For stochastic systems, the value function satisfies:

$$V^*(s) = \max_a \left[ R(s, a) + \gamma \mathbb{E}[V^*(h(s, a; \theta_e) + \sigma\epsilon)] \right]$$

For specific functional forms of $h$, this PDE may have closed-form solutions or efficient numerical solutions (e.g., via spectral methods, finite differences).

**Iterative LQR (iLQR).** For nonlinear dynamics with a known analytic form, the **iterative Linear-Quadratic Regulator** (Tassa et al. 2012) provides an efficient trajectory optimization method:

1. **Forward pass**: roll out the trajectory under the current action sequence
2. **Backward pass**: linearize dynamics (Jacobians $A_t, B_t$) and quadraticize cost at each timestep, then solve the Riccati recursion to obtain feedforward $k_t$ and feedback $K_t$ gains
3. **Line search**: apply the control update $u_t = \bar{u}_t + \alpha k_t + K_t(x_t - \bar{x}_t)$ with backtracking on $\alpha$
4. **Regularization**: Levenberg-Marquardt damping on $Q_{uu}$ ensures positive-definiteness

The closed-loop policy uses time-varying feedback gains: $a_t = \bar{a}_t + K_t (s_t - \bar{s}_t)$, providing robustness to perturbations from the nominal trajectory.

**Multi-start optimization.** iLQR is a local optimizer that converges to the nearest local optimum. For systems with non-convex cost landscapes (e.g., pendulum swing-up), the zero-initialized action sequence may converge to a poor local minimum. Multi-start optimization runs iLQR from $N$ random action initializations $u^{(i)} \sim \mathcal{U}[-u_{\max}, u_{\max}]$ in addition to the default (zero or warm-start), returning the best solution:

$$\bar{u}^* = \arg\max_{i \in \{0, \ldots, N\}} J\bigl(\text{iLQR}(s_0, u^{(i)})\bigr)$$

This is embarrassingly parallel and dramatically improves performance on high-inertia systems where the swing-up requires multi-swing energy pumping that a single zero-init run cannot discover.

When analytic Jacobians are available (computed symbolically via $\partial h / \partial s$ and $\partial h / \partial a$ from the validated hypothesis), iLQR avoids finite-difference approximation and gains both speed and numerical accuracy.

**Model Predictive Control.** For complex nonlinear dynamics where closed-form solutions are unavailable, solve the optimization problem online over a receding horizon:

$$a^*_t = \arg\max_{a_t, \ldots, a_{t+H}} \sum_{k=0}^{H} \gamma^k R(s_{t+k}, a_{t+k}) \quad \text{s.t.} \quad s_{t+k+1} = h(s_{t+k}, a_{t+k}; \theta_e)$$

The validated dynamics hypothesis serves as the **model** in model predictive control, but unlike learned neural world models, this model is an explicit analytic expression with known error bounds from the falsification phase. iLQR is the default MPC solver in CIRC-RL: the planning horizon equals the episode length, and replanning occurs at regular intervals with the previous solution as warm-start.

#### 3.6.3 Action Normalization for Varying Dynamics

When the dynamics are **action-multiplicative** (Theorem 3.8.1 below), the optimal policy admits a decomposition:

$$\pi^*(a \mid s; e) = \mathcal{N}_e\bigl(\pi^*_{\text{abs}}(a \mid s)\bigr)$$

where $\pi^*_{\text{abs}}$ is the optimal policy in a **normalized action space** (independent of $e$) and $\mathcal{N}_e$ rescales actions by the dynamics ratio $r_e = D_e / D_{\text{ref}}$.

For the analytic policy, this means: derive $\pi^*_{\text{abs}}$ once (using the reference dynamics $D_{\text{ref}}$), then obtain the environment-specific policy via:

$$a^*_t(e) = r_e \cdot \tilde{a}^*_t$$

where $\tilde{a}^*_t = \pi^*_{\text{abs}}(s_t)$ is the action in normalized space.

This is not an approximation imposed by architectural choice (as in the previous framework version) -- it is a **consequence of the validated dynamics hypothesis**. If the hypothesis specifies that the dynamics are action-multiplicative with scale $D_e$, the normalization follows logically.

#### 3.6.4 Constraint Integration

Safety constraints (Priority 1) are incorporated into the analytic derivation:

**For LQR:** Constrained LQR adds inequality constraints to the Riccati equation solution, solvable via quadratic programming at each step.

**For Pontryagin/HJB:** Constraints enter as path constraints in the optimal control formulation, handled via augmented Lagrangian or barrier methods.

**For MPC:** Constraints are directly included in the optimization problem:

$$a^*_t = \arg\max_{a_t, \ldots, a_{t+H}} \sum_{k=0}^{H} \gamma^k R(s_{t+k}, a_{t+k}) \quad \text{s.t.} \quad \begin{cases} s_{t+k+1} = h(s_{t+k}, a_{t+k}; \theta_e) \\ C_i(\tau) \leq \delta_i \quad \forall i \end{cases}$$

### 3.7 Phase 6: Bounded Residual Learning

The validated hypothesis explains a fraction of the dynamics variance. Let $\eta^2$ denote the fraction of variance explained. If $\eta^2$ is sufficiently high (e.g., $> 0.90$), the analytic policy from Phase 5 is the primary policy, and the residual is small.

For the residual component:

$$s_{t+1} = h_{\text{validated}}(s_t, a_t; \theta_e) + \underbrace{\epsilon_t}_{\text{residual}}$$

where $\epsilon_t$ captures the unexplained variance, we learn a **correction policy**:

$$\pi(a \mid s; e) = \pi_{\text{analytic}}(a \mid s; e) + \delta\pi(a \mid s)$$

The correction $\delta\pi$ is learned via standard RL methods (e.g., PPO), but with critical constraints:

1. **Bounded magnitude**: $\|\delta\pi\| \leq \eta_{\max} \cdot \|\pi_{\text{analytic}}\|$ where $\eta_{\max}$ is proportional to the unexplained variance fraction $(1 - \eta^2)$. This prevents the residual from dominating the analytic component.

2. **No access to environment parameters**: $\delta\pi$ depends only on state $s$, not on $\theta_e$. All environment-dependent adaptation is handled by the analytic component.

3. **Complexity penalty**: Standard MDL regularization on the residual network.

**When is residual learning unnecessary?** If $\eta^2 > 0.98$ and the analytic policy achieves return within $\epsilon$ of the theoretical optimum in held-out environments, skip this phase entirely. The analytic policy is sufficient.

**When is residual learning insufficient?** If $\eta^2 < 0.70$, the dynamics hypothesis is too coarse. Return to Phase 3 and generate richer hypotheses rather than relying on a large residual correction.

### 3.8 Theoretical Results

#### Theorem 3.8.1 (Invariance of Optimal Abstract Policy)

If (1) the reward mechanism $P_e(R \mid s, a)$ is invariant across $\mathcal{E}$, and (2) the transition dynamics are **action-multiplicative**, i.e.:

$$f_s(s, a; e) = g(s) + B_e \cdot a + U_s$$

then the optimal abstract policy $\pi_{\text{abs}}^*$ that maximizes worst-case return is invariant: $\pi_{\text{abs}}^*(a \mid s)$ does not depend on $e$.

*Proof Sketch.* Under action-multiplicative dynamics, applying normalizer $\mathcal{N}_e$ to abstract action $\tilde{a}$ yields physical action $a = r_e \cdot \tilde{a}$, producing effective transition $B_e \cdot a = B_e \cdot (D_e / D_{\text{ref}}) \cdot \tilde{a}$. Since $D_e = \|B_e\|_F$, this equalizes the amplitude of action effects in abstract space (up to directional effects in the structure of $B_e$). Given that the reward depends on state (invariant mechanism), and the effective dynamics in abstract action space are equalized, the optimal abstract policy is invariant. $\square$

*Limitation.* The result is exact only for dynamics that are linear in the action. For nonlinear systems, the decomposition is a local approximation whose error grows with action magnitude and dynamics nonlinearity. However, when the dynamics hypothesis is validated via falsification (Phase 4), the regime of validity is empirically bounded.

#### Theorem 3.8.2 (Causal Invariance Implies Generalization)

Let $\mathcal{E}$ be an environment family with shared causal graph $\mathcal{G}$. If the dynamics hypothesis $h$ passes all falsification tests in Phase 4 on training environments $\mathcal{E}_{\text{train}}$, and the analytic policy $\pi_h$ is derived from $h$ via Phase 5, then for a novel environment $e' \in \mathcal{E} \setminus \mathcal{E}_{\text{train}}$ whose parameters $\theta_{e'}$ fall within the convex hull of training parameters:

$$P\left(R^{e'}(\pi_h) \geq R_{\text{predicted}} - \epsilon\right) \geq 1 - \delta$$

where $R_{\text{predicted}}$ is the return predicted by the hypothesis, $\epsilon$ depends on the residual variance $(1 - \eta^2)$ and the horizon, and $\delta$ depends on the number of falsification tests survived.

*Key difference from v1:* The guarantee is **conditional on the hypothesis being correct** (which was empirically tested), not on a hope that function approximation generalizes. The failure mode is diagnosable: if the guarantee fails, either the hypothesis is wrong (go back to Phase 3) or the novel environment is outside the family $\mathcal{E}$ (which the framework explicitly acknowledges as a limitation).

#### Theorem 3.8.3 (MDL Bound for Symbolic Policies)

Let $\Pi_K = \lbrace\pi_h : C_{\text{sym}}(h) \leq K\rbrace$ be the class of analytic policies derived from symbolic hypotheses with complexity at most $K$. The sample complexity required to falsify incorrect hypotheses and identify a correct one in $\Pi_K$ is:

$$n = O\left(\frac{K \cdot \log(|\mathcal{H}_K|) + \log(1/\delta)}{\epsilon^2}\right)$$

where $|\mathcal{H}_K|$ is the number of candidate hypotheses at complexity $K$, $\epsilon$ is the tolerance, and $\delta$ is the failure probability.

*Implication.* Symbolic hypotheses have dramatically lower effective complexity than neural networks (typically $K \sim 10$--$30$ nodes vs. $10^4$--$10^6$ parameters), yielding exponentially lower sample requirements for reliable identification.

#### Theorem 3.8.4 (Constraint Satisfaction)

If constraints $\lbrace C_i\rbrace$ are satisfied by the analytic policy in all training environments, and the dynamics hypothesis is validated, then with probability at least $1 - \delta$ over deployment:

$$P_{\text{deploy}}\left(C_i(\tau) > \delta_i\right) \leq \delta_i + O\left(\sqrt{\frac{\log(m/\delta)}{n}}\right) + O(1 - \eta^2)$$

where $n$ is the number of training trajectories, $m$ the number of constraints, and the last term accounts for residual model error.

*Implication.* The constraint violation probability has an additive term proportional to the unexplained variance. High-quality hypotheses ($\eta^2 \to 1$) yield tighter safety guarantees.

### 3.9 Phase 7: Diagnostic Validation

**The final phase tests not "does the policy work?" but the entire causal chain from hypothesis to outcome.** This is the key advantage over RL with neural networks: failures are localizable.

#### 3.9.1 Premise Test (Hypothesis Validity)

In held-out test environments $\mathcal{E}_{\text{test}}$:
1. Are the dynamics observed compatible with the validated hypothesis?
2. Quantify: prediction error on $\Delta s$ given $(s, a, \theta_e)$

**If this test fails:** The dynamics hypothesis is wrong or incomplete. Return to Phase 3 with richer functional forms. The policy is not broken -- the understanding is incomplete.

#### 3.9.2 Derivation Test (Policy Correctness)

1. Do the trajectories generated by the analytic policy follow the trajectories predicted by the hypothesis?
2. Quantify: divergence between predicted and observed state sequences under the policy

**If this test fails but 3.9.1 passes:** The derivation method (LQR, Pontryagin, MPC) has a bug or the constraints are interfering unexpectedly. This is a computational issue, not a structural one.

#### 3.9.3 Conclusion Test (Return Prediction)

1. Is the return obtained in test environments compatible with the return predicted by the theory?
2. Quantify: $|R_{\text{observed}} - R_{\text{predicted}}| / R_{\text{predicted}}$

**If this test fails but 3.9.1 and 3.9.2 pass:** The reward hypothesis is wrong. Return to Phase 3 for the reward model.

#### 3.9.4 Diagnostic Summary

| Test | Passes | Fails | Action |
|------|--------|-------|--------|
| Premise (dynamics) | ✓ | ✗ | Return to Phase 3: richer dynamics hypothesis |
| Derivation (trajectory) | ✓ | ✗ | Debug derivation method (computational issue) |
| Conclusion (return) | ✓ | ✗ | Return to Phase 3: richer reward hypothesis |
| All three | ✓ | — | Policy validated: deploy with monitoring |

This diagnostic capability is **impossible** with neural network policies, where a failure provides essentially zero information about its cause.

---

## 4. Fundamental Limitations

### 4.1 Epistemological Boundaries

**No Free Lunch.** Even with infinite computational resources, there exists no algorithm that performs optimally across all possible environments. Every framework embeds inductive biases. CIRC-RL's bias is toward environments with discoverable analytic structure.

**Hume's Problem of Induction.** We cannot logically guarantee that validated hypotheses will hold in deployment. Empirical falsification reduces risk but does not eliminate it.

**Uncomputable Ideals.** Kolmogorov complexity is uncomputable, and exact causal discovery is NP-hard. All practical implementations are approximations.

**Symbolic Regression Limits.** Symbolic regression is not guaranteed to find the correct functional form, especially for high-dimensional problems or systems with no compact analytic description.

### 4.2 Untestable Assumptions

1. **Causal graph correctness**: Causal discovery from observational data admits multiple compatible graphs (Markov equivalence classes). Domain knowledge is essential to disambiguate.

2. **No hidden confounders**: We assume all relevant variables are observed. Unobserved confounders can invalidate causal conclusions.

3. **Environment family structure**: We assume deployment environments belong to the same family $\mathcal{E}$ as training environments. Black swan events lie outside any finite family.

4. **Constraint completeness**: We assume specified constraints capture all safety requirements. Unknown unknowns are by definition not captured.

5. **Analytic describability**: We assume the dynamics have a compact analytic form. Some systems (turbulence, protein folding, complex social dynamics) may not admit such descriptions, limiting the applicability of Phases 3-5.

### 4.3 Computational Challenges

**Symbolic regression** has exponential search space in expression complexity. Current methods (PySR, SINDy) scale well to $\sim 10$ input variables but struggle with $> 50$.

**Optimal control** for nonlinear systems may be computationally expensive, especially with many constraints or long horizons.

**Falsification** across many environments and many hypotheses requires careful experimental design to manage multiple testing corrections.

### 4.4 What CIRC-RL Does NOT Guarantee

- Zero overfitting (impossible in principle)
- Generalization to arbitrary distribution shifts (only to shifts within the validated hypothesis regime)
- Safety under unspecified threats (only under formalized constraints)
- Applicability to all domains (requires partially discoverable analytic structure)
- Computational efficiency (symbolic search and optimal control can be expensive)

### 4.5 What CIRC-RL DOES Provide

- **Falsifiable outputs**: Every component produces a testable claim, not an opaque weight vector
- **Diagnosticable failures**: When the framework fails, the failure is localized to a specific hypothesis or derivation step
- **Principled generalization**: Policies generalize because they are consequences of validated structural knowledge, not because a function approximator happened to extrapolate correctly
- **Minimal overfitting surface**: Analytic policies have $O(10)$ free parameters vs. $O(10^6)$ for neural policies
- **Domain knowledge integration**: The hypothesis generation and falsification cycle naturally incorporates expert knowledge as constraints on the search space
- **Interpretability**: The policy is an explicit formula whose logic can be inspected, audited, and understood

---

## 5. Relationship to Existing Work

### 5.1 Causal Reinforcement Learning

**Causal Confusion (de Haan et al., 2019).** Identifies the problem of correlational policies. CIRC-RL addresses this at a deeper level: instead of regularizing a neural policy to prefer causal features, it discovers the causal mechanisms explicitly and derives the policy from them.

**Counterfactual Data Augmentation (Buesing et al., 2018).** Uses learned world models for counterfactual reasoning. CIRC-RL uses validated analytic models, which are interpretable and have known error bounds.

### 5.2 Symbolic and Physics-Informed Methods

**PySR (Cranmer, 2023).** State-of-the-art symbolic regression. CIRC-RL uses PySR (or similar) as a component in Phase 3, embedded in a falsification framework that PySR alone does not provide.

**SINDy (Brunton et al., 2016).** Sparse identification of nonlinear dynamics. Closely related to Phase 3 dynamics discovery, but SINDy assumes a library of candidate functions while CIRC-RL's symbolic regression is more general.

**Physics-Informed Neural Networks (Raissi et al., 2019).** Encode known physics as constraints on neural networks. CIRC-RL goes further: it discovers the physics (or quasi-physics) first, then derives the policy without neural networks.

### 5.3 Robust RL and Domain Randomization

**Domain Randomization (Tobin et al., 2017).** CIRC-RL subsumes this: Phase 1 uses multiple environments for causal discovery, but instead of training a neural policy to be robust across them, it discovers why environments differ and exploits this understanding analytically.

**EPOpt (Rajeswaran et al., 2017).** Ensemble policy optimization. CIRC-RL replaces the ensemble of neural policies with a single analytic policy (or a small set derived from different valid hypotheses).

### 5.4 Invariant Risk Minimization

**IRM (Arjovsky et al., 2019).** Proposes learning invariant predictors via neural network regularization. CIRC-RL's Phase 2 performs invariance testing explicitly, and Phase 3 discovers the invariant structure analytically rather than regularizing a neural network toward it.

### 5.5 Optimal Control

**Classical Optimal Control (Pontryagin, Bellman, Kalman).** CIRC-RL's Phase 5 is classical optimal control. The novelty is not in the control method but in the **automated pipeline** from raw multi-environment data to validated analytic models to derived controllers, with explicit falsification at each step.

**Model Predictive Control.** MPC with a learned model is well-established. CIRC-RL's contribution is that the model is not a neural network but a validated symbolic expression with known error bounds and interpretable structure.

### 5.6 Safe RL

**Constrained MDPs (Altman, 1999).** CIRC-RL builds on CMDPs but adds: causal structure discovery, analytic policy derivation, and tighter constraint guarantees via validated dynamics models.

**CPO (Achiam et al., 2017).** Compatible with CIRC-RL's residual learning phase (Phase 6) as an optimization method for the bounded correction.

### 5.7 Minimum Description Length

**MDL Principle (Rissanen, 1978).** CIRC-RL applies MDL naturally: symbolic complexity of hypotheses provides a direct, meaningful complexity measure, unlike parameter counting in neural networks where architectural choices dominate.

---

## 6. Practical Implementation Considerations

### 6.1 Tool Chain

| Phase | Primary Tools | Fallback |
|-------|--------------|----------|
| Phase 1 (Causal Discovery) | PC algorithm, GES, FCI | Domain expert elicitation |
| Phase 2 (Invariance Testing) | LOEO $R^2$, linear regression | Nonparametric tests |
| Phase 2b (Observation Analysis) | Circle constraint detection, atan2 reparametrization | Manual coordinate identification |
| Phase 3 (Hypothesis Generation) | PySR (multi-seed symbolic regression), SINDy | Manual hypothesis formulation from domain knowledge |
| Phase 4 (Falsification) | Coordinate-aware two-pass falsification, Pareto $R^2$ threshold selection | Cross-validation, MDL/BIC scoring |
| Phase 5 (Policy Derivation) | LQR/DARE solvers, iLQR with multi-start optimization, MPC | Shooting methods, CasADi |
| Phase 6 (Residual Learning) | PPO with bounded correction | SAC, TD3 |
| Phase 7 (Validation) | Trajectory simulation, statistical comparison | Monte Carlo testing |

### 6.2 Environment Family Construction

**Simulation-Based.** For simulatable domains, construct $\mathcal{E}$ via:
- Parametric variation: Randomize physics parameters within plausible ranges
- Procedural generation: Generate diverse configurations
- Adversarial generation: Generate configurations that stress-test hypotheses

**Real-World Data.** For non-simulatable domains:
- Temporal environments: Treat different time periods as different environments
- Contextual environments: Partition data by regime (market regime, weather condition, operational mode)
- Geographic environments: Treat different locations as different environments

### 6.3 When to Use CIRC-RL

**Strong fit (use Phases 1-7):**
- Domains with known or discoverable physical/economic laws
- Systems with moderate state/action dimensionality ($< 50$)
- Applications requiring interpretability and auditability
- Safety-critical domains where diagnostic capability is essential

**Partial fit (use Phases 1-2, then classical RL with causal features):**
- High-dimensional systems where symbolic regression struggles
- Domains with no compact analytic description
- Rapid prototyping where the full pipeline is too slow

**Poor fit (use standard RL):**
- Purely perceptual domains (image-based control)
- Systems with no environment family available
- Problems where the reward function is the only source of information

### 6.4 Validation Protocol

**Multi-Level Holdout:**
1. **Training environments** $\mathcal{E}_{\text{train}}$: Used for Phases 1-3 (discovery, hypothesis generation)
2. **Falsification environments** $\mathcal{E}_{\text{falsify}}$: Used for Phase 4 (hypothesis testing)
3. **Test environments** $\mathcal{E}_{\text{test}}$: Used for Phase 7 (diagnostic validation)
4. **Deployment monitoring**: Continuous evaluation on live data with circuit breakers

**Critical: $\mathcal{E}_{\text{falsify}}$ and $\mathcal{E}_{\text{test}}$ must be strictly held out.** Data from these environments must never inform hypothesis generation or policy derivation.

---

## 7. Open Problems and Future Directions

### 7.1 Theoretical Frontiers

**Tighter Generalization Bounds.** Develop bounds that exploit the specific structure of symbolic hypotheses (expression tree depth, operation types) rather than generic VC/Rademacher bounds.

**Sample Complexity of Symbolic Discovery.** How many samples and environments are required to reliably discover the correct functional form via symbolic regression? This connects to the theory of identification in algebraic statistics.

**Compositional Hypothesis Spaces.** Develop methods for composing sub-hypotheses (e.g., "the dynamics decompose into a gravity term + a friction term + an actuation term") to handle high-dimensional systems.

### 7.2 Algorithmic Developments

**Scalable Symbolic Regression.** Current methods struggle beyond $\sim 10$ input variables. Developing symbolic regression that scales to $\sim 100$ variables would dramatically expand the applicability of CIRC-RL.

**Automated Falsification Design.** Optimally design falsification experiments: which environments to hold out, which trajectory initial conditions to test, to maximize the probability of catching incorrect hypotheses.

**Adaptive Hypothesis Refinement.** When a hypothesis is falsified, automatically suggest refinements (e.g., "add a quadratic term", "allow parameter interaction") rather than restarting symbolic regression from scratch.

### 7.3 Domain-Specific Instantiations

**Financial Markets.** Market microstructure has well-established analytic models (Almgren-Chriss, Avellaneda-Stoikov, Kyle). CIRC-RL can validate which model best describes a given market regime and derive optimal execution/market-making strategies as consequences.

**Robotics.** Rigid body dynamics are analytically tractable (Lagrangian/Hamiltonian mechanics). CIRC-RL can discover effective parameters (friction, mass distribution) and derive controllers via validated dynamics.

**Energy Systems.** Grid dynamics follow known differential equations with uncertain parameters. CIRC-RL can identify regime-dependent parameters and derive optimal dispatch policies.

### 7.4 Connections to Philosophy of Science

**Lakatos' Research Programs.** CIRC-RL's hypothesis refinement cycle mirrors Lakatos' concept of progressive research programs: a hard core (causal graph) surrounded by a protective belt (parametric hypotheses) that is modified in response to anomalies.

**Kuhn's Paradigm Shifts.** When a dynamics hypothesis is fundamentally falsified (not just in parameters but in structure), CIRC-RL must return to Phase 1 and reconsider the causal graph -- a "paradigm shift" within the framework.

**Popper's Falsificationism.** Phase 4 is explicitly Popperian. The strength of a hypothesis is measured not by the data it fits but by the tests it survives.

---

## 8. Philosophical Foundations

### 8.1 Epistemic Humility

CIRC-RL acknowledges fundamental limits:

**We cannot eliminate uncertainty.** All learning is inductive; we cannot deduce the future from the past with logical certainty.

**We cannot verify assumptions.** Causal structure, environment families, and constraint completeness are assumptions, not derivable truths.

**We cannot prevent all failures.** Black swan events lie outside any finite training distribution.

**What we can do:** Construct frameworks that produce **falsifiable, interpretable, and diagnosticable** outputs, so that when failures occur, they are informative rather than opaque.

### 8.2 Derivation vs. Approximation

The central philosophical claim of CIRC-RL is that there is a fundamental difference between:

**Derived knowledge:** "The optimal action is $a^* = -Ks$ because the dynamics are $s' = As + Ba$ and the reward penalizes $s^TQs + a^TRa$, and $K$ is the unique solution of the Riccati equation." -- This is a chain of reasoning. Each link can be tested independently. The conclusion follows from the premises with logical necessity.

**Approximated knowledge:** "The optimal action is whatever the neural network outputs after training." -- This is a black box. When it fails, nothing is learned about why.

CIRC-RL maximizes the fraction of the policy that is derived and minimizes the fraction that is approximated. The residual (Phase 6) is an honest acknowledgment of the limits of current understanding, not a substitute for understanding.

### 8.3 The Scientific Method as Operational Framework

CIRC-RL embodies the scientific method not as metaphor but as algorithm:

1. **Observation** (Phase 1): Collect data, discover causal structure
2. **Hypothesis** (Phase 3): Propose explicit functional forms
3. **Prediction** (Phase 4): Derive quantitative consequences
4. **Experimentation** (Phase 4): Test predictions in held-out environments
5. **Falsification** (Phase 4): Reject hypotheses that fail tests
6. **Theory** (Phase 5): Derive policy from surviving hypothesis
7. **Application** (Phase 7): Deploy with monitoring, ready to revise

This cycle is not a one-shot process. CIRC-RL is designed for iteration: a falsified hypothesis leads to a richer one, which leads to a better policy, which reveals new anomalies, which lead to further refinement. The framework converges not toward a fixed policy but toward a progressively deeper understanding of the environment.

### 8.4 Pragmatic Truth

Following Peirce and James, we define "useful knowledge" not as correspondence to absolute truth, but as:

**Robustness**: Predictions that remain stable under perturbations

**Falsifiability**: Clear failure modes that can be detected and corrected

**Actionability**: Enables better decisions than alternatives

**Transferability**: Applies beyond narrow training conditions

**Interpretability**: Can be inspected, audited, and understood by humans

CIRC-RL targets *useful* policies in this pragmatic sense -- policies that are consequences of our best current understanding of the environment, with explicit acknowledgment of what we do not understand.

---

## 9. Conclusion

We have presented **CIRC-RL v2**, a methodological framework that transforms reinforcement learning from a process of function approximation into a process of scientific discovery.

**Key contributions:**

- **Epistemological shift**: From "optimize a neural network" to "discover structure, derive policy, learn residual"
- **Explicit hypothesis generation**: Symbolic regression discovers analytic functional forms for dynamics and reward
- **Systematic falsification**: Hypotheses are tested via cross-environment consistency, OOD prediction, and trajectory prediction before being used for policy derivation
- **Analytic policy derivation**: Optimal policies are derived as logical consequences of validated hypotheses via classical optimal control methods (LQR, Pontryagin, HJB, MPC)
- **Bounded residual learning**: Neural networks are used only for unexplained variance, with explicit bounds on their contribution
- **Diagnostic validation**: Failures are localized to specific hypotheses or derivation steps, enabling targeted correction
- **Formal integration**: Causal inference, invariance testing, symbolic regression, optimal control, and constrained optimization are unified in a coherent pipeline

**Central thesis:**

The optimal policy for a well-understood system is not an object to be learned -- it is a consequence to be derived. Learning enters only where understanding ends. CIRC-RL operationalizes this principle: it automates the cycle from data to causal structure to analytic hypothesis to falsification to derived policy, producing outputs that are falsifiable, interpretable, and diagnosticable at every stage.

**The question is not** "*Can we train a neural network to approximate the optimal policy?*" (Usually yes, but without guarantees.)

**The question is** "*Can we understand the environment well enough to derive the optimal policy?*" (Often yes, and with much stronger guarantees.)

CIRC-RL is an answer to the latter question -- a framework that prioritizes understanding over approximation, derivation over fitting, and falsifiable structure over opaque performance.

---

## Acknowledgments

This framework synthesizes ideas from causal inference (Pearl, Spirtes, Glymour), robust statistics (Huber, Hampel), information theory (Rissanen, Cover), robust optimization (Ben-Tal, Nemirovski), safe RL (Altman, Achiam), symbolic regression (Cranmer, Brunton), optimal control (Pontryagin, Bellman, Kalman), philosophy of science (Popper, Lakatos, Kuhn, Peirce), and domain randomization (Tobin, Sadeghi). We acknowledge these foundational contributions while taking responsibility for any errors or overreach in the synthesis.

---

## References

*[Standard academic references would be listed here in a full paper format. Key citations include: Pearl (2009) on causality, Arjovsky et al. (2019) on IRM, Rissanen (1978) on MDL, Altman (1999) on constrained MDPs, Cranmer (2023) on PySR, Brunton et al. (2016) on SINDy, Popper (1959) on falsificationism, Pontryagin et al. (1962) on optimal control, and numerous others mentioned throughout the text.]*

---

**Document Status:** Methodological Framework v2.0

**Changelog from v1.0:**
- Replaced neural policy optimization (Phase 3) with scientific hypothesis discovery, falsification, and analytic policy derivation (Phases 3-6)
- Added symbolic regression as primary tool for hypothesis generation
- Added explicit falsification protocol (Phase 4) with cross-environment, OOD, and trajectory tests
- Added diagnostic validation (Phase 7) with localized failure analysis
- Integrated transition dynamics normalization from Extension 3.7 as a consequence of the analytic framework
- Revised theoretical results to account for the new pipeline
- Expanded philosophical foundations to formalize the epistemological shift

**Intended Use:** Foundation for research, implementation, and critical evaluation

**Limitations:** This is a framework, not a complete algorithm. Domain-specific instantiations require additional design choices and validation. The framework is most applicable to domains with partially discoverable analytic structure.

**License:** AGPL-3.0 -- Open methodology for scientific and commercial use with attribution and copyleft