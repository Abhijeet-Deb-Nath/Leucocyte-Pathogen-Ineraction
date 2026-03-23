# Theory And Formulation

## 1. Problem Class

The current task should be formalized as a partially observable Markov decision process:

`M = (S, A, O, T, Omega, R, gamma)`

where:

- `S` is the latent simulator state
- `A` is the macrophage action space
- `O` is the observation space available to the controlled agent
- `T` is the transition kernel induced by macrophage, bacteria, neutrophil, nutrient, and chemokine dynamics
- `Omega` is the observation map from full state to local observation
- `R` is the training reward
- `gamma` is the discount factor used in optional RL fine-tuning

This is the correct theoretical framing because the learner controls only the macrophage, while bacteria and neutrophils evolve as part of the environment state.

## 2. Latent State

The latent state at time `t` includes at least the following components:

- macrophage position and health
- bacterial positions, health, and phenotype states
- neutrophil positions and remaining lifespan
- nutrient field
- chemokine field
- recruitment queue state
- accumulated tissue damage
- recent chemokine burden and episode time

The student policy does not observe this state directly.

## 3. Observation Model

The learned controller receives a local centered observation built in `simulator/rl_interface.py`.

It contains:

- a local grid tensor with blocked cells, nutrient, chemokine, visible bacteria, visible neutrophils, and macrophage location
- scalar features such as local counts, local chemokine peak, normalized health, and directional offsets
- optional belief-style memory features derived from recent history, including last seen bacteria compartment, strongest recent chemokine compartment, and visited compartments

Formally, the student acts on histories:

`h_t = (o_0, a_0, o_1, a_1, ..., o_t)`

The recurrent policy is therefore an approximation to a history-dependent policy `pi(a_t | h_t)`.

## 4. Action Space

At the environment interface level, the cleaned learned-controller action set is:

`A = {move_up, move_down, move_left, move_right, stay, attack, request_help}`

Internally, the hierarchical policy factorizes control into:

- macro mode `m_t` in `{engage, request_help, follow_chem, patrol}`
- micro action `u_t` in `{move_up, move_down, move_left, move_right, stay, attack}`

The executed action is produced by:

`a_t = g(m_t, u_t, s_t)`

where `g` is the action-resolution function.

For `request_help`, `g` maps the abstract escalation request into the strongest currently valid concrete signal action exposed by the environment. This abstraction reduces action branching while preserving the policy's escalation decision.

## 5. Objective

The environment-level host utility is:

`U_t = K_t - alpha * D_t - beta * I_t`

where:

- `K_t` is cumulative bacterial kill progress
- `D_t` is tissue damage
- `I_t` is immune usage cost

The final evaluation reports:

- `win_rate`
- `tissue_damage`
- `host_utility`
- `episode_length`
- `recruited_neutrophils`

Optional RL fine-tuning does not optimize final host utility directly. Instead, the wrapper reward is shaped around utility difference and local decision-specific bonuses/penalties:

`r_t = U_{t+1} - U_t + shaping_t`

This distinction matters in the paper. The benchmark is ultimately evaluated on episode-level outcomes, while the RL ablation is trained on a shaped per-step surrogate.

## 6. Why Plain Behavior Cloning Fails Here

In standard supervised imitation, training examples are sampled from teacher-induced state histories. At deployment, the learner induces its own histories. Under partial observability, even small action errors can change future observations enough to push the learner outside the teacher distribution.

This is the classic covariate-shift problem in imitation learning. Ross and Bagnell show that treating sequential imitation as i.i.d. supervised learning can produce compounding error that worsens with horizon length:

- Ross and Bagnell, 2010, "Efficient Reductions for Imitation Learning"
  - https://proceedings.mlr.press/v9/ross10a.html

The `BC-only` result in this project is consistent with that analysis. On the held-out `100`-seed benchmark, `BC-only` falls to `0.02` win rate, showing that sequence-aware recurrence alone is not enough when the policy is trained only on teacher trajectories.

## 7. Why DAgger Is The Correct Mainline Here

DAgger-style aggregation changes the training distribution by collecting labels on learner-induced states. In practical terms, the student is rolled out, the heuristic teacher relabels those visited states, and the supervised dataset is expanded accordingly.

In this project, that matters for two reasons:

1. The task is partially observed, so recovery states after early mistakes are common.
2. The delayed-support mechanic makes timing errors costly and trajectory-dependent.

The empirical result supports the theoretical expectation:

- `BC-only` collapses
- `BC + DAgger` becomes competitive with the heuristic

Therefore the current paper should present DAgger not as a minor training detail, but as the mechanism that makes imitation viable in this benchmark.

## 8. Why RL Fine-Tuning Is Only A Negative Ablation

The current RL stage starts from a stronger imitation prior, then applies small-sample PPO-style fine-tuning with shaped rewards. In principle this could improve policy quality. In practice, it does not.

The likely reason is a mismatch between:

- a strong teacher-shaped prior
- noisy recurrent rollouts
- a small amount of policy-gradient data
- a shaped reward that does not perfectly coincide with the benchmark's episode-level evaluation objective

The correct interpretation is therefore not "RL never helps." The correct interpretation is:

- the current RL fine-tuning regime is not a reliable improvement operator in this benchmark

## 9. Formal Hypotheses Supported By The Current Paper

The manuscript can support the following hypotheses.

### H1

Under partial observability and delayed support, plain sequence-aware behavior cloning is insufficient for competitive control.

### H2

Aggregating teacher labels on learner-induced trajectories is sufficient to produce a competitive learned controller in the current benchmark.

### H3

The resulting learned controller can remain competitive or stronger under stress perturbations involving stochasticity, burden, delay, tissue fragility, and movement bottlenecks.

## 10. Hypotheses The Current Paper Does Not Support

The manuscript should not claim any of the following.

- that the learned agent discovered a qualitatively new immune strategy
- that the benchmark is a validated biological model
- that delayed-support neutrophil dynamics already constitute a deep coordination problem
- that the current teacher-student formulation solves privileged-control distillation
