# Partial-Observation Immune Control with Delayed Support: A Spatial Benchmark and Hierarchical Recurrent Baseline

## Abstract

Immune control in tissue is a sequential decision problem with incomplete local sensing, delayed reinforcement, and a nontrivial trade-off between clearance and collateral damage. This project studies that problem through a spatial benchmark in which a macrophage acts under partial observability against adaptive bacterial dynamics while escalation through neutrophil recruitment is delayed and biologically costly. The environment combines compartmental bottlenecks, nutrient-limited bacterial progression, chemokine-mediated support recruitment, and explicit host-utility accounting.

We compare a strong interpretable heuristic baseline against three learned variants: sequence-aware behavior cloning, hierarchical recurrent `BC + DAgger`, and `BC + DAgger + RL` fine-tuning. On a held-out `100`-seed benchmark, plain behavior cloning collapses under covariate shift (`0.02` win rate), while hierarchical `BC + DAgger` becomes competitive with the heuristic (`0.60` vs `0.58` win rate) and improves tissue damage and host utility. Additional RL fine-tuning degrades the stronger imitation-based controller (`0.34` win rate), showing that unconstrained post-imitation optimization is not automatically beneficial in this setting.

We further evaluate the frozen hierarchical `BC + DAgger` controller under robustness stress tests spanning higher bacterial stochasticity, hotspot overload, recruitment delay, fragile tissue, and narrow movement bottlenecks. The learned controller remains competitive or stronger across all reported suites, with the clearest gains under stronger stochasticity, heavier hotspot burden, and delayed support. The resulting contribution is a reproducible benchmark/control study: a partially observable immune-control task, an interpretable strong baseline, and the first learned controller in this setting that is competitive under held-out and robustness evaluation.

## 1. Introduction

Local innate immune control is naturally sequential and partially observed. A macrophage does not receive a privileged global map of bacterial burden, support status, or future pathogen motion. Instead it acts from local contact, local chemokine, and its own recent experience. At the same time, escalation is delayed and costly: once signaling begins, reinforcement through neutrophil recruitment is not immediate, and stronger immune activity can damage tissue even when it helps clear infection.

These properties make local host control a suitable sequential-decision problem for machine learning, but not all learning-based approaches are equally appropriate. In particular, a strong hand-crafted controller already captures much of the local contain-versus-escalate logic, so a learned agent must justify itself empirically rather than rhetorically. The relevant question is therefore not whether learning can be applied at all, but whether a learned controller can become competitive with a competent interpretable baseline under a fixed, reproducible benchmark.

This paper presents a benchmark/control study rather than a new generic learning algorithm. The benchmark centers a single macrophage in a compartmental spatial environment with adaptive bacteria, a chemokine field, delayed neutrophil support, and explicit host-utility accounting. The empirical question is addressed through an ablation ladder: sequence-aware behavior cloning, `BC + DAgger`, and `BC + DAgger + RL`. This structure is important because it separates three distinct possibilities: that simple imitation is enough, that learner-state relabeling is necessary, or that post-imitation RL reliably improves the policy.

The evidence supports a narrow but useful conclusion. Plain behavior cloning is not viable in this benchmark. Hierarchical recurrent `BC + DAgger` is the first learned controller in the project that becomes competitive with the heuristic under held-out evaluation. Lightweight RL fine-tuning does not improve that controller and instead degrades it. The frozen `BC + DAgger` policy also remains competitive or stronger under multiple robustness stress tests. The contribution is therefore a reproducible partially observable immune-control benchmark together with a strong heuristic baseline, a successful learned baseline, and informative negative ablations.

## 2. Problem Formulation

### 2.1 Task as a POMDP

The benchmark is formalized as a partially observable Markov decision process

`M = (S, A, O, T, Omega, R, gamma)`,

where `S` contains the full simulator state, `A` is the macrophage action space, `O` is the observation space exposed to the learner, `T` is the transition kernel, `Omega` is the observation map, `R` is the training reward used in optional RL fine-tuning, and `gamma` is the discount factor used by that RL stage.

The latent state includes macrophage position and health, bacterial positions and phenotype states, neutrophil positions and remaining lifespan, the nutrient and chemokine fields, the recruitment queue, and accumulated tissue damage. The learner does not observe this state directly.

### 2.2 Partial Observation

The learned controller receives a local centered observation built in `simulator/rl_interface.py`. The observation contains:

- a local grid tensor with blocked cells, nutrient intensity, chemokine intensity, visible bacteria, visible neutrophils, and macrophage location
- scalar features summarizing local counts, health, and directional cues
- optional belief-style memory features, including last seen bacteria compartment, strongest recent chemokine compartment, and visited compartments

The policy therefore acts on histories `h_t = (o_0, a_0, ..., o_t)` rather than on the full state. This is why recurrence is structurally justified in the benchmark design.

### 2.3 Action Abstraction

At the environment interface, the cleaned learned-controller action set is:

`A = {move_up, move_down, move_left, move_right, stay, attack, request_help}`.

Internally, the hierarchical policy factorizes control into:

- macro mode `m_t` in `{engage, request_help, follow_chem, patrol}`
- micro action `u_t` in `{move_up, move_down, move_left, move_right, stay, attack}`

The executed action is produced by an environment-side resolution function. In particular, `request_help` is an abstract escalation decision that is mapped to the strongest currently valid concrete signal action. This abstraction reduces action branching while preserving the key decision of whether to escalate.

### 2.4 Objective and Evaluation

The benchmark explicitly tracks host utility as a clearance-versus-cost trade-off:

`U_t = K_t - alpha * D_t - beta * I_t`,

where `K_t` denotes cumulative clearance progress, `D_t` is tissue damage, and `I_t` is immune usage cost. Final evaluation reports:

- `win_rate`
- `tissue_damage`
- `host_utility`
- `episode_length`
- `recruited_neutrophils`

Behavior diagnostics are also logged, including attack frequency, signaling frequency, blind signals, stay actions, coverage ratio, and visible-bacteria ratio. These metrics are used to interpret policy style rather than replace the outcome measures.

## 3. Environment

The environment is intentionally biologically abstract. It is not a lung simulator and should not be presented as one. Its purpose is to isolate a structured local control problem with enough mechanistic detail to make policy choices meaningful.

The world is spatial and compartmental. Bottlenecks constrain movement between compartments and create local persistence pockets for bacteria. The nutrient field makes bacterial growth resource dependent. Bacteria follow a phenotypic progression from planktonic to more protected states such as attached, microcolony, and biofilm-like behavior. Chemokine pressure is produced by immune activity and influences both reinforcement and bacterial behavior. Neutrophil support is not available at the start of an episode; it is recruited only after sufficient signaling pressure and arrives with a delay through a queue.

This structure makes the benchmark different from systemic treatment-optimization settings. The macrophage is not selecting global therapies from a privileged state representation. It is making local spatial decisions with delayed consequences and uncertain downstream support.

Figure 1 should be the benchmark control-loop diagram in `paper_material/figures/benchmark_control_loop.svg`.

## 4. Baselines and Learned Controller

### 4.1 Heuristic Baseline

The heuristic is deliberately strong rather than decorative. It uses local sensing, short-term memory, and explicit contain-versus-recruit logic. It attacks when bacteria are adjacent, signals under sufficiently high local pressure, moves toward visible bacteria, follows local chemokine cues when direct targets are absent, and otherwise patrols.

This matters because the learned controller is only interesting if it competes with a competent baseline. The current paper should emphasize that the heuristic is a teacher and a serious benchmark policy, not a strawman.

### 4.2 Hierarchical Recurrent Policy

The learned controller uses a recurrent hierarchical architecture. A macro controller selects between engagement, escalation, chemokine-following, and patrol, while a micro controller selects a local movement or attack action conditioned on that mode. Auxiliary heads predict burden-related signals used during training. The architecture is designed for partial observability and sequential control, not for one-step classification.

### 4.3 Training Variants

Three learned variants are compared:

1. `BC-only`
2. `BC + DAgger`
3. `BC + DAgger + RL`

This ladder is the core experimental logic of the paper.

Plain behavior cloning tests whether supervised imitation on teacher trajectories is sufficient. It is not. This is consistent with the imitation-learning literature: Ross and Bagnell show that i.i.d. supervised imitation suffers from compounding error when the learner's actions affect future inputs, which is exactly the setting here ([Ross and Bagnell, 2010](https://proceedings.mlr.press/v9/ross10a.html)).

DAgger-style aggregation addresses that problem by collecting teacher labels on learner-induced trajectories. In this benchmark, that change is decisive: it moves the learned policy from collapse to competitiveness.

The RL fine-tuning stage is treated only as an ablation. In the current setup it is not a reliable improvement operator.

## 5. Experimental Protocol

The empirical protocol is fixed and reproducible.

- held-out `50`-seed evaluation is used for an earlier mainline snapshot
- held-out `100`-seed evaluation is the final comparison benchmark
- training-seed reproducibility is assessed on seeds `42`, `43`, and `44`
- robustness is evaluated through fixed override suites

All comparison methods are evaluated on the same held-out seed stream. Summary tables report arithmetic means across episodes. Metric definitions are centralized in `paper_material/METRIC_DEFINITIONS.md`.

The robustness suites test:

- baseline reference
- high bacterial stochasticity
- hotspot overload
- recruitment delay stress
- fragile tissue
- narrow bottleneck

This suite is not meant to claim universal robustness. It is meant to test whether the frozen learned controller survives a controlled set of meaningful task perturbations.

## 6. Main Results

### 6.1 Four-Row Held-Out Comparison

Table 1 is the central comparison for the paper.

| Method | Win Rate | Tissue Damage | Host Utility | Episode Length | Recruited Neutrophils |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic | 0.58 | 94.1710 | -90.3152 | 162.76 | 2.22 |
| Hierarchical BC-only | 0.02 | 133.9490 | -133.4290 | 194.49 | 0.00 |
| Hierarchical BC+DAgger | 0.60 | 90.6470 | -86.72115 | 137.07 | 4.44 |
| Hierarchical BC+DAgger+RL | 0.34 | 100.9305 | -97.91645 | 160.57 | 2.69 |

Three conclusions follow directly.

First, `BC-only` is not viable. It collapses under covariate shift, which is exactly what should be expected when a sequential partially observed policy is trained only on teacher-distribution trajectories.

Second, `BC + DAgger` is the only learned method that becomes competitive with the heuristic. It slightly exceeds the heuristic on win rate while also reducing tissue damage and improving host utility in the frozen best run. This makes it the correct learned mainline for the current paper.

Third, RL fine-tuning does not improve the stronger imitation-based policy. Instead it degrades the DAgger controller substantially. This is an informative negative result: unconstrained post-imitation optimization is not automatically beneficial in this benchmark.

Figure 2 should be `results/paper_ready/benchmark_comparison.png`.

### 6.2 Behavioral Differences

The learned controller is not identical to the heuristic in behavior. On the held-out `100`-seed benchmark, the frozen `BC + DAgger` policy uses fewer attack actions and fewer signal actions than the heuristic while recruiting more neutrophils overall and ending episodes faster. A restrained interpretation is that the learned controller reaches a different escalation pattern rather than discovering a qualitatively novel immune doctrine.

This distinction is important. The correct claim is not that the policy invented a surprising biological strategy. The correct claim is that it learned a competitive control style with a different timing and support-usage profile under the same information regime.

Figure 4 should be `results/paper_ready/behavior_tradeoffs.png`.

## 7. Reproducibility and Robustness

### 7.1 Reproducibility

Training-seed variation is real. Across train seeds `42`, `43`, and `44`, the held-out DAgger win rates are `0.60`, `0.50`, and `0.54`. The best run beats the heuristic, while the average learned behavior remains competitive and often more tissue-preserving. This supports a competitive-method claim but does not support a domination claim.

That distinction should be explicit in the manuscript. The strongest correct statement is:

- hierarchical `BC + DAgger` is competitive and often more tissue-preserving

The paper should avoid saying:

- the learned controller always outperforms the heuristic

### 7.2 Robustness

The corrected robustness suite strengthens the mainline claim. The frozen DAgger controller is competitive or stronger in all reported suites. The clearest gains appear under:

- `high_bacteria_stochasticity`: win rate `0.65` vs `0.54`
- `hotspot_overload`: win rate `0.55` vs `0.39`
- `recruitment_delay_stress`: win rate `0.57` vs `0.51`

The remaining suites continue to support competitiveness:

- `fragile_tissue`: `0.51` vs `0.50`
- `narrow_bottleneck`: `0.39` vs `0.29`

These results matter because they show that the mainline is not only a narrow in-distribution success on the base benchmark.

Figure 3 should be `results/paper_ready/robustness_summary.png`.

## 8. Relation to Prior Work

The current paper is adjacent to, but distinct from, previous immune-control RL work. Prior literature has studied intervention and immunomodulation in mechanistic inflammatory models, including simulation-based deep reinforcement learning for systemic inflammation control ([Cockrell et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9720328/)). Those works motivate the broader use of sequential decision methods in immune-inspired systems, but they do not target the same task formulation used here.

The gap addressed by the current paper is narrower:

- local spatial macrophage control
- partial observability by construction
- delayed support rather than immediate direct control
- benchmark-style evaluation with a strong heuristic baseline and negative ablations

This is why the manuscript should be positioned as a benchmark/control study rather than as a broad immunomodulation paper or a new imitation-learning method paper.

## 9. Limitations

The manuscript should state its limitations explicitly.

1. The environment is biologically abstract and is better viewed as a mechanistic control benchmark than as a validated physiological model.
2. The neutrophil mechanism is a delayed-support mechanism, not a rich multi-agent coordination model.
3. The teacher and student operate in the same information regime, which limits methodological novelty and keeps the learned controller strongly teacher-shaped.
4. The benchmark is currently one curated environment family rather than a broader benchmark generator.

These limitations do not invalidate the paper. They define its correct scope.

## 10. Conclusion

This project now supports a clean empirical conclusion. In a spatial partially observable immune-control benchmark with delayed support and adaptive pathogen dynamics, plain behavior cloning fails, hierarchical recurrent `BC + DAgger` becomes competitive with a strong heuristic, and lightweight RL fine-tuning degrades the stronger imitation-based controller. The frozen DAgger policy also remains competitive or stronger across the reported robustness suites.

That is enough for a defensible benchmark/control paper. It is not yet a privileged-distillation paper, a deep-coordination paper, or a biological-discovery paper. The value of the current work lies in its clarity: a fixed task, a strong interpretable baseline, an informative ablation ladder, and a reproducible learned mainline that survives both held-out evaluation and controlled robustness stress.

## References

1. Stephane Ross and Drew Bagnell. Efficient Reductions for Imitation Learning. AISTATS 2010. https://proceedings.mlr.press/v9/ross10a.html
2. Chase Cockrell, Dale Larie, and Gary An. Preparing for the next pandemic: Simulation-based deep reinforcement learning to discover and test multimodal control of systemic inflammation using repurposed immunomodulatory agents. Frontiers in Immunology, 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9720328/
