# Detailed Game Rules and Interaction Logic

## 1. Executive Summary

This benchmark is a deliberately simplified but mechanistically structured innate-immune interaction model.

It is not a full biological simulator of lung physiology. It is a focused research sandbox designed to answer:
1. How local spatial structure changes host-pathogen dynamics.
2. How delayed immune reinforcement shifts outcomes.
3. How clearance-vs-collateral-damage trade-offs emerge.

So the short answer to your question "is this too simple biologically?" is:
1. It is biologically abstract, yes.
2. It is not biologically meaningless.
3. The simplification is intentional so more budget can go into interaction dynamics and decision policies.

## 2. Why This Level of Simplicity Is Reasonable

The model keeps only the components needed for the target research question:
1. Local compartments and bottlenecks (spatial constraints).
2. Nutrient ecology (resource-limited growth and movement choices).
3. Bacterial phenotypic progression (planktonic -> attached -> microcolony -> biofilm).
4. Host local control plus delayed reinforcement (macrophage + neutrophil queue).
5. Chemokine field (signal-driven escalation).
6. Damage economics and utility (clearance is not free).

What is intentionally omitted:
1. Full cytokine network.
2. Adaptive immunity.
3. Detailed receptor pathways.
4. Continuous fluid mechanics.
5. Tissue microanatomy beyond a compartment graph.

This is an interaction-first model: lower biological dimensionality, higher clarity of cause-effect between policies and outcomes.

## 3. The Actual Story of One Episode

Think of each run as this narrative:
1. A local bacterial hotspot is seeded in one compartment.
2. A resident macrophage patrols and either attacks or signals.
3. Bacteria try to survive by moving, attaching, and building protected states.
4. Local contact/signaling raises chemokine concentration.
5. If chemokine pressure becomes strong enough, a delayed neutrophil wave is queued.
6. Neutrophils enter from boundaries and increase kill pressure.
7. Strong response may clear infection but can increase tissue damage.
8. Final outcome reflects both control success and biological cost.

This is the core interaction story encoded by the environment step loop.

## 4. World Geometry and Accessibility

Source of implementation: simulator/environment.py and config.py.

### 4.1 Grid and compartments

With default values:
1. WORLD_ROWS = 2
2. WORLD_COLS = 2
3. COMPARTMENT_SIZE = 10
4. GRID_SIZE = 21

So the world is 21x21 with 4 compartments separated by wall lanes.

### 4.2 Walls and bottlenecks

Black wall lanes are non-walkable barriers except carved doorway spans.

Role in interaction:
1. Prevents unlimited open-field chase/evasion.
2. Creates chokepoints where signaling and interception matter more.
3. Forces strategic timing when moving between compartments.
4. Enables local persistence pockets for bacteria.

### 4.3 Inter-compartment entrance correctness

Doorways are now truly enterable, not only visually open.

The implementation guarantees this by:
1. Removing doorway cells from blocked_tiles.
2. Marking doorway cells as walkable in compartment_map.
3. Evaluating walkability through both checks.
4. Centering doorway spans with odd width internally, even if configured width is even.

That last point avoids directional bias that can isolate a quadrant at wall intersections.

## 5. Agents and Their Algorithms

## 5.1 Macrophage (host decision agent)

Current repo baseline policy: heuristic local controller (agents/heuristic.py).

Current research mainline: learned recurrent controller trained from partial observations, sequence-aware behavior cloning, and later fine-tuning. The learned controller uses an abstract `request_help` decision rather than directly selecting signal intensity.

Heuristic priorities:
1. Attack if bacteria adjacent.
2. Signal if local burden high and signal available.
3. Move toward nearest sensed bacteria.
4. If none seen, move toward chemokine peak.
5. Else random legal move.

Learned-controller design goals:
1. Observe only local state and local chemical cues.
2. Maintain memory across steps.
3. Decide when to move, attack, or request help.
4. Stay fast enough for live GUI use, unlike online tree search.

## 5.2 Bacteria (adaptive local policy)

Policy: rule-based local adaptation with stochastic branch (agents/bacteria_adaptive.py).

Priorities:
1. Random move with probability BACTERIA_STOCHASTICITY.
2. Stay if fortified and pressure is low.
3. Flee nearby macrophage if planktonic and threatened.
4. Move to richer nutrient neighbor if local nutrient is poor.
5. Move down chemokine gradient when pressure rises.
6. Otherwise cluster with nearby bacteria to support colony formation.

## 5.3 Neutrophils

Not present at start; recruited by chemokine threshold events.

Behavior:
1. Move toward nearest local bacterium within sense radius.
2. Otherwise random walk.
3. Attack adjacent bacteria.
4. Die at lifespan limit.

## 6. How Interaction Happens Per Step

Source: Environment.step.

Order:
1. Select macrophage action.
2. Execute macrophage action.
3. Execute bacteria phase.
4. Execute neutrophil phase.
5. Update nutrients, chemokine, recruitment queue, damage.
6. Increment time and update phase label.
7. Record history.
8. Check terminal conditions.

This order matters. Example: if macrophage signals now, chemokine can influence recruitment pressure soon after, which changes future neutrophil arrival and bacterial decisions.

## 7. Chemokine Pressure: Meaning and Role

Chemokine pressure is the local immune-signaling burden represented by the chemokine field value at each tile.

Operational meaning in this model:
1. High chemokine means the local region is immunologically "hot".
2. High chemokine increases recruitment likelihood (via threshold crossing).
3. High chemokine reduces bacterial replication probability.
4. High chemokine encourages dispersal from attached/biofilm trajectories.

Field update dynamics:
1. Production: macrophage attack/signal and contact-associated contributions.
2. Spread: diffusion to walkable neighbors.
3. Loss: decay each step.
4. Perturbation: optional noise.

This is called pressure because it makes bacterial persistence harder while pushing the system toward stronger immune intervention.

## 8. Phase Model: What Phases Mean and How They Are Chosen

Current phase labels are produced by direct rules in _update_phase_label, not by a learned classifier.

Selection logic:
1. escalation: if neutrophils > 0 and bacteria > 0
2. recruitment: else if recruitment queue is non-empty
3. detection_containment: else if chemokine max > 1.0
4. seeding: otherwise

Interpretation:
1. seeding: initial local establishment, low signal.
2. detection_containment: macrophage has detected/engaged enough to raise signal.
3. recruitment: threshold crossed recently and delayed reinforcements are pending.
4. escalation: neutrophils and bacteria coexist, high-pressure conflict with damage risk.

Important note:
Phase labels are descriptive runtime tags for analysis and plotting. They are not hard constraints that force actions.

## 9. Nutrient Field: Why It Matters

Nutrients are not decorative.

They regulate:
1. Whether attachment progression is sustained.
2. Whether replication is allowed.
3. Where bacteria prefer to move.
4. How fast local burden can recover after attacks.

Dynamics:
1. Regeneration: each active tissue cell recovers toward PATCH_NUTRIENT_CAPACITY.
2. Consumption: bacteria drain local nutrient and may gain health if enough consumed.

So nutrient heterogeneity directly shapes survival strategy, replication windows, and phase timing.

## 10. Bacterial State Machine and Replication Gate

State chain:
1. planktonic -> attached
2. attached -> microcolony
3. microcolony -> biofilm
4. biofilm -> planktonic (under high pressure + cooldown conditions)

Replication requires all gates passing:
1. Attachment gate if REPLICATION_REQUIRES_ATTACHMENT is true.
2. Nutrient gate at tile.
3. Local crowding gate (carrying capacity).
4. Probabilistic gate reduced by chemokine pressure.

This prevents trivial runaway growth from pure movement luck.

## 11. Tissue Damage and Utility

Host utility equation:

HostUtility = killed_bacteria - alpha * tissue_damage - beta * immune_usage

Where:
1. alpha = HOST_UTILITY_ALPHA
2. beta = HOST_UTILITY_BETA

Damage sources:
1. Neutrophil collateral effect.
2. Ongoing bacterial burden.
3. Extra inflammation penalty when both coexist.

This makes aggressive response a strategic trade-off, not always optimal.

## 12. Visual Semantics in the Current GUI

This section explains exactly what you asked about: black walls, green blocks, pale white dots, slight red color, and white-like regions.

### 12.1 Dark black/charcoal lines

Meaning:
1. Structural wall cells between compartments.
2. These are non-walkable except at carved doorway spans.

Role:
1. Constrains paths.
2. Creates bottlenecks.
3. Makes containment/local escape patterns non-trivial.

### 12.2 Green blocks (different intensity)

Meaning:
1. Tissue nutrient level.
2. Darker green = richer nutrient.
3. Paler green/near-white = poorer nutrient.

Role:
1. Controls bacterial replication viability.
2. Influences bacterial movement strategy.

### 12.3 Pale white dots

Meaning:
1. Surface patches (attachment-friendly micro-sites).

Role:
1. Enable and stabilize transition from planktonic toward attached states when nutrients are sufficient.
2. Without these, biofilm trajectory opportunities reduce sharply.

### 12.4 Slight red translucent color (wash/overlay)

Meaning:
1. Chemokine concentration overlay.
2. Stronger red wash = higher local chemokine pressure.

Role:
1. Visual indicator of immune signaling hot zones.
2. Predicts likely recruitment and local replication suppression.

### 12.5 White or almost-white cells

Meaning in current UI:
1. Usually low-nutrient tissue cells (base nutrient color is very light).
2. In old minimal UI versions, white-like cells could also mean generic empty/background tile.

Role:
1. Biologically in this model: resource-poor zones where bacteria are less likely to replicate effectively.
2. Visually: contrast anchor to make gradients and overlays interpretable.

## 13. Terminal Conditions

Run ends when one condition is reached:
1. Macrophage health <= 0.
2. Tissue damage >= fail threshold.
3. Bacteria count == 0.
4. Time horizon reached with bacterial persistence.

## 14. Key Logs and What They Mean

Recorded per step:
1. phase
2. macrophage health
3. bacterial count
4. neutrophil count
5. tissue damage
6. host utility
7. colonized compartments
8. chemokine peak
9. nutrient total

Why these matter:
1. They separate "won" from "won safely".
2. They expose whether control required harmful escalation.
3. They let you compare policy behavior, not just final labels.

## 15. Practical Interpretation: Is This Good Enough for Research Use?

For exploratory algorithmic interaction studies: yes.

For high-fidelity translational biology claims: no, not by itself.

Appropriate claim level right now:
1. This is a controlled mechanistic interaction model inspired by biological principles.
2. It is suitable for studying policy behavior, trade-offs, and phase transitions.
3. It is not a substitute for multi-scale validated physiological simulation.

That is the correct scientific positioning of the current system.

## 16. Minimal Invariant Checklist

When modifying code, verify:
1. Adjacent compartments have at least one traversable doorway path.
2. Replication gates still enforce attachment/nutrient/crowding logic.
3. Recruitment still requires threshold plus delay.
4. Damage rises under sustained neutrophil activity.
5. Utility can penalize over-escalation despite clearance.
6. Phase labels still follow runtime logic and remain interpretable.
