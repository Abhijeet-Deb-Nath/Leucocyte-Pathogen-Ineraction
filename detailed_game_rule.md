# Detailed Game Rules and Interaction Logic: Pathway 1

## 1. What This "Game" Actually Is

This project is a mechanistic simulation of early innate immune response in a compartmentalized alveolar-like environment.

The simulation is framed as a game between:
1. Host side (macrophage as the controllable decision agent; neutrophils as delayed recruited effectors).
2. Bacteria side (adaptive local policy and state transitions from planktonic to biofilm-like protection).

The objective is not visual entertainment. The objective is to study the clearance-versus-damage trade-off:
1. Fast bacterial clearance is good.
2. Excess immune escalation causes host tissue damage.
3. Persistent bacteria can outlast the host even if immediate bacterial count is low.

## 2. Why There Are Many Green Nutrient Cells

Yes, this was an intentional modeling choice.

Green tissue cells represent local nutrient availability. A broad nutrient field is used to mimic heterogeneous but generally available substrate in alveolar tissue compartments, instead of making only a few single "food points."

This design creates richer interactions because:
1. Bacterial replication depends on local nutrient thresholds, not just movement luck.
2. Bacterial movement has meaningful choices (seek nutrients, avoid chemokine pressure, cluster with neighbors).
3. Macrophage decisions are not only chase-or-attack; signaling timing changes future immune pressure.
4. Chemokine dynamics and delayed neutrophil arrival create second-order effects across space and time.

In short: nutrient-rich regions are deliberate to create ecology-like dynamics, not random color fill.

## 3. World Topology and Compartments

Source of implementation: simulator/environment.py and config.py.

### 3.1 Grid construction

With current defaults:
1. WORLD_ROWS = 2
2. WORLD_COLS = 2
3. COMPARTMENT_SIZE = 10
4. GRID_SIZE = WORLD_ROWS * COMPARTMENT_SIZE + (WORLD_ROWS - 1) = 21

So the map is 21x21 with 4 tissue compartments separated by wall lanes.

### 3.2 Black lines in the UI

Black cells are structural walls and bottlenecks between compartments.

They come from blocked_tiles, which marks wall rows/columns between compartments.

### 3.3 Inter-compartment accessibility (your direct question)

After the latest fix, bottleneck doorways are genuinely traversable.

Technical detail:
1. Doorway cells are removed from blocked_tiles.
2. Doorway cells are also explicitly marked walkable in compartment_map (non-negative compartment id).
3. _is_walkable checks both conditions, so both must be valid.
4. If PASSAGE_BOTTLENECK_WIDTH is even, the implementation internally uses an odd centered span
	for doorway carving to avoid directional bias that can isolate one quadrant near wall intersections.

This means the inter-compartment pathways are now accessible/enterable in practice, not just visually open.

## 4. Entity Types and State Variables

Source of implementation: simulator/entities.py.

### 4.1 Macrophage

State:
1. position
2. health
3. signal_cooldown
4. kills

Actions:
1. Move (4-neighborhood plus stay)
2. Attack adjacent bacteria
3. Signal chemokine burst (if cooldown allows)

### 4.2 Bacteria

State:
1. position
2. health
3. age
4. state in {planktonic, attached, microcolony, biofilm}
5. attach_timer
6. biofilm_timer
7. dispersal_cooldown

### 4.3 Neutrophil

State:
1. position
2. health
3. age

Neutrophils are recruited later (not present at start).

## 5. Exact Turn Loop (Current Implementation)

Source of implementation: Environment.step in simulator/environment.py.

Per step order:
1. Host action chosen (MCTS by default, unless provided externally).
2. Macrophage action executes.
3. Bacteria phase executes (bacterial attacks, state updates, movement, replication).
4. Neutrophil phase executes (movement, attack, lifespan filtering).
5. Field updates execute (nutrient regen/consumption, chemokine diffusion/decay/noise, recruitment queue, tissue damage accumulation).
6. step_count increments.
7. phase label updates.
8. history logging updates.
9. termination checks run.

This ordering is important because signaling before bacterial/neutrophil phases changes the same-step chemokine dynamics.

## 6. Agent Algorithms

## 6.1 Host policy (Macrophage): MCTS rollout planner

Source: agents/mcts.py.

It is a lightweight Monte Carlo rollout selector, not full tree backpropagation.

Algorithm in practice:
1. Enumerate legal macrophage actions.
2. For each action, run cloned-environment rollouts.
3. First simulated move uses the candidate action.
4. Remaining rollout depth uses random legal macrophage actions.
5. Score terminal/non-terminal state with utility-based evaluator.
6. Pick action with highest average rollout score.

Controls:
1. ROLLOUT_BUDGET
2. MCTS_SIM_DEPTH
3. 1.5s safety timeout fallback to heuristic policy

Evaluator shape:
1. Large positive bonus for Host win.
2. Large negative bonus for Bacteria win.
3. Otherwise: host utility adjusted by bacterial burden, neutrophil burden, macrophage health.

## 6.2 Host fallback policy: Heuristic contain-vs-recruit policy

Source: agents/heuristic.py.

Decision logic:
1. If bacteria are adjacent and attack is legal: attack.
2. If local bacterial burden is high (>=3) and signaling legal: signal.
3. Else move toward nearest sensed bacterium.
4. If no bacteria sensed, move toward chemokine peak.
5. Else random legal move.

## 6.3 Bacteria policy: Adaptive local movement

Source: agents/bacteria_adaptive.py.

Decision hierarchy:
1. With probability BACTERIA_STOCHASTICITY, random move/stay.
2. If in microcolony/biofilm and local chemokine pressure is not high: stay.
3. If macrophage is close and bacterium is planktonic: move away.
4. If local nutrient is below replication threshold: move toward richer neighboring nutrient.
5. If local chemokine is high: move toward lower-chemokine neighbor.
6. Otherwise move toward local clustering (higher nearby friend count).

## 6.4 Neutrophil movement and attack

Source: simulator/environment.py.

Behavior:
1. If bacteria are within NEUTROPHIL_SENSE_RADIUS, move one step toward nearest target.
2. Otherwise random walk.
3. Attack adjacent bacteria with NEUTROPHIL_KILL_DAMAGE.
4. Die when age reaches NEUTROPHIL_LIFESPAN.

## 7. Bacterial State Machine and Replication

Source: simulator/environment.py + config.py.

### 7.1 State transitions

Planktonic -> Attached:
1. Must be on a surface patch.
2. Nutrient at tile must be >= REPLICATION_REQUIRES_MIN_PATCH_RESOURCE.
3. attach_timer must reach ATTACHMENT_TIME_REQUIRED.

Attached -> Microcolony:
1. biofilm_timer increments each step.
2. If chemokine pressure gets high and dispersal_cooldown allows, can revert to planktonic.
3. Else reaches microcolony after BIOFILM_BUILD_TIME.

Microcolony -> Biofilm:
1. biofilm_timer continues increasing.
2. Switch to biofilm when timer exceeds mature threshold.

Biofilm -> Planktonic dispersal:
1. If chemokine pressure becomes high enough and cooldown allows.

### 7.2 Replication gate

Replication requires all of:
1. If REPLICATION_REQUIRES_ATTACHMENT is true, planktonic cannot replicate.
2. Local nutrient >= REPLICATION_REQUIRES_MIN_PATCH_RESOURCE.
3. Local bacterial crowding < LOCAL_CARRYING_CAPACITY.
4. Random success under probability:

p = BASE_REPLICATION_PROB - (chemokine * IMMUNE_PRESSURE_REPLICATION_PENALTY)

Then scaled down in microcolony/biofilm by BIOFILM_REPLICATION_MODIFIER.

## 8. Chemokine and Recruitment Dynamics

Source: simulator/environment.py.

Per step field process:
1. Chemokine increases via macrophage attack/signal and contact-associated effects.
2. Diffusion redistributes concentration to walkable neighbors.
3. Decay applies global damping.
4. Optional Gaussian noise perturbs gradients.

Recruitment queue logic:
1. If chemokine peak >= NEUTROPHIL_RECRUITMENT_THRESHOLD, enqueue a future spawn event at current_step + NEUTROPHIL_ARRIVAL_DELAY.
2. When due, spawn neutrophils from boundary walkable cells (if pool cap not exceeded).
3. Each recruited neutrophil contributes to immune_usage.

## 9. Tissue Damage and Utility

Host utility used for planning/reporting:

HostUtility = killed_bacteria - alpha * tissue_damage - beta * immune_usage

Where:
1. alpha = HOST_UTILITY_ALPHA
2. beta = HOST_UTILITY_BETA

Tissue damage increases from:
1. Active neutrophil burden.
2. Existing bacterial burden.
3. Additional inflammation penalty when both are present.

## 10. End Conditions

Simulation terminates when first true:
1. Macrophage health <= 0 (Bacteria win).
2. Tissue damage >= TISSUE_DAMAGE_FAIL_THRESHOLD (Bacteria win).
3. No bacteria remain (Host win).
4. step_count >= TIME_HORIZON with persistence (Bacteria win).

## 11. What Is Logged Per Episode

History tracks:
1. step
2. phase
3. macrophage_health
4. bacteria_count
5. neutrophil_count
6. tissue_damage
7. host_utility
8. colonized_compartments
9. chemokine_peak
10. nutrient_total

Additional internal counters:
1. killed_bacteria
2. immune_usage
3. chemokine_events
4. macrophage_position_frequency
5. action_effect_frequency

## 12. Clarification of Visual Semantics in Current GUI

Current research-style GUI meanings:
1. Dark wall cells: compartment boundaries and bottlenecks.
2. Green intensity: local nutrient level.
3. Red circles: bacteria.
4. Blue circle: macrophage.
5. Cyan circles: neutrophils.
6. Red translucent overlay: chemokine concentration.
7. Small pale dots: surface patches that support attachment progression.

## 13. Answers to Your Specific Questions

Q1: "Was maximum nutrient coverage intentional to mimic biology and create richer interaction?"
A1: Yes. Nutrient-rich tissue distribution is intentional and supports non-trivial bacterial behavior, replication gating, and spatial trade-offs.

Q2: "What algorithm are agents using?"
A2:
1. Macrophage: rollout-based MCTS-style planner with utility evaluator and heuristic fallback.
2. Bacteria: adaptive local rule policy with stochastic branch.
3. Neutrophils: deterministic nearest-target pursuit within sensing radius, otherwise random walk.

Q3: "How does interaction really happen?"
A3: Through a strict step pipeline where host action changes chemokine pressure, bacterial state/movement/replication reacts to nutrients and pressure, then recruited neutrophils enter with delay and increase both kill pressure and collateral damage.

Q4: "Are inter-compartment pathways truly enterable by cells?"
A4: Yes after the doorway-walkability fix: doorway tiles are now both unblocked and walkable in compartment_map, so agents can traverse bottlenecks across compartments.

## 14. Minimal Validation Checklist for Future Changes

When changing environment dynamics, verify these invariants:
1. At least one traversable path exists between any adjacent compartments through doorway bottlenecks.
2. Planktonic bacteria cannot replicate when REPLICATION_REQUIRES_ATTACHMENT is true.
3. Recruitment only occurs after threshold crossing plus arrival delay.
4. Tissue damage rises under sustained neutrophil activity.
5. Host utility decreases if damage and immune overuse dominate clearance.

Keeping this checklist prevents silent regressions in the biological interaction logic.
