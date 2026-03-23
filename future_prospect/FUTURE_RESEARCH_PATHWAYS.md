# Future Research Pathways

## 1. Purpose of This Document

This document records the strongest credible future directions for the project after the current benchmark/control paper is frozen.

It is not a wish list. Each pathway is evaluated on:

- scientific upside
- publication upside
- technical realism
- risk of failure
- whether it should be a separate paper

## 2. Current Baseline From Which Future Work Starts

The current paper leaves the project in a good but bounded state:

- the environment is fixed and documented
- the heuristic is a strong teacher and baseline
- `BC-only` fails
- hierarchical `BC + DAgger` is competitive
- `BC + DAgger + RL` is a negative ablation
- robustness evidence is complete

This means future work should not ask "is there any viable learned path at all?"

That question is already answered.

The new question is:

- what next step raises the scientific ceiling enough to justify a second phase?

## 3. Summary Ranking

### Highest scientific upside

- privileged-teacher / inferability-aware distillation branch

### Highest probability of eventually producing a second publishable paper

- privileged-teacher branch, if treated as a separate method paper

### Best benchmark-only extension

- benchmark-family expansion with train/test/OOD splits

### Best domain-rich extension

- stronger delayed-support / support-allocation mechanics

### Weakest direction relative to effort

- longer blind RL training on the current setup

## 4. Pathway A: Privileged Teacher and Student Transfer

### Short description

Build a stronger `Teacher v2`, potentially with partial, selected, or full current-state privileged visibility, then train:

- `Student B` from `Teacher v2`
- `Student A -> B` by continuing from the current frozen `BC + DAgger` controller

### Why this is promising

This is the only direction that clearly raises the method ceiling beyond the current paper.

It turns the research question from:

- can a learned controller become competitive in this benchmark?

into:

- how should privileged control be distilled into a partially observed deployable student?

That is a real ML-method question.

### What makes this branch publishable

This branch is publishable only if the core contribution is not "better heuristic."

The real contribution would need to be one of:

- privileged-to-partial distillation
- inferability-aware supervision
- belief-state distillation
- teacher-improvement plus student-transfer study

### Main technical warning

If `Teacher v2` uses information the student can never reconstruct from observation history, then pure action cloning becomes a noisy or contradictory target.

So this branch should not be designed as:

- stronger teacher -> clone actions

It should be designed as:

- stronger teacher -> distill inferable structure and decision intent

### Minimum viable version

1. Build `Teacher v2` with current-state privilege only.
2. Benchmark `Teacher v2` against the current heuristic.
3. If `Teacher v2` is not clearly better, stop.
4. If `Teacher v2` is better, train `Student B`.
5. Fine-tune `Student A -> B`.
6. Compare:
   - Heuristic v1
   - Teacher v2
   - Student A
   - Student B
   - Student A -> B

### What should be distilled

Candidate teacher targets:

- macro mode
- escalation-needed score
- hidden local burden score
- target compartment
- soft action targets rather than only hard actions

### Publication upside

- highest upside in the current project
- strongest chance of becoming a more ML-centric paper

### Failure modes

- teacher is better only because of information the student cannot infer
- student target becomes contradictory
- project turns into uncontrolled scope drift

### Recommendation

- strongest future pathway
- should be pursued as a separate paper branch
- should not be mixed into the current manuscript incrementally

## 5. Pathway B: Benchmark-Family Expansion

### Short description

Generalize the current environment into a benchmark family rather than a single curated setup.

Possible axes:

- compartment layouts
- bottleneck patterns
- hotspot count and location
- bacterial burden
- stochasticity level
- support delay and jitter
- sensing radius or observation corruption

### Why this matters

The current benchmark is coherent but narrow.

A benchmark-family extension would answer the criticism:

- this is one custom simulator with one frozen map family

by turning it into:

- a controlled family of related partially observable tasks

### What makes this publishable

This becomes publishable if the emphasis is on:

- train/test/OOD splits
- reproducible evaluation protocols
- policy ranking stability under shift

### Main technical warning

If the benchmark family is expanded without a clean design, it becomes a random parameter zoo.

The benchmark must remain:

- structured
- interpretable
- reproducible

### Minimum viable version

1. Define a small parameterized family.
2. Freeze train-distribution seeds and test-distribution seeds.
3. Add one OOD split.
4. Reevaluate heuristic and DAgger.
5. Check whether conclusions survive.

### Publication upside

- moderate to high
- stronger for benchmark venues than for method venues

### Failure modes

- too much environment variation without enough design logic
- more engineering than science
- unclear new claim

### Recommendation

- good second-best direction
- especially useful if the goal shifts toward benchmark prestige

## 6. Pathway C: Stronger Support Mechanics

### Short description

Increase the sophistication of neutrophil/support dynamics so the escalation decision creates richer control tension.

Examples:

- support capacity limits
- region-specific or compartment-specific support
- support overshoot and collateral damage sensitivity
- support saturation or delayed failure
- competing hotspots that force triage

### Why this matters

One honest limitation of the current paper is that the neutrophil story is a delayed-support mechanic, not a deep coordination problem.

This branch directly addresses that limitation.

### What makes this publishable

This becomes interesting if the new mechanics create:

- real triage decisions
- nontrivial coordination trade-offs
- clear failure cases for naive heuristics

### Main technical warning

It is easy to make the environment more complicated without making it more scientific.

The added mechanics should change the control structure, not just add decorative biology.

### Minimum viable version

1. Add one new source of support tension.
2. Reevaluate heuristic and DAgger.
3. Check whether the new mechanic changes optimal policy structure, not just scores.

### Publication upside

- moderate
- stronger as an environment paper than as an ML-method paper

### Failure modes

- extra complexity with weak explanatory value
- broken comparability with the current benchmark

### Recommendation

- promising, but lower priority than Pathway A

## 7. Pathway D: Biological Realism Expansion

### Short description

Move the simulator toward richer biological detail.

Possible directions:

- richer cytokine signaling
- more immune cell types
- adaptive immunity
- more detailed tissue microanatomy

### Why this is tempting

It can make the environment feel less toy-like.

### Why this is dangerous

More biology does not automatically make the work better.

Without validation, it can produce:

- more parameters
- more implementation burden
- weaker interpretability
- no stronger ML claim

### Publication upside

- unclear unless paired with domain validation

### Failure modes

- large complexity increase
- no strong evaluation story
- weaker rather than stronger paper

### Recommendation

- low priority for the next phase
- do not pursue unless there is a separate validation plan

## 8. Pathway E: More Training on the Current Stack

### Short description

Continue longer training for:

- `BC + DAgger`
- `BC + DAgger + RL`
- more seeds

### Why this looks attractive

It is the easiest path to start.

### Why it is weak

The current evidence already suggests:

- `BC-only` is structurally insufficient
- `BC + DAgger` is the correct mainline
- the current RL regime is not reliable

So longer blind training is unlikely to change the scientific story much.

### Publication upside

- low

### Recommendation

- not a good next branch

## 9. Recommended Order of Future Work

If a second research phase is started, the recommended order is:

1. Pathway A: privileged teacher and student transfer
2. Pathway B: benchmark-family expansion
3. Pathway C: stronger support mechanics
4. Pathway D: biological realism expansion
5. Pathway E: more training on the current stack

## 10. Go / No-Go Criteria

### For Pathway A

Proceed only if:

- `Teacher v2` clearly beats the current heuristic
- the teacher logic can be stated clearly
- at least some teacher targets are plausibly inferable from student history

Stop if:

- the better teacher relies mainly on unrecoverable hidden information
- the student objective becomes mostly contradictory

### For Pathway B

Proceed only if:

- the benchmark family can be made structured and reproducible
- train/test/OOD splits are clearly defined

Stop if:

- it becomes a parameter sweep without a clean hypothesis

### For Pathway C

Proceed only if:

- the new support mechanics create genuine decision tension

Stop if:

- the changes add complexity without changing policy structure

## 11. Recommended Strategic Choice

The best overall strategy is:

- freeze and submit the current paper
- then start Pathway A in a separate branch

This keeps the current result safe while opening the strongest higher-upside research direction.

That is the highest-quality portfolio strategy available from the current state of the project.
