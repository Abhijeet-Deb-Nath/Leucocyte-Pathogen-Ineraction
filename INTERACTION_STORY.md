# Interaction Story

## 1. The Core Story

The simulator is built around a simple but strong control story:

- a local bacterial hotspot appears inside structured tissue
- a resident macrophage is the first responder
- the macrophage does not see the full infection state
- escalation is possible, but not free and not immediate
- bacteria are adaptive enough that passive containment can fail
- excessive immune response can also damage the host

That gives the benchmark a real decision tension:

- act aggressively now and risk collateral damage
- delay escalation and risk losing control of the infection

This is the core reason the environment is compelling. It is not just "immune system versus bacteria." It is a sequential decision problem with uncertainty, delay, and cost.

## 2. Why The Story Is Stronger Than A Simple Chase Game

If this were only a local chase-and-attack simulator, the problem would be flat:

- see bacteria
- move toward them
- attack until they are gone

The benchmark is more interesting because that simple loop is not enough.

The macrophage has to decide under partial information:

- whether the current local burden is small enough to handle alone
- whether chemokine escalation is necessary
- whether support will arrive in time if requested
- whether stronger immune activity is worth the damage cost

At the same time, bacteria are not static targets. They move stochastically, exploit nutrients, and progress toward more protected states. This makes delayed or weak control meaningfully worse.

## 3. How The Simulator Encodes This Story

### 3.1 Partial observability

The macrophage receives only a local observation rather than the full simulator state. That means the controller must act on:

- local contact
- local chemokine
- recent history

instead of on a global infection map.

This is what turns the task into a true sequential decision problem instead of a fully observed planning toy.

### 3.2 Delayed support

Neutrophils are not available on demand. Support is recruited only after chemokine pressure crosses a threshold, and then arrives through a delayed queue.

This creates the benchmark's main temporal tension:

- signaling too late means support arrives after burden has already grown
- signaling too early or too often can increase damage and immune usage

### 3.3 Adaptive pathogen pressure

Bacteria are not passive targets. They:

- move stochastically
- seek nutrients
- react to local immune pressure
- progress toward more protected states

This prevents the environment from collapsing into a deterministic pursuit problem.

### 3.4 Spatial bottlenecks

Compartments and doorways matter. They create:

- chokepoints
- local persistence pockets
- delayed traversal between regions

That means movement and escalation decisions are spatially meaningful, not just cosmetic.

### 3.5 Clearance versus damage

The benchmark does not treat all host wins as equal. It records tissue damage and host utility, so over-aggressive immune control can still be undesirable.

This is important because it prevents the problem from degenerating into:

- maximize killing at any cost

Instead, the controller must manage a trade-off:

- clear the infection
- while limiting collateral harm

## 4. Why This Is Compelling As A Game Or Simulation

The environment is compelling because every major mechanic contributes to the same tension rather than existing as decoration.

- spatial structure makes local control hard
- partial observability prevents trivial optimal behavior
- delayed support forces anticipation
- adaptive bacteria punish passivity
- damage accounting punishes overreaction

That unified structure is what gives the simulator a strong identity.

The story is therefore:

- a local defender with incomplete information
- facing a dynamic infection
- deciding when to contain and when to escalate
- under a cost for being wrong in either direction

That is a much stronger framing than simply saying the simulator models an immune-pathogen interaction.

## 5. Why This Is Scientifically Useful

The environment is biologically abstract, but it is not scientifically empty.

It is useful because it isolates a class of control problems that matter for ML:

- partial observation
- delayed consequences
- stochastic adversarial pressure
- cost-sensitive intervention

Those are the same ingredients that make many sequential decision problems difficult.

So the simulator is compelling in two ways at once:

- as a game-like control environment with real tension
- as a benchmark for studying memory, imitation, robustness, and escalation decisions

## 6. The Right One-Sentence Story

If the project needs to be summarized in one sentence, the strongest version is:

- a partially observable spatial immune-control benchmark where a macrophage must decide when to contain infection locally and when to trigger delayed, costly reinforcement against adaptive bacterial pressure

That is the central interaction story the simulator is designed to encode.
