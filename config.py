""" 
Global configuration for the macrophage vs Pseudomonas simulation.
All tunable parameters and hyperparameters for the simulation are defined here.
"""
GRID_SIZE = 30  # Grid dimension (GRID_SIZE x GRID_SIZE)
TIME_HORIZON = 300  # Maximum number of steps in a simulation

# Macrophage parameters
MACROPHAGE_HEALTH = 100  # Initial health of macrophage
MACROPHAGE_VISION_RADIUS = 999  # Vision radius of macrophage (999 = entire grid visibility)
ROLLOUT_BUDGET = 0  # MCTS rollout budget (0 = use heuristic only for debugging)
MACROPHAGE_TOXIN_RADIUS = 1  # Radius of toxin burst (Manhattan distance)
MACROPHAGE_TOXIN_DAMAGE = 50  # Damage dealt by toxin burst to bacteria in range

# Bacteria parameters
INITIAL_BACTERIA_COUNT = 5  # Initial number of bacteria
BACTERIA_MAX_HEALTH = 20  # Health of a fully mature bacteria
BACTERIA_INITIAL_HEALTH_WEAK = 10  # Health of a newly replicated bacteria
BACTERIA_HEALTH_REGEN = 1  # Health regained per step (for newly spawned bacteria to mature)
BACTERIA_MATURITY_AGE = 5  # Steps needed for bacteria to become fully mature
BACTERIA_ATTACK_DAMAGE = 10  # Damage to macrophage per successful attack
BACTERIA_ATTACK_PROB = 0.3  # Probability of a successful attack when adjacent
BACTERIA_REPLICATION_PROB = 0.02  # Base probability of replication per step (if bacteria is healthy)
BACTERIA_REPLICATION_PROB_BOOST = 0.1  # Additional replication probability if the bacteria just consumed a nutrient
BACTERIA_MAX_COUNT = 40  # If bacteria population reaches this, they win

# Quorum sensing and biofilm (Pseudomonas aeruginosa characteristics)
QUORUM_SENSING_ENABLED = True  # Enable bacterial communication
QUORUM_THRESHOLD = 3  # Number of nearby bacteria to trigger quorum sensing
QUORUM_RADIUS = 3  # Radius to check for quorum sensing
BIOFILM_DEFENSE_BONUS = 5  # Damage reduction when in biofilm
BIOFILM_ATTACK_BONUS = 5  # Extra damage when attacking from biofilm

# Bacterial virulence factors
VIRULENCE_ENABLED = True  # Enable bacterial toxin production
VIRULENCE_DAMAGE_PER_STEP = 2  # Passive damage to macrophage from nearby virulent bacteria
VIRULENCE_RADIUS = 2  # Range of bacterial toxin effect
VIRULENCE_ACTIVATION_PROB = 0.15  # Probability mature bacteria activates virulence per step

# Macrophage exhaustion
MACROPHAGE_EXHAUSTION_ENABLED = True  # Macrophage gets tired from fighting
MACROPHAGE_EXHAUSTION_PER_KILL = 3  # Exhaustion gained per bacteria killed
MACROPHAGE_EXHAUSTION_RECOVERY = 1  # Exhaustion recovered per step when not fighting
MACROPHAGE_EXHAUSTION_DAMAGE_THRESHOLD = 20  # Above this exhaustion, macrophage takes extra damage
MACROPHAGE_TOXIN_COOLDOWN = 3  # Steps needed before toxin can be used again

# Nutrient parameters
NUTRIENTS_ENABLED = True  # Whether nutrients are placed on the grid
N_INITIAL_NUTRIENTS = 10  # Number of nutrient cells initially
NUTRIENT_HEAL_AMOUNT = 10  # Health boost for bacteria when stepping on a nutrient

# Performance / simulation parameters
MCTS_SIM_DEPTH = 20  # Depth of simulation in each MCTS rollout (how many steps to simulate ahead)
SAVE_PLOTS = True  # Whether to save performance plots (bacteria count vs time, etc.)

# Bacteria behavior modes
BACTERIA_MODE = "scatter"  # Default behavior mode: "cluster", "scatter", "replicate", "defend", or "adaptive"
BACTERIA_ADAPTIVE_ENABLED = True  # Use adaptive AI instead of fixed behavior modes
BACTERIA_STOCHASTICITY = 0.15  # Amount of randomness in bacteria decisions (0.0 = deterministic, 1.0 = random)
