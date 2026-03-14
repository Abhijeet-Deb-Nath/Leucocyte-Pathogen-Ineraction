"""Pathway 1 configuration: Acute alveolar hotspot immune-pathogen simulation."""

# ---------------------------
# Core episode configuration
# ---------------------------
TIME_HORIZON = 220
SEED = None

# ---------------------------
# Compartment world geometry
# ---------------------------
WORLD_ROWS = 2
WORLD_COLS = 2
COMPARTMENT_SIZE = 10
PASSAGE_BOTTLENECK_WIDTH = 2
SURFACE_PATCH_DENSITY = 0.22

# Derived global grid size used by existing visualization tooling.
GRID_SIZE = WORLD_ROWS * COMPARTMENT_SIZE + (WORLD_ROWS - 1)
N_COMPARTMENTS = WORLD_ROWS * WORLD_COLS

# ---------------------------
# Initial populations
# ---------------------------
INITIAL_BACTERIA_COUNT = 3
INITIAL_MACROPHAGE_COUNT = 1
MAX_NEUTROPHIL_POOL = 12

# ---------------------------
# Macrophage model
# ---------------------------
MACROPHAGE_MAX_HEALTH = 140
MACROPHAGE_BASE_KILL_DAMAGE = 16
MACROPHAGE_ATTACK_RADIUS = 1
MACROPHAGE_SENSE_RADIUS = 6
MACROPHAGE_SIGNAL_STRENGTH = 5.0
MACROPHAGE_SIGNAL_COOLDOWN = 2

# Compatibility alias for older code paths.
MACROPHAGE_HEALTH = MACROPHAGE_MAX_HEALTH

# ---------------------------
# Bacteria model
# ---------------------------
BACTERIA_MAX_HEALTH = 24
BACTERIA_INITIAL_HEALTH = 14
BACTERIA_MATURITY_AGE = 3
BACTERIA_ATTACK_DAMAGE = 5
BACTERIA_ATTACK_PROB = 0.25
BACTERIA_SENSE_RADIUS_MACROPHAGE = 4
BACTERIA_SENSE_RADIUS_NUTRIENT = 4

# State transition and biofilm trade-offs.
ATTACHMENT_TIME_REQUIRED = 2
BIOFILM_BUILD_TIME = 3
BIOFILM_DEFENSE_MULTIPLIER = 0.55
BIOFILM_MOTILITY_PENALTY = True
BIOFILM_REPLICATION_MODIFIER = 0.75
BIOFILM_DISPERSAL_DELAY = 2

# ---------------------------
# Resource and replication model
# ---------------------------
PATCH_NUTRIENT_CAPACITY = 14.0
PATCH_REGEN_RATE = 0.2
REPLICATION_REQUIRES_ATTACHMENT = True
REPLICATION_REQUIRES_MIN_PATCH_RESOURCE = 4.0
LOCAL_CARRYING_CAPACITY = 4
BASE_REPLICATION_PROB = 0.35
IMMUNE_PRESSURE_REPLICATION_PENALTY = 0.22

# ---------------------------
# Chemokine and recruitment
# ---------------------------
CHEMOKINE_RELEASE_PER_CONTACT = 2.5
CHEMOKINE_DECAY = 0.9
CHEMOKINE_DIFFUSION_RATE = 0.06
CHEMOKINE_GRADIENT_NOISE = 0.03
NEUTROPHIL_RECRUITMENT_THRESHOLD = 8.0
NEUTROPHIL_ARRIVAL_DELAY = 7
NEUTROPHIL_SENSE_RADIUS = 7
NEUTROPHIL_KILL_DAMAGE = 22
NEUTROPHIL_LIFESPAN = 18

# ---------------------------
# Tissue damage and utility
# ---------------------------
TISSUE_DAMAGE_FROM_NEUTROPHILS = 0.45
TISSUE_DAMAGE_FROM_BACTERIA = 0.1
DAMAGE_FROM_PROLONGED_INFLAMMATION = 0.05
TISSUE_DAMAGE_FAIL_THRESHOLD = 180.0

HOST_UTILITY_ALPHA = 1.0
HOST_UTILITY_BETA = 0.35

# ---------------------------
# Bacteria strategy controls
# ---------------------------
BACTERIA_ADAPTIVE_ENABLED = True
BACTERIA_STOCHASTICITY = 0.12
BACTERIA_MODE = "adaptive"

# ---------------------------
# Planning/search
# ---------------------------
ROLLOUT_BUDGET = 40
MCTS_SIM_DEPTH = 12

# ---------------------------
# Output controls
# ---------------------------
SAVE_PLOTS = True
