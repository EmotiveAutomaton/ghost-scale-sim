"""Index names for states, observations, and actions.

Ghost Scale branding throughout (Spec §2): the four tiers are CREATOR, POLISHED,
CURATOR, GHOST — the published four-tier scale (100% / 95% / 60% / 5% opacity).
These names are NOT to be renamed to neutral terms.
"""

# --- Hidden state factor 0: provenance (true origin of the artifact) ---
CREATOR = 0
POLISHED = 1
CURATOR = 2
GHOST = 3
PROVENANCE_NAMES = ["CREATOR", "POLISHED", "CURATOR", "GHOST"]
TIER_OPACITY = {"CREATOR": 1.00, "POLISHED": 0.95, "CURATOR": 0.60, "GHOST": 0.05}

# --- Hidden state factor 1: creator_goal (the R being extracted) ---
GOAL_NAMES = ["G0", "G1", "G2", "G3"]

# --- Hidden state factor 2: attention (the only controllable factor) ---
DEEP = 0
SKIM = 1
ATTENTION_NAMES = ["DEEP", "SKIM"]
# Actions on factor 2 map one-to-one onto the target attention state (B[2] is a set).
ACTION_DEEP = DEEP
ACTION_SKIM = SKIM

# --- Observation modality 1: ghost_signal (declared tier) ---
SIG_CREATOR = 0
SIG_POLISHED = 1
SIG_CURATOR = 2
SIG_GHOST = 3
UNSIGNED = 4
SIGNAL_NAMES = ["SIG_CREATOR", "SIG_POLISHED", "SIG_CURATOR", "SIG_GHOST", "UNSIGNED"]
# The truthful signal emitted for each provenance tier (index i -> SIG_i).
TRUTHFUL_SIGNAL = {CREATOR: SIG_CREATOR, POLISHED: SIG_POLISHED,
                   CURATOR: SIG_CURATOR, GHOST: SIG_GHOST}

# --- Observation modality 2: effort ---
LOW_COST = 0
HIGH_COST = 1
EFFORT_NAMES = ["LOW_COST", "HIGH_COST"]

# Factor / modality indices (for readability at call sites).
F_PROVENANCE, F_GOAL, F_ATTENTION = 0, 1, 2
M_FEATURES, M_SIGNAL, M_EFFORT = 0, 1, 2
