"""Central configuration for the RSNA missing-view-robust screening mammography study.

Every hyper-parameter, path and seed used in the manuscript lives here so the
reproducibility claims (Notebook 08 / reproducibility pack) are backed by one
frozen, versioned config.  Nothing is scattered across notebooks.

Run-time overrides (for the supercomputer):
    RSNA_DIR   -> where train.csv / train_images / test.csv live
    OUT_DIR    -> where splits / checkpoints / figures / results are written
"""
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RSNA_DIR = os.environ.get(
    "RSNA_DIR",
    "/root/private_data/datasets/rsna-breast-cancer-detection/extracted",
)
OUT_DIR = os.environ.get("OUT_DIR", os.path.expanduser("~/breast_sci_out"))

FIG_DIR        = os.path.join(OUT_DIR, "figures")
CACHE_DIR      = os.path.join(OUT_DIR, "img_cache")
SPLIT_DIR      = os.path.join(OUT_DIR, "splits")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")
RESULT_DIR     = os.path.join(OUT_DIR, "results")

TRAIN_CSV        = os.path.join(RSNA_DIR, "train.csv")
TEST_CSV         = os.path.join(RSNA_DIR, "test.csv")      # UNLABELED competition set
TRAIN_IMAGES_DIR = os.path.join(RSNA_DIR, "train_images")

for _d in (OUT_DIR, FIG_DIR, CACHE_DIR, SPLIT_DIR, CHECKPOINT_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Task & view definition
# --------------------------------------------------------------------------- #
# Breast-level unit = patient_id + laterality.  Extra lateral views are pooled
# into the MLO channel; CC stays its own channel.  Genuinely single-view
# breasts are kept with the missing channel flagged (never dropped).
VIEW_CC   = {"CC"}
VIEW_MLO  = {"MLO", "ML", "LM", "LMO", "AT", "FB"}
IMG_SIZE  = 768                      # preprocessing output resolution (raised from 512 to test resolution hypothesis)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
BACKBONE  = "resnet50"               # resnet50 (kept for small-data regularisation; resnet101 available as fallback)
D_MODEL   = 512                      # view-token dimensionality
N_HEADS   = 8                        # transformer heads
N_LAYERS  = 2                        # transformer layers
PRETRAINED = True                    # ImageNet init
DROP_RATE = 0.1

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
EPOCHS              = 30                 # keep =30: all methods (B2/P1/P2/P3/B0/B1) trained with 30 epochs
EARLY_STOP_PATIENCE = 12             # 12 (was 8) to avoid premature stop on noisy 81-pos val
BATCH               = 16             # breast-pairs per step (8-16 per protocol)
LR                  = 1e-4           # legacy single-LR (fallback)
BACKBONE_LR         = 1e-4           # revert: 1e-5 was too low (P2 stopped learning)
HEAD_LR             = 1e-4           # revert: equal to backbone (proven for B2)
POS_SAMPLE_FRAC     = 0.2            # target positive fraction in balanced sampler (was 0.5)
AUGMENT             = True           # strong augment (rotate/scale/translate/contrast)
USE_COSINE          = True           # cosine LR schedule
WARMUP_EPOCHS       = 3              # linear LR warmup epochs
WD                  = 1e-4
KD_T                = 2.0            # distillation temperature
LAMBDA_MASKED       = 0.5            # weight of masked-view classification loss (was 1.0)
LAMBDA_KD           = 0.25           # weight of KL(teacher||student) (was 1.0; T^2=4 made it ~4x)
LAMBDA_GROUP        = 0.1            # GroupDRO weight (P3 only)
GROUP_DRO_STEP_SIZE = 0.01           # eta_q adversarial step
MASK_PROB           = 0.5            # prob. of masking one present view during training
FOCAL_GAMMA         = 2.0

# >=3 random seeds required by the protocol; report every seed, never only the best.
SEEDS = [42, 2024, 7]

# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
BOOTSTRAP          = 2000            # patient-level (clustered) bootstrap
FIXED_SPECIFICITY  = 0.90            # report sensitivity at this locked specificity
FIXED_SENSITIVITY  = 0.80            # report specificity at this locked sensitivity

# --------------------------------------------------------------------------- #
# Feasibility gates (protocol section 4).  These are PROJECT gates, not RSNA
# official criteria.  A claim is only attempted when its gate passes.
# --------------------------------------------------------------------------- #
MIN_TEST_POSITIVES     = 100         # locked test cancer breasts
MIN_VIEW_COMPLETENESS  = 0.80        # cancer breasts with CC AND MLO both present
MIN_MACHINE_GROUPS     = 3           # pre-specified machine groups for E5 / P3
MIN_MACHINE_POSITIVES  = 30          # positives per machine group
