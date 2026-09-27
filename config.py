"""
Central configuration for the Autism Detection project.
Keeping paths and hyperparameters here means train.py, evaluate.py,
predict.py, and the Streamlit app all stay in sync automatically.
"""

import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid") 
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "autism_mobilenetv2.keras")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ---- Data / model ----
# NOTE: the current trained model (from teammate's Kaggle notebook) is built
# on EfficientNet, not MobileNetV2 as originally documented, and expects
# 200x200 raw (0-255) pixel input — it has its own Rescaling/Normalization
# layers built in. If you retrain from scratch with src/train.py (which uses
# MobileNetV2), update these values back to (224, 224) and see the note in
# src/train.py about removing the /255 rescale, since that script's
# architecture is currently MobileNetV2-based, not EfficientNet-based.
IMG_SIZE = (200, 200)
BATCH_SIZE = 32
NUM_CLASSES = 2
CLASS_NAMES = ["autistic", "non_autistic"]  # must match subfolder names in data/
# ^ order assumed alphabetical (Keras' flow_from_directory default sort order).
# Double check this against how the original notebook's generator was set up
# if predictions seem swapped.

# ---- Training ----
EPOCHS_HEAD = 15               # training just the new classifier head
EPOCHS_FINE_TUNE = 10          # fine-tuning the last MobileNetV2 layers
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
FINE_TUNE_AT_LAYER = 100        # unfreeze layers from this index onward
