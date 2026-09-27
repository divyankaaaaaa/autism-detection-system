"""
Builds train/validation/test data generators.

Expected folder layout (already created for you under data/):

    data/
      train/
        Autistic/
        Non_Autistic/
      val/
        Autistic/
        Non_Autistic/
      test/
        Autistic/
        Non_Autistic/

Drop your existing images into the matching folders. If you only have a
single labeled folder right now, see split_dataset() below to create the
train/val/test split automatically.
"""

import os
import shutil
import random
from src import tf_compat  # noqa: F401  — must be imported before tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import config


def get_generators():
    """Returns (train_gen, val_gen, test_gen) with augmentation on train only."""
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        classes=config.CLASS_NAMES,
        shuffle=True,
    )
    val_gen = eval_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Optional helper: if your images currently live in a single folder per
    class (e.g. source_dir/Autistic, source_dir/Non_Autistic) rather than
    already split into train/val/test, this copies them into the expected
    structure under data/train, data/val, data/test.

    Usage:
        python -c "from src.data_loader import split_dataset; split_dataset('path/to/raw_data')"
    """
    random.seed(seed)
    for class_name in config.CLASS_NAMES:
        src_folder = os.path.join(source_dir, class_name)
        if not os.path.isdir(src_folder):
            print(f"Skipping {class_name}: folder not found at {src_folder}")
            continue

        files = [f for f in os.listdir(src_folder) if not f.startswith(".")]
        random.shuffle(files)

        n_train = int(len(files) * train_ratio)
        n_val = int(len(files) * val_ratio)

        splits = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split_name, split_files in splits.items():
            dest_folder = os.path.join(config.DATA_DIR, split_name, class_name)
            os.makedirs(dest_folder, exist_ok=True)
            for fname in split_files:
                shutil.copy2(os.path.join(src_folder, fname), os.path.join(dest_folder, fname))

        print(f"{class_name}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
