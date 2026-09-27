"""
Run a prediction on a single image from the command line.

Usage:
    python -m src.predict path/to/image.jpg
"""

import sys
import numpy as np
from src import tf_compat  # noqa: F401  — must be imported before tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

from src import config
from src.grad_cam import generate_gradcam_overlay


def load_model():
    # compile=False: we only need inference, and the original saved model
    # references a custom 'F1_score' metric that isn't needed here.
    return tf.keras.models.load_model(config.MODEL_PATH, compile=False)


def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=config.IMG_SIZE)
    # NOTE: raw 0-255 values, NOT divided by 255 — this model has its own
    # Rescaling/Normalization layers built in (see src/config.py notes).
    img_array = image.img_to_array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    probs = model.predict(img_array, verbose=0)[0]  # shape (2,) softmax output
    class_idx = int(np.argmax(probs))
    label = config.CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])
    return label, confidence


def main():
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--gradcam"):
        print("Usage: python -m src.predict path/to/image.jpg [--gradcam]")
        sys.exit(1)

    img_path = sys.argv[1]
    model = load_model()
    label, confidence = predict_image(model, img_path)
    print(f"Prediction: {label}  (confidence: {confidence:.2%})")

    if len(sys.argv) == 3:
        original_img = Image.open(img_path).convert("RGB")
        resized = original_img.resize(config.IMG_SIZE)
        # NOTE: raw 0-255 values, matching predict_image() above.
        img_array = np.expand_dims(np.array(resized).astype("float32"), axis=0)
        overlay = generate_gradcam_overlay(model, original_img, img_array)
        out_path = "gradcam_output.png"
        overlay.save(out_path)
        print(f"Saved Grad-CAM overlay to {out_path}")


if __name__ == "__main__":
    main()
