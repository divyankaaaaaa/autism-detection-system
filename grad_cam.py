"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the autism
detection model.

Produces a heatmap showing which regions of the input image most
influenced the model's prediction, overlaid on the original image.

NOTE: this targets the currently-deployed trained model, which is an
EfficientNet-based architecture with a 2-unit softmax output (see
src/config.py notes). Unlike a MobileNetV2-with-nested-submodel setup,
this model's layers are all flat/top-level, so we target the last
convolutional-ish layer by name directly.

Usage (standalone):
    python -m src.grad_cam path/to/image.jpg
"""

import sys
import numpy as np
import cv2
from src import tf_compat  # noqa: F401  — must be imported before tensorflow
import tensorflow as tf
from PIL import Image

from src import config

# Last spatial-feature layer before global pooling. For the EfficientNet-based
# model this is 'top_activation'. If you swap in a different architecture,
# update this (or write logic to auto-detect the last 4D-output layer).
LAST_CONV_LAYER_NAME = "top_activation"


def make_gradcam_heatmap(img_array, model, target_class=None, last_conv_layer_name=LAST_CONV_LAYER_NAME):
    """
    img_array: preprocessed image, shape (1, H, W, 3) — raw 0-255 range for
        this model, since it has Rescaling/Normalization built in.
    target_class: which class index to explain. If None, uses the model's
        own top prediction (argmax).
    Returns a 2D heatmap (values 0-1).
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if target_class is None:
            target_class = tf.argmax(predictions[0])
        loss = predictions[:, target_class]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img: Image.Image, alpha=0.4):
    """Resizes heatmap to the original image size and overlays it as a color map."""
    heatmap_resized = cv2.resize(heatmap, original_img.size)  # PIL size = (W, H)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    original_arr = np.array(original_img.convert("RGB"))
    overlayed = (colored * alpha + original_arr * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlayed)


def generate_gradcam_overlay(model, original_img: Image.Image, img_array, target_class=None):
    """Convenience wrapper: returns a PIL image with the Grad-CAM overlay applied."""
    heatmap = make_gradcam_heatmap(img_array, model, target_class=target_class)
    return overlay_heatmap(heatmap, original_img)


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.grad_cam path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)

    original_img = Image.open(img_path).convert("RGB")
    resized = original_img.resize(config.IMG_SIZE)
    # NOTE: raw 0-255 values — this model rescales/normalizes internally.
    img_array = np.expand_dims(np.array(resized).astype("float32"), axis=0)

    overlay = generate_gradcam_overlay(model, original_img, img_array)
    out_path = "gradcam_output.png"
    overlay.save(out_path)
    print(f"Saved Grad-CAM overlay to {out_path}")


if __name__ == "__main__":
    main()
