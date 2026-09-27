"""
IMPORTANT: import this BEFORE any `tensorflow` import, in every file that
touches TensorFlow. It must be the very first import line.

Why this exists: the currently deployed trained model
(models/autism_mobilenetv2.keras) was originally saved with an older
Keras 2 version. TensorFlow >= 2.16 defaults to the newer Keras 3, which
is stricter about layer configs and fails to load that file with an error
like "Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}".

Setting this environment variable BEFORE tensorflow is imported anywhere
forces TensorFlow to use the bundled legacy Keras 2 API (via the tf-keras
package, listed in requirements.txt) instead, which loads the file fine.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
