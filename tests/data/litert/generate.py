"""Generate the tiny .tflite fixtures for tests/test_litert.py.

Run once, in a throwaway environment. The main environment has no TensorFlow:

    uv run --no-project --python 3.12 --with tensorflow python tests/data/litert/generate.py

Both models take an NHWC image of shape (1, 4, 5, 3) and give an output of shape (1, 20, 6).
The float32 model keeps float32 tensors. The int8 model has int8 input and output tensors.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

OUTPUT_DIR = Path(__file__).parent
HEIGHT, WIDTH, CHANNELS = 4, 5, 3
OUT_CHANNELS = 6
# Column j of the weights is a fixed mix of the channels, so a wrong layout changes the output.
WEIGHTS = np.array(
    [[0.5, -0.25, 1.0, 0.0, 0.125, -1.0], [0.25, 0.5, -0.5, 1.0, 0.0, 0.75], [-0.5, 0.125, 0.25, 0.5, 1.0, 0.0]],
    dtype=np.float32,
)


def build_model() -> tf.keras.Model:
    """image (1, H, W, 3) -> 1x1 conv to 6 channels -> reshape to (1, H*W, 6)."""
    inputs = tf.keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), batch_size=1, name="images")
    conv = tf.keras.layers.Conv2D(OUT_CHANNELS, 1, use_bias=False, name="mix")
    x = conv(inputs)
    outputs = tf.keras.layers.Reshape((HEIGHT * WIDTH, OUT_CHANNELS), name="detections")(x)
    model = tf.keras.Model(inputs, outputs)
    conv.set_weights([WEIGHTS.reshape(1, 1, CHANNELS, OUT_CHANNELS)])
    return model


def representative_images():
    rng = np.random.default_rng(0)
    for _ in range(50):
        yield [rng.uniform(0.0, 1.0, size=(1, HEIGHT, WIDTH, CHANNELS)).astype(np.float32)]


def convert_float32(model: tf.keras.Model) -> bytes:
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def convert_int8(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_images
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


if __name__ == "__main__":
    model = build_model()
    (OUTPUT_DIR / "float32_nhwc.tflite").write_bytes(convert_float32(model))
    (OUTPUT_DIR / "int8_nhwc.tflite").write_bytes(convert_int8(model))
