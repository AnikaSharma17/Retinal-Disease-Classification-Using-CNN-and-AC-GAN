"""Model builders. All models take float32 RGB in [0, 255] and rescale internally, so the
saved model is self-contained for inference (no external normalisation to get wrong)."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def _head(x, n_outputs: int, task: str, dropout: float):
    x = layers.Dropout(dropout, name="head_dropout")(x)
    activation = "softmax" if task == "multiclass" else "sigmoid"
    return layers.Dense(n_outputs, activation=activation, name="predictions", dtype="float32")(x)


def small_cnn(img_size: int, n_outputs: int, task: str, dropout: float = 0.5) -> tf.keras.Model:
    """The original project CNN (3x Conv-BN-Pool, Dense 128), kept as an in-repo baseline."""
    inputs = layers.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for i, filters in enumerate((32, 64, 128)):
        x = layers.Conv2D(filters, 3, activation="relu", name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # Flatten in the original; GAP keeps params small
    x = layers.Dense(128, activation="relu", name="fc")(x)
    x = layers.BatchNormalization(name="fc_bn")(x)
    return tf.keras.Model(inputs, _head(x, n_outputs, task, dropout), name="small_cnn")


def _backbone(arch: str, inputs, pretrained: bool):
    weights = "imagenet" if pretrained else None
    if arch == "efficientnet_b0":  # includes its own [0,255] -> normalised preprocessing
        return tf.keras.applications.EfficientNetB0(include_top=False, weights=weights, input_tensor=inputs)
    if arch == "mobilenet_v2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)
        return tf.keras.applications.MobileNetV2(include_top=False, weights=weights, input_tensor=x)
    raise ValueError(f"unknown backbone {arch!r}")


def build_model(arch: str, img_size: int, n_outputs: int, task: str, pretrained: bool = True,
                dropout: float = 0.3) -> tuple[tf.keras.Model, tf.keras.Model | None]:
    """Returns (model, backbone). backbone is None for small_cnn (trained end to end)."""
    if arch == "small_cnn":
        return small_cnn(img_size, n_outputs, task, max(dropout, 0.5)), None
    inputs = layers.Input((img_size, img_size, 3), name="image")
    base = _backbone(arch, inputs, pretrained)
    base.trainable = False  # phase 1: train the head only
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    model = tf.keras.Model(inputs, _head(x, n_outputs, task, dropout), name=f"{arch}_classifier")
    return model, base


def unfreeze_top(base: tf.keras.Model, n_layers: int) -> int:
    """Unfreeze the last n backbone layers except BatchNorm (keeps running stats stable on tiny data)."""
    base.trainable = True
    trainable = 0
    cutoff = max(len(base.layers) - n_layers, 0)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
        trainable += int(layer.trainable)
    return trainable


def last_conv_layer_name(model: tf.keras.Model) -> str:
    """Name of the last layer with a 4-D (spatial) output: the Grad-CAM target."""
    for layer in reversed(model.layers):
        shape = getattr(layer, "output_shape", None)
        if isinstance(shape, tuple) and len(shape) == 4 and not isinstance(layer, layers.InputLayer):
            return layer.name
    raise ValueError("model has no spatial (4-D) layer for Grad-CAM")
