from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import fail


def require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        fail("TensorFlow is not installed. Use Python 3.10/3.11 and install tools/ml/requirements.txt.")
    return tf


def build_model(tf, x_train: np.ndarray, class_count: int, seed: int):
    normalizer = tf.keras.layers.Normalization(axis=-1, name="normalize_landmarks")
    normalizer.adapt(x_train)
    inputs = tf.keras.Input(shape=(x_train.shape[1],), name="landmarks")
    x = normalizer(inputs)
    x = tf.keras.layers.GaussianNoise(0.015, seed=seed, name="training_noise")(x)
    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.30, seed=seed)(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.20, seed=seed + 1)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fslames_landmark_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_sequence_model(tf, x_train: np.ndarray, class_count: int, seed: int):
    normalizer = tf.keras.layers.Normalization(axis=-1, name="normalize_landmarks")
    normalizer.adapt(x_train)
    inputs = tf.keras.Input(shape=x_train.shape[1:], name="landmark_sequence")
    x = normalizer(inputs)
    x = tf.keras.layers.GaussianNoise(0.01, seed=seed)(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.30, seed=seed + 1)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="fslames_sequence_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weights(targets: np.ndarray, class_count: int) -> dict[int, float]:
    counts = np.bincount(targets, minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        fail("At least one label is missing from the training split.")
    total = counts.sum()
    return {i: float(total / (class_count * count)) for i, count in enumerate(counts)}


def export_tflite(tf, model, destination: Path, quantization: str) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization in {"float16", "dynamic"}:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
    destination.write_bytes(converter.convert())


def verify_tflite(tf, model, tflite_path: Path, sample: np.ndarray) -> dict:
    keras_probabilities = model.predict(sample[None, :], verbose=0)[0]
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], sample[None, :].astype(input_detail["dtype"], copy=False))
    interpreter.invoke()
    tflite_probabilities = interpreter.get_tensor(output_detail["index"])[0]
    keras_class = int(np.argmax(keras_probabilities))
    tflite_class = int(np.argmax(tflite_probabilities))
    if keras_class != tflite_class:
        fail("TFLite verification failed: converted and Keras predictions differ.")
    return {
        "keras_class_index": keras_class,
        "tflite_class_index": tflite_class,
        "maximum_probability_difference": float(np.max(np.abs(keras_probabilities - tflite_probabilities))),
        "input_shape": list(input_detail["shape"].astype(int)),
        "input_dtype": str(input_detail["dtype"]),
        "output_shape": list(output_detail["shape"].astype(int)),
        "output_dtype": str(output_detail["dtype"]),
    }
