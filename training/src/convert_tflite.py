"""Convert the trained GRU Keras model to a dynamically quantized TFLite model."""

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_MODEL_SIZE_BYTES = 500 * 1024


def make_tflite_compatible_model(model, tf):
    gru_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.GRU) and not layer.unroll
    ]
    if not gru_layers:
        return model

    input_shape = model.input_shape
    if len(input_shape) != 3 or input_shape[1] is None:
        raise ValueError(
            "TFLite conversion of an unrolled GRU requires a fixed frame count."
        )

    def clone_layer(layer):
        config = layer.get_config()
        if isinstance(layer, tf.keras.layers.GRU):
            config["unroll"] = True
        return layer.__class__.from_config(config)

    converted_model = tf.keras.models.clone_model(model, clone_function=clone_layer)
    converted_model.set_weights(model.get_weights())
    return converted_model


def main():
    parser = argparse.ArgumentParser(
        description="Convert the GRU Keras model to TensorFlow Lite."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models" / "model_gru.keras",
        help="Path to the source Keras model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "backend" / "models" / "model_gru.tflite",
        help="Path for the converted TFLite model.",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Source model does not exist: {args.model}")

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model, compile=False)
    model = make_tflite_compatible_model(model, tf)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    model_content = converter.convert()
    model_size = len(model_content)

    if model_size >= MAX_MODEL_SIZE_BYTES:
        parser.error(
            f"Converted model is {model_size} bytes; it must be smaller than "
            f"{MAX_MODEL_SIZE_BYTES} bytes."
        )

    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model_content)

    print(f"Converted model: {args.output}")
    print(f"Size: {model_size} bytes ({model_size / 1024:.1f} KiB)")
    print("TFLite interpreter loaded and tensors allocated successfully.")


if __name__ == "__main__":
    main()
