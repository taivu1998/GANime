"""
Image I/O helpers shared across preprocessing, data loading, testing, and metrics.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_image_paths(directory):
    """Lists supported image paths in a deterministic order."""
    image_paths = []
    for extension in IMAGE_EXTENSIONS:
        image_paths.extend(tf.io.gfile.glob(os.path.join(directory, "*" + extension)))
    return sorted(image_paths)


def resolve_tf_dtype(dtype):
    """Resolves a TensorFlow dtype from either a dtype object or dtype name."""
    if isinstance(dtype, tf.dtypes.DType):
        return dtype

    if isinstance(dtype, str):
        if not hasattr(tf, dtype):
            raise ValueError("Unsupported TensorFlow dtype: {}".format(dtype))
        resolved = getattr(tf, dtype)
        if not isinstance(resolved, tf.dtypes.DType):
            raise ValueError("Unsupported TensorFlow dtype: {}".format(dtype))
        return resolved

    raise TypeError("dtype must be a TensorFlow dtype or dtype name.")


def read_image(image_file, dtype=tf.uint8, channels=3):
    """Reads a JPEG or PNG image and returns a dense tensor with a fixed channel shape."""
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=channels, expand_animations=False)
    image.set_shape([None, None, channels])
    return tf.cast(image, resolve_tf_dtype(dtype))


def infer_image_format(image_path, default_format="jpeg"):
    """Infers an image format from a file path."""
    extension = os.path.splitext(image_path)[1].lower()
    if extension in (".jpg", ".jpeg"):
        return "jpeg"
    if extension == ".png":
        return "png"
    return default_format


def encode_image(image, image_format="jpeg"):
    """Encodes an image tensor using the requested image format."""
    image = tf.cast(image, tf.uint8)
    image_format = image_format.lower()

    if image_format in ("jpg", "jpeg"):
        return tf.io.encode_jpeg(image)
    if image_format == "png":
        return tf.io.encode_png(image)

    raise ValueError("Unsupported image format: {}".format(image_format))


def write_image(image, image_path, image_format=None):
    """Writes an image tensor to disk."""
    if image_format is None:
        image_format = infer_image_format(image_path)
    encoded = encode_image(image, image_format=image_format)
    tf.io.write_file(image_path, encoded)
