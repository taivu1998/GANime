"""
Runtime helpers for seeding, paths, and output validation.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import random

import numpy as np
import tensorflow as tf

try:
    from utils.image_io import IMAGE_EXTENSIONS
except ImportError:
    from image_io import IMAGE_EXTENSIONS


def configure_seeds(seed):
    """Applies a seed to Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path):
    """Ensures a directory exists."""
    if path:
        os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path):
    """Ensures the parent directory for a file path exists."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)


def is_image_file_path(path):
    """Returns True if a path looks like a file path for a supported image."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def resolve_output_image_path(output_path, default_filename):
    """Treats image-like paths as file paths and all other paths as directories."""
    if output_path is None:
        raise ValueError("output_path must be provided.")

    if is_image_file_path(output_path):
        ensure_parent_dir(output_path)
        return output_path

    ensure_dir(output_path)
    return os.path.join(output_path, default_filename)


def validate_existing_dir(path, label):
    """Validates that a directory exists."""
    if not path:
        raise ValueError("{} is required.".format(label))
    if not os.path.isdir(path):
        raise ValueError("{} does not exist or is not a directory: {}".format(label, path))


def validate_existing_file(path, label):
    """Validates that a file exists."""
    if not path:
        raise ValueError("{} is required.".format(label))
    if not os.path.isfile(path):
        raise ValueError("{} does not exist or is not a file: {}".format(label, path))
