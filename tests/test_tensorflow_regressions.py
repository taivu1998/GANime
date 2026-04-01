import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


tf = pytest.importorskip("tensorflow")

from dataloaders.Pix2PixDataLoader import Pix2Pix_DataLoader
from utils.evaluation_metrics import SSIM
from utils.preprocess_data import preprocess_dataset_pix2pix


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_script_module(module_name, filename):
    module_path = os.path.join(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_pair_png(path, height=8, width=8):
    left = np.full((height, width, 3), 48, dtype=np.uint8)
    right = np.full((height, width, 3), 208, dtype=np.uint8)
    image = np.concatenate([left, right], axis=1)
    Image.fromarray(image).save(path, format="PNG")


def write_jpeg(path, value):
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path, format="JPEG")


def test_preprocessed_jpeg_dataset_loads_in_pix2pix_loader(tmp_path):
    source_path = tmp_path / "source"
    output_path = tmp_path / "processed"

    for split in ["train", "val"]:
        split_path = source_path / split
        split_path.mkdir(parents=True)
        write_pair_png(split_path / "sample.png")

    preprocess_dataset_pix2pix(str(source_path), str(output_path))

    loader = Pix2Pix_DataLoader(data_path=str(output_path), batch_size=1, augment=False)
    train_dataset, test_dataset = loader.load_dataset()

    train_input, train_target = next(iter(train_dataset.take(1)))
    test_input, test_target = next(iter(test_dataset.take(1)))

    assert tuple(train_input.shape) == (1, 256, 256, 3)
    assert tuple(train_target.shape) == (1, 256, 256, 3)
    assert tuple(test_input.shape) == (1, 256, 256, 3)
    assert tuple(test_target.shape) == (1, 256, 256, 3)
    assert float(tf.reduce_max(train_input)) <= 1.0
    assert float(tf.reduce_min(train_input)) >= -1.0


def test_generate_outputs_gan_writes_every_sample(tmp_path):
    test_script = load_script_module("ganime_test_script", "test.py")

    inputs = tf.zeros((3, 8, 8, 3), dtype=tf.float32)
    targets = tf.zeros((3, 8, 8, 3), dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)

    class IdentityModel(object):
        def __call__(self, image_batch, training=False):
            return image_batch

    test_script.generate_outputs_gan(5, IdentityModel(), dataset, str(tmp_path))

    real_files = sorted((tmp_path / "Epoch 5" / "real").glob("*.jpg"))
    fake_files = sorted((tmp_path / "Epoch 5" / "fake").glob("*.jpg"))

    assert len(real_files) == 3
    assert len(fake_files) == 3


def test_ssim_can_read_jpeg_outputs(tmp_path):
    real_path = tmp_path / "real"
    fake_path = tmp_path / "fake"
    real_path.mkdir()
    fake_path.mkdir()

    write_jpeg(real_path / "sample.jpg", 127)
    write_jpeg(fake_path / "sample.jpg", 127)

    score = SSIM().evaluate(str(real_path), str(fake_path))

    assert 0.99 <= score <= 1.0


def test_style_transfer_evaluation_writes_report_and_plot(tmp_path, monkeypatch):
    evaluate_script = load_script_module("ganime_evaluate_script", "evaluate.py")

    (tmp_path / "real").mkdir()
    (tmp_path / "fake").mkdir()
    write_jpeg(tmp_path / "real" / "sample.jpg", 90)
    write_jpeg(tmp_path / "fake" / "sample.jpg", 100)

    args = SimpleNamespace(
        metric="ssim",
        model="fast_neural_style_transfer",
        output_path=str(tmp_path),
        report_path=None,
        plot_path=None,
        start_epoch=0,
        epochs=150,
        save_ckpt_freq=5,
        seed=0,
        output_channels=3,
        img_width=256,
        img_height=256,
    )

    class DummyMetric(object):
        def evaluate(self, data_path_real, data_path_fake):
            assert os.path.isdir(data_path_real)
            assert os.path.isdir(data_path_fake)
            return 0.5

    monkeypatch.setattr(evaluate_script, "parseArgs", lambda: args)
    monkeypatch.setattr(evaluate_script, "SSIM", lambda: DummyMetric())

    evaluate_script.main()

    assert (tmp_path / "ssim_fast_neural_style_transfer.txt").exists()
    assert (tmp_path / "ssim_fast_neural_style_transfer.jpg").exists()


def test_cyclegan_main_uses_cyclegan_loader(tmp_path, monkeypatch):
    test_script = load_script_module("ganime_test_script_for_loader", "test.py")

    calls = []
    output_path = tmp_path / "outputs"

    class FakePix2PixLoader(object):
        def __init__(self, *args, **kwargs):
            calls.append("pix2pix")

        def load_dataset(self):
            raise AssertionError("Pix2Pix loader should not be used for CycleGAN tests.")

    class FakeCycleGANLoader(object):
        def __init__(self, *args, **kwargs):
            calls.append("cyclegan")

        def load_dataset(self):
            dataset = tf.data.Dataset.from_tensor_slices(
                tf.zeros((1, 8, 8, 3), dtype=tf.float32)
            ).batch(1)
            return dataset, dataset, dataset, dataset

    class IdentityModel(object):
        def __call__(self, image_batch, training=False):
            return image_batch

    class FakeCycleGAN(object):
        def build_model(self, **kwargs):
            self.generator_g = IdentityModel()

        def configure_losses(self):
            pass

        def configure_optimizers(self):
            pass

        def configure_checkpoints(self, checkpoint_path):
            self.checkpoint = SimpleNamespace(step=tf.Variable(0))

        def get_checkpoints(self):
            return []

        def restore_checkpoint(self, ckpt):
            raise AssertionError("No checkpoints should be restored in this test.")

    args = SimpleNamespace(
        model="cyclegan",
        data_path=str(tmp_path),
        output_path=str(output_path),
        batch_size=1,
        img_width=256,
        img_height=256,
        output_channels=3,
        arch_gen="unet",
        arch_disc="patchgan",
        checkpoint_path=str(tmp_path / "checkpoints"),
        norm=None,
        seed=0,
    )

    monkeypatch.setattr(test_script, "parseArgs", lambda: args)
    monkeypatch.setattr(test_script, "Pix2Pix_DataLoader", FakePix2PixLoader)
    monkeypatch.setattr(test_script, "CycleGAN_DataLoader", FakeCycleGANLoader)
    monkeypatch.setattr(test_script, "CycleGAN", FakeCycleGAN)

    test_script.main()

    assert calls == ["cyclegan"]


def test_cyclegan_main_rejects_mismatched_validation_counts(tmp_path, monkeypatch):
    test_script = load_script_module("ganime_test_script_for_count_guard", "test.py")

    (tmp_path / "valA").mkdir()
    (tmp_path / "valB").mkdir()
    write_jpeg(tmp_path / "valA" / "001.jpg", 10)
    write_jpeg(tmp_path / "valA" / "002.jpg", 20)
    write_jpeg(tmp_path / "valB" / "001.jpg", 30)

    args = SimpleNamespace(
        model="cyclegan",
        data_path=str(tmp_path),
        output_path=str(tmp_path / "outputs"),
        batch_size=1,
        img_width=256,
        img_height=256,
        output_channels=3,
        arch_gen="unet",
        arch_disc="patchgan",
        checkpoint_path=str(tmp_path / "checkpoints"),
        norm=None,
        seed=0,
    )

    monkeypatch.setattr(test_script, "parseArgs", lambda: args)

    with pytest.raises(ValueError, match="matching image counts"):
        test_script.main()
