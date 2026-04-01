import pytest

from options.EvaluateOptions import EvaluateOptions
from options.TrainOptions import TrainOptions


def test_train_options_use_correct_defaults():
    args = TrainOptions().parser.parse_args([])

    assert args.lambda_cycle_loss == 10.0
    assert args.augment is True
    assert args.norm is None


def test_train_options_reject_unknown_flags():
    with pytest.raises(SystemExit):
        TrainOptions().parser.parse_args(["--does-not-exist"])


def test_evaluate_options_accept_report_and_plot_paths():
    args = EvaluateOptions().parser.parse_args([
        "--metric", "ssim",
        "--report-path", "reports/scores.txt",
        "--plot-path", "reports/plot.jpg",
    ])

    assert args.metric == "ssim"
    assert args.report_path == "reports/scores.txt"
    assert args.plot_path == "reports/plot.jpg"

