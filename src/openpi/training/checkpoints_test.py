import json

from openpi.training import checkpoints


def test_best_metric(tmp_path):
    assert checkpoints.read_best_validation(tmp_path) == (None, float("inf"))
    checkpoints.write_best_validation_loss(tmp_path, 1000, 0.031)
    assert checkpoints.read_best_validation(tmp_path) == (1000, 0.031)
    assert json.loads((tmp_path / "best_validation.json").read_text())["step"] == 1000
