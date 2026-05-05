from pathlib import Path

from gsae.cli import main


def test_synthetic_smoke(tmp_path: Path):
    main(
        [
            "smoke",
            "--work-dir",
            str(tmp_path),
            "--layers",
            "0",
            "--hidden-dim",
            "16",
            "--latent-dim",
            "24",
            "--num-train",
            "32",
            "--num-val",
            "24",
            "--steps",
            "3",
            "--candidate-pool-size",
            "8",
            "--graph-threshold",
            "0.1",
            "--device",
            "cpu",
        ]
    )
    assert (tmp_path / "artifacts" / "gsae" / "layer_0" / "model.pt").exists()
    assert (tmp_path / "artifacts" / "bank.pt").exists()
    assert (tmp_path / "artifacts" / "gate.joblib").exists()
