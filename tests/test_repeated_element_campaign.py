from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_campaign_script():
    script = Path(__file__).parents[1] / "scripts" / "run_repeated_element_campaign.py"
    spec = importlib.util.spec_from_file_location("run_repeated_element_campaign", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_passes_each_manifest_split_seed(tmp_path: Path) -> None:
    module = _load_campaign_script()
    splits_dir = tmp_path / "splits"
    output_dir = tmp_path / "runs"
    splits_dir.mkdir()
    campaign = {
        "splits": [
            {
                "index": 2,
                "seed": 43,
                "split_file": "split-02.json",
                "validation_elements": ["Cu"],
            }
        ]
    }
    (splits_dir / "campaign.json").write_text(json.dumps(campaign), encoding="utf-8")

    result = module.main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--splits-dir",
            str(splits_dir),
            "--output-dir",
            str(output_dir),
            "--model-seed",
            "42",
            "--dry-run",
        ]
    )

    assert result == 0
    for log_name in ("regression.log", "set_transformer.log"):
        command = (output_dir / "split-02" / log_name).read_text(encoding="utf-8")
        assert "--seed 42" in command
        assert "--split-seed 43" in command

    status = json.loads((output_dir / "campaign_status.json").read_text(encoding="utf-8"))
    assert status["runs"][0]["split_seed"] == 43
