from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.configuration import load_config, resolve_project_path
from vision_track.device import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Structurally prune a fine-tuned YOLO model")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "app.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--sparsity", type=float)
    parser.add_argument("--recovery-epochs", type=int)
    args = parser.parse_args()

    import torch
    import torch_pruning as tp
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    config = load_config(args.config)
    pruning = config.raw["pruning"]
    selected = select_device(force=None if args.device == "auto" else args.device)
    source = resolve_project_path(config.model.checkpoint)
    if not source.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {source}")
    dataset_yaml = resolve_project_path(config.raw["training"]["dataset_yaml"])
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {dataset_yaml}")

    wrapper = YOLO(str(source), task="detect")
    network = wrapper.model.to(selected.kind)
    network.train()
    for parameter in network.parameters():
        parameter.requires_grad_(True)
    example = torch.randn(
        1, 3, config.model.image_size, config.model.image_size, device=selected.kind
    )
    before_ops, before_params = tp.utils.count_ops_and_params(network, example)
    ignored = [module for module in network.modules() if isinstance(module, Detect)]
    pruner = tp.pruner.MagnitudePruner(
        network,
        example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=float(args.sparsity or pruning["channel_sparsity"]),
        iterative_steps=int(pruning["iterative_steps"]),
        ignored_layers=ignored,
    )
    pruner.step()
    after_ops, after_params = tp.utils.count_ops_and_params(network, example)
    if after_params >= before_params or after_ops >= before_ops:
        raise RuntimeError("Structured pruning did not reduce parameters and operations")

    interim = ROOT / "models" / "checkpoints" / "pruned_before_recovery.pt"
    wrapper.save(str(interim))
    recovery = YOLO(str(interim), task="detect")
    train_results = recovery.train(
        data=str(dataset_yaml),
        epochs=args.recovery_epochs or int(pruning["recovery_epochs"]),
        imgsz=config.model.image_size,
        batch=int(config.raw["training"]["batch_size"]),
        patience=max(3, int(config.raw["training"]["patience"]) // 2),
        device=selected.torch_device,
        seed=config.seed,
        project=str(ROOT / "models" / "training_runs"),
        name="person_yolo26n_pruned_recovery",
        exist_ok=True,
        val=True,
        classes=[config.model.person_class_id],
    )
    recovered_best = Path(train_results.save_dir) / "weights" / "best.pt"
    destination = resolve_project_path(config.model.pruned_checkpoint)
    shutil.copy2(recovered_best, destination)
    report = {
        "status": "pruned",
        "method": pruning["method"],
        "channel_sparsity": float(args.sparsity or pruning["channel_sparsity"]),
        "before_parameter_count": int(before_params),
        "after_parameter_count": int(after_params),
        "before_flops": int(before_ops),
        "after_flops": int(after_ops),
        "parameter_reduction": 1 - after_params / before_params,
        "flops_reduction": 1 - after_ops / before_ops,
        "recovery_epochs": args.recovery_epochs or int(pruning["recovery_epochs"]),
        "artifact": str(destination),
    }
    (ROOT / "reports" / "pruning_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

