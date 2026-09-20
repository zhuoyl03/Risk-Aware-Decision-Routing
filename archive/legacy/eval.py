import argparse
import os
import csv
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import autocast
from tqdm import tqdm

from models import initialize_model
from process_data import get_data_loader

parser = argparse.ArgumentParser(description='Transfer learning for disaster image classification')
parser.add_argument('--name', default='vit', type=str, help='name of the run')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--task', default='informative', type=str, help='name of the task')
parser.add_argument('--train', default='data/MEDIC_train.tsv', type=str, help='path to train image files/labels')
parser.add_argument('--dev', default='data/MEDIC_dev.tsv', type=str, help='path to dev image files/labels')
parser.add_argument('--test', default='data/MEDIC_test.tsv', type=str, help='path to test image files/labels')
parser.add_argument('--data-dir', default='data/', type=str, help='root directory of images')
parser.add_argument('--fig-dir', default='out/figures', type=str, help='directory path for output figures')
parser.add_argument('--checkpoint-dir', default='out/models', type=str, help='directory for output models/states')
parser.add_argument('--checkpoint', default='stage2_best.pth', type=str, help='checkpoint filename')
parser.add_argument('--num-workers', default=0, type=int, help='dataloader workers (set 0 to avoid hang)')
parser.add_argument(
    '--risk-metric',
    default='entropy',
    type=str,
    choices=['entropy', '1-max', 'margin'],
    help='risk score for deferral ranking',
)
parser.add_argument(
    '--defer-fracs',
    default='0,0.05,0.1,0.2,0.3,0.4,0.5',
    type=str,
    help='comma-separated deferral fractions',
)
parser.add_argument('--arch', default='vit', type=str,
                    help='model architecture [vit, CLIP (default: vit)')
parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate (default: 1e-5)')
parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay (default: 0)')
parser.add_argument('--num-epochs', default=50, type=int, help='number of epochs(default: 50)')
parser.add_argument('--use-rand-augment', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='use random augment or not')
parser.add_argument('--keep-frozen', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to keep feature layers frozen (i.e., only update classification layers weight)')
parser.add_argument('--rand-augment-n', default=2, type=int,
                    help='random augment parameter N or number of augmentations applied sequentially')
parser.add_argument('--rand-augment-m', default=9, type=int,
                    help='random augment parameter M or shared magnitude across all augmentation operations')



def test_model(model, test_loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    all_severity = []
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="TEST", total=len(test_loader)):
            if batch is None:
                continue
            images, labels, severity = batch
            images = images.to(device)
            labels = labels.to(device)

            if use_cuda:
                with autocast(device_type="cuda"):
                    logits = model(images).logits
            else:
                logits = model(images).logits

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_severity.append(severity.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    severity = torch.cat(all_severity, dim=0).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    accuracy = accuracy_score(labels, preds)
    errors = (labels != preds).astype(int)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "labels": labels,
        "preds": preds,
        "probs": probs,
        "severity": severity,
        "errors": errors,
    }



def risk_scores_from_probs(probs: np.ndarray, metric: str) -> np.ndarray:
    if metric == "entropy":
        eps = 1e-12
        return -(probs * np.log(probs + eps)).sum(axis=1)
    if metric == "1-max":
        return 1.0 - probs.max(axis=1)
    if metric == "margin":
        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        top2.sort(axis=1)
        margin = top2[:, 1] - top2[:, 0]
        return 1.0 - margin
    raise ValueError(f"Unknown risk metric: {metric}")


def evaluate_deferral(labels, preds, risk_scores, severity, errors, defer_fracs):
    n = len(labels)
    order_risk = np.argsort(-risk_scores)
    order_severity = np.argsort(-severity)
    results = []
    
    for frac in defer_fracs:
        k = int(frac * n)
        defer_idx = order_risk[:k]
        severity_idx = order_severity[:k]
        error_idx = set(np.where(errors == 1)[0])

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[defer_idx] = False
        kept = keep_mask.sum()
        if kept == 0:
            acc = float("nan")
        else:
            acc = (preds[keep_mask] == labels[keep_mask]).mean()

        captured_risk = len(set(severity_idx) & set(defer_idx)) / k if k > 0 else 0.0
        severity_error_rate = len(set(error_idx) & set(severity_idx)) / len(error_idx) if len(error_idx) > 0 else 0.0
        results.append(
            {
                "defer_frac": frac,
                "kept": int(kept),
                "accuracy": float(acc),
                "risk_capture_rate": captured_risk,
                "severity_error_rate": severity_error_rate,
            }
        )
    return results




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, num_classes, _, _, _ = get_data_loader(args.name,
        args.task, args.batch_size, num_workers=args.num_workers
    )

    model, _, _ = initialize_model(args.name, num_classes, keep_frozen=False, use_pretrained=True)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    print(f"Evaluating on {len(test_loader.dataset)} samples...")
    metrics = test_model(model, test_loader, device)
    print(
        f"Test metrics | acc={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1_score']:.4f}"
    )

    defer_fracs = [float(x) for x in args.defer_fracs.split(",") if x.strip() != ""]
    risks = risk_scores_from_probs(metrics["probs"], args.risk_metric)
    severity = metrics["severity"]
    results = evaluate_deferral(metrics["labels"], metrics["preds"], risks, severity, metrics["errors"], defer_fracs)
    print(f"Deferral baseline (risk={args.risk_metric})")
    for row in results:
        print(
            f"  defer={row['defer_frac']:.2f} | kept={row['kept']} | "
            f"acc={row['accuracy']:.4f} | risk_capture_rate={row['risk_capture_rate']:.4f} | "
            f"severity_error_rate={row['severity_error_rate']:.4f}"
        )

    os.makedirs(args.fig_dir, exist_ok=True)
    curve_path = os.path.join(args.fig_dir, f"deferral_curve_{args.risk_metric}.csv")
    with open(curve_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["defer_frac", "kept", "accuracy", "risk_capture_rate", "severity_error_rate"]
        )
        writer.writeheader()
        writer.writerows(results)
    metrics_path = os.path.join(args.fig_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "checkpoint": checkpoint_path,
                "risk_metric": args.risk_metric,
                "test_metrics": {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"]
                },
            },
            f,
            indent=2,
        )
    print(f"Saved deferral curve to {curve_path}")
    print(f"Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
