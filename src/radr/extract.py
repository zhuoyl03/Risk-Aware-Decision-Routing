"""Cache fixed ViT features once so routing experiments are cheap to repeat."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTConfig, ViTForImageClassification

from .data import build_manifests

# Match historical decoding/preprocessing, and disclose this in the manifest.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Images(Dataset):
    def __init__(self, frame, root):
        self.frame, self.root = frame, Path(root)
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = self.root / self.frame.iloc[index]["path"]
        try:
            with Image.open(path) as image:
                return index, self.transform(image.convert("RGB")), ""
        except (OSError, ValueError) as exc:
            return index, None, f"{type(exc).__name__}: {exc}"


def collate(rows):
    good = [(i, image) for i, image, error in rows if image is not None]
    bad = [(i, error) for i, image, error in rows if image is None]
    return ([i for i, _ in good], torch.stack([im for _, im in good]) if good else None, bad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="out/models/stage2_best.pth")
    parser.add_argument("--output", default="out/features")
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.train_limit < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("train-limit/workers must be nonnegative; batch-size must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic()
    print("Auditing image hashes and split overlap...", flush=True)
    frames, audit = build_manifests(args.data_dir)
    if args.train_limit and len(frames["train"]) > args.train_limit:
        # Fixed random subset, independent of test and all experimental seeds.
        frames["train"] = frames["train"].sample(n=args.train_limit, random_state=2026).sort_index()
    for split, frame in frames.items():
        frame.reset_index(drop=True, inplace=True)
        frame.to_csv(out / f"{split}_manifest.csv", index=False)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Exact known architecture; no model download or hidden cache mutation.
    config = ViTConfig(num_labels=2, hidden_size=768, num_hidden_layers=12,
                       num_attention_heads=12, intermediate_size=3072,
                       layer_norm_eps=1e-12, image_size=224, patch_size=16)
    model = ViTForImageClassification(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    meta = dict(checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                checkpoint_metadata={k: v for k, v in checkpoint.items() if k != "model"},
                torch=torch.__version__, python=platform.python_version(), device=str(device),
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                preprocessing="PIL RGB; Resize256 bilinear; CenterCrop224; ToTensor; normalize .5/.5",
                precision="float32, no autocast, cuDNN/matmul TF32 disabled", truncated_images_allowed=True,
                train_subset_seed=2026, arguments=vars(args), audit=audit)
    for split, frame in frames.items():
        loader = DataLoader(Images(frame, args.data_dir), batch_size=args.batch_size,
                            num_workers=args.workers, collate_fn=collate, pin_memory=True)
        embeddings, logits, indices, failures = [], [], [], []
        split_start = time.monotonic()
        with torch.inference_mode():
            for batch_id, (ids, images, bad) in enumerate(loader):
                failures.extend(bad)
                if images is not None:
                    hidden = model.vit(images.to(device)).last_hidden_state[:, 0]
                    embeddings.append(hidden.cpu().numpy())
                    logits.append(model.classifier(hidden).cpu().numpy())
                    indices.extend(ids)
                if batch_id % 40 == 0:
                    print(f"{split}: {len(indices)}/{len(frame)} decoded; {time.monotonic()-split_start:.0f}s", flush=True)
        if not embeddings:
            raise RuntimeError(f"No readable images in {split}")
        valid = frame.iloc[indices]
        np.savez_compressed(out / f"{split}.npz", features=np.concatenate(embeddings),
                            logits=np.concatenate(logits), labels=valid.label.to_numpy(dtype=np.int64),
                            severity=valid.severity.to_numpy(dtype=np.int64),
                            sources=valid.source.to_numpy(dtype=str), paths=valid.path.to_numpy(dtype=str),
                            hashes=valid.sha256.to_numpy(dtype=str))
        meta[split] = dict(rows=len(valid), seconds=time.monotonic()-split_start,
                           decode_failures=[dict(path=frame.iloc[i].path, error=e) for i,e in failures])
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"Saved {split}: {len(valid)} images, {len(failures)} decoding failures", flush=True)
    meta["total_seconds"] = time.monotonic()-start
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
