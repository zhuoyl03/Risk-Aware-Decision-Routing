"""Route a batch of real images or cached image features to a model or review."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.special import softmax
from transformers import ViTConfig, ViTForImageClassification

from .deployment import guarded_decision
from .extract import Images
from .models import LossMixture, predict_loss


def image_features(paths, checkpoint, expected_sha256, corruption="clean"):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    path = Path(checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint differs from the encoder used to train this router")
    model = ViTForImageClassification(ViTConfig(num_labels=2, layer_norm_eps=1e-12))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    transform = Images(None, ".").transform
    if corruption not in {"clean", "blur", "jpeg"}:
        raise ValueError("Unknown image corruption")
    features, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(paths), 32):
            batch = []
            for p in paths[start:start+32]:
                with Image.open(p) as im:
                    image = im.convert("RGB")
                    # Fix the field of view before applying reproducible corruption.
                    image = transform.transforms[1](transform.transforms[0](image))
                    if corruption == "blur":
                        image = image.filter(ImageFilter.GaussianBlur(radius=2))
                    elif corruption == "jpeg":
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=15)
                        buffer.seek(0)
                        with Image.open(buffer) as decoded:
                            image = decoded.convert("RGB")
                    batch.append(transform.transforms[3](transform.transforms[2](image)))
            hidden = model.vit(torch.stack(batch).to(device)).last_hidden_state[:, 0]
            features.append(hidden.cpu().numpy())
            logits.append(model.classifier(hidden).cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)


def route_batch(bundle, features, logits, budget):
    z = bundle["expert_scaler"].transform(features)
    probabilities = np.stack([softmax(logits, axis=1)]+[head.predict_proba(z) for head in bundle["heads"]], axis=1)
    raw = np.concatenate([bundle["pca"].transform(z), probabilities.reshape(len(z), -1)], axis=1)
    x = bundle["router_scaler"].transform(raw).astype(np.float32)
    model = LossMixture(bundle["input_dim"], len(bundle["experts"]))
    model.load_state_dict(bundle["model_state"])
    model.eval()
    q = predict_loss(model, x)
    actions, scores, review, reason = guarded_decision(probabilities[:, 0], q, bundle["tail_weight"],
                                                     bundle["guard_accepted"], budget)
    predicted = probabilities.argmax(-1)[np.arange(len(actions)), actions]
    return actions, scores, review, predicted, reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", default="out/routing/router_seed17.joblib")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", nargs="+")
    inputs.add_argument("--feature-cache", help="NPZ produced by radr.extract")
    parser.add_argument("--checkpoint", default="out/models/stage2_best.pth")
    parser.add_argument("--budget", type=float, default=.2)
    parser.add_argument("--limit", type=int, default=20, help="Maximum cached rows, 0 means all")
    parser.add_argument("--output", help="Optional JSON destination")
    args = parser.parse_args()
    if not 0 <= args.budget <= 1 or args.limit < 0:
        parser.error("budget must be in [0,1], limit must be nonnegative")
    torch.set_num_threads(4)
    # This file is a local training artifact. Do not load untrusted pickle/joblib files.
    bundle = joblib.load(args.router)
    if args.feature_cache:
        digest = hashlib.sha256(Path(args.feature_cache).read_bytes()).hexdigest()
        if digest not in bundle["feature_fingerprints"].values():
            raise ValueError("Feature cache does not match this router's recorded inputs; refit or use --images")
        with np.load(args.feature_cache, allow_pickle=False) as d:
            stop = args.limit or len(d["features"])
            features, logits, paths = d["features"][:stop], d["logits"][:stop], d["paths"][:stop]
    else:
        paths = args.images
        features, logits = image_features(paths, args.checkpoint, bundle["preprocessing"]["checkpoint_sha256"])
    if not len(features):
        raise ValueError("Cannot route an empty batch")
    actions, scores, review, predicted, reason = route_batch(bundle, features, logits, args.budget)
    names = ["informative", "not_informative"]
    output = dict(batch_size=len(features), review_capacity=args.budget, review_count=int(review.sum()),
                  policy_reason=reason, scope="Research triage suggestions; human outcomes are not observed",
                  decisions=[dict(image=str(path), action="review" if r else "model",
                                  proposed_label=names[int(label)], model=bundle["experts"][int(a)],
                                  priority_score=float(score))
                             for path,a,score,r,label in zip(paths,actions,scores,review,predicted)])
    text = json.dumps(output, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
