import os
import argparse
import copy
import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import initialize_model
from process_data import get_data_loader
from torch.amp import autocast, GradScaler
import json


parser = argparse.ArgumentParser(description='Transfer learning for disaster image classification')
parser.add_argument('--name', default='vit', type=str, help='name of the run')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--task', default='informative', type=str, help='name of the task')
parser.add_argument('--train', default='data/MEDIC_train.tsv', type=str, help='path to train image files/labels')
parser.add_argument('--dev', default='data/MEDIC_dev.tsv', type=str, help='path to dev image files/labels')
parser.add_argument('--test', default='data/MEDIC_test.tsv', type=str, help='path to test image files/labels')
parser.add_argument('--savepath', default='out/models/best.pth', type=str, help='path to checkpoint file')
parser.add_argument('--sep', default='\t', type=str, help='column separator used in csv(default: "\t")')
parser.add_argument('--data-dir', default='data/', type=str, help='root directory of images')
parser.add_argument('--best-state-path', default='models/best.pth', type=str, help='path to best state checkpoint')
parser.add_argument('--fig-dir', default='out/figures', type=str, help='directory path for output figures')
parser.add_argument('--checkpoint-dir', default='out/models', type=str, help='directory for output models/states')
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train_model(
    model,
    dataloaders,
    criterion,
    device,
    scheduler,
    optimizer,
    num_epochs,
    save_path=None,
    patience=5,
    min_delta=1e-4,
):
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_acc = 0.0
    no_improve = 0

    use_cuda = (device.type == "cuda")
    scaler = GradScaler("cuda") if use_cuda else None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        epoch_stats = {}

        for phase in ["train", "val"]:
            is_train = (phase == "train")
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            for batch in tqdm(
                dataloaders[phase],
                total=len(dataloaders[phase]),
                desc=f"[Epoch {epoch+1}/{num_epochs}] {phase.upper()}",
                leave=(phase == "train"),
            ):
                if batch is None:
                    continue

                inputs, labels, severity = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    if use_cuda:
                        with autocast(device_type="cuda"):
                            logits = model(inputs).logits
                            loss = criterion(logits, labels)
                        if is_train:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        logits = model(inputs).logits
                        loss = criterion(logits, labels)
                        if is_train:
                            loss.backward()
                            optimizer.step()

                preds = logits.argmax(dim=1)
                bs = labels.size(0)

                n_samples += bs
                running_loss += loss.item() * bs
                running_corrects += (preds == labels).sum().item()

            if n_samples == 0:
                epoch_loss = float("nan")
                epoch_acc = float("nan")
            else:
                epoch_loss = running_loss / n_samples
                epoch_acc = running_corrects / n_samples

            epoch_stats[phase] = {"loss": epoch_loss, "acc": epoch_acc}


        history["train_loss"].append(epoch_stats["train"]["loss"])
        history["val_loss"].append(epoch_stats["val"]["loss"])
        history["train_acc"].append(epoch_stats["train"]["acc"])
        history["val_acc"].append(epoch_stats["val"]["acc"])

        val_loss = epoch_stats["val"]["loss"]
        val_acc = epoch_stats["val"]["acc"]

        if not (val_acc != val_acc): 
            best_acc = max(best_acc, val_acc)

        # scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # early stopping / save best on val loss
        if (val_loss + min_delta) < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0

            if save_path is not None:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "model": best_model_wts,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "val_acc": val_acc,
                    },
                    save_path,
                )

            print(f"[BEST] val loss improved to {best_val_loss:.4f}")
        else:
            no_improve += 1
            print(f"[EARLYSTOP] no improvement: {no_improve}/{patience}")

        

        if no_improve >= patience:
            print(f"[EARLYSTOP] stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
            break
        
        print(f"Epoch {epoch+1}: "
        f"train_loss={epoch_stats['train']['loss']:.4f}, train_acc={epoch_stats['train']['acc']:.4f} | "
        f"val_loss={epoch_stats['val']['loss']:.4f}, val_acc={epoch_stats['val']['acc']:.4f}")


    model.load_state_dict(best_model_wts)
    time_elapsed = datetime.now() - since
    return model, best_acc, time_elapsed, history



def set_trainable(model, train_backbone: bool):
    for p in model.vit.parameters():
        p.requires_grad = train_backbone
    for p in model.classifier.parameters():
        p.requires_grad = True


def build_param_groups_vit(model, lr_backbone, lr_head, weight_decay):
    decay, no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = (
            name.endswith(".bias")
            or "LayerNorm.weight" in name
            or "layernorm.weight" in name.lower()
            or "layer_norm.weight" in name.lower()
        )

        if name.startswith("vit."):
            (no_decay if is_no_decay else decay).append(p)
        elif name.startswith("classifier."):
            (head_no_decay if is_no_decay else head_decay).append(p)
        else:
            (no_decay if is_no_decay else decay).append(p)

    return [
        {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0},
    ]


def make_optimizer_stage1(model, lr_head=3e-4, weight_decay=0.05):
    param_groups = build_param_groups_vit(model, lr_backbone=0.0, lr_head=lr_head, weight_decay=weight_decay)
    param_groups = [g for g in param_groups if g["lr"] > 0 and len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups)

def make_optimizer_stage2(model, lr_backbone=1e-5, lr_head=1e-4, weight_decay=0.05):
    param_groups = build_param_groups_vit(model, lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay)
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)
    model_name = args.arch
    # data
    train_loader, dev_loader, test_loader, num_classes, _, _, _ = get_data_loader(model_name, args.task, args.batch_size)
    dataloaders = {"train": train_loader, "val": dev_loader}

    # model
    model, _, _ = initialize_model(model_name, num_classes, keep_frozen=False, use_pretrained=True)
    model.to(device)

    # dirs / ckpts
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    save_path_stage1 = os.path.join(args.checkpoint_dir, "stage1_best.pth")
    save_path_stage2 = os.path.join(args.checkpoint_dir, "stage2_best.pth")

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0)

    # -------------------
    # Stage 1: train head only
    # -------------------
    set_trainable(model, train_backbone=False)
    optimizer1 = make_optimizer_stage1(model, lr_head=3e-4, weight_decay=0.05)

    model, best_acc_1, elapsed_1, hist1 = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        device=device,
        scheduler=None,
        optimizer=optimizer1,
        num_epochs=2,
        save_path=save_path_stage1,
        patience=2,
    )

    # -------------------
    # Stage 2: unfreeze backbone + discriminative lr
    # -------------------
    set_trainable(model, train_backbone=True)
    optimizer2 = make_optimizer_stage2(model, lr_backbone=3e-6, lr_head=3e-5, weight_decay=0.05)


    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="min", factor=0.5, patience=5
    )

    model, best_acc_2, elapsed_2, hist2 = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        device=device,
        scheduler=scheduler2,
        optimizer=optimizer2,
        num_epochs=5,
        save_path=save_path_stage2,
        patience=2,
        min_delta=1e-3,
    )

    # merge history for plotting / saving
    history = {
        "train_loss": hist1["train_loss"] + hist2["train_loss"],
        "val_loss":   hist1["val_loss"]   + hist2["val_loss"],
        "train_acc":  hist1["train_acc"]  + hist2["train_acc"],
        "val_acc":    hist1["val_acc"]    + hist2["val_acc"],
        "stage1_len": len(hist1["train_loss"]),
    }

    print(f"Stage1 best acc: {best_acc_1:.4f}, time: {elapsed_1}")
    print(f"Stage2 best acc: {best_acc_2:.4f}, time: {elapsed_2}")

    # save history for reproducibility
    with open(os.path.join(args.fig_dir, "loss_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # plot loss
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss")

    if history["stage1_len"] > 0:
        plt.axvline(history["stage1_len"] + 0.5, linestyle="--", linewidth=1)

    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.fig_dir, "loss_curve.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()




        

    
