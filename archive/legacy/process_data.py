import os
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T
from transformers import ViTImageProcessor, CLIPProcessor
import argparse 
ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser(description='Transfer learning for disaster image classification')
parser.add_argument('--name', default="vit", type=str, help='name of the run')
parser.add_argument('--task', default="informative", type=str, help='task of the run')

VIT_SIZE = 224  # ViT base patch16 default
import warnings
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency.*"
)

MEDIC_train = "data/MEDIC_train.tsv"
MEDIC_dev = "data/MEDIC_dev.tsv"
MEDIC_test = "data/MEDIC_test.tsv"
image_dir = "data/"



def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def make_transform(aug: T.Compose | None, processor):
    """
    Apply torchvision augmentation on PIL image first, then ViTImageProcessor
    to get pixel_values tensor [3, 224, 224].
    """
    def _tf(img: Image.Image):
        if aug is not None:
            img = aug(img) 
        inputs = processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)
    return _tf


train_aug = T.Compose([
    T.RandomResizedCrop(VIT_SIZE, scale=(0.7, 1.0), ratio=(3/4, 4/3)),
    T.RandomHorizontalFlip(p=0.5),
])

val_aug = T.Compose([
    T.Resize(256),
    T.CenterCrop(VIT_SIZE),
])


class SingleTaskDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        task_name: str,
        sep: str,
        root_dir: str,
        transform=None,
        class_to_idx=None,
        severity_to_idx=None,
        drop_unknown_labels=True,
    ):
        self.file_path = file_path
        self.root_dir = root_dir
        self.transform = transform
        self.task_name = task_name
        self.drop_unknown_labels = drop_unknown_labels

        df = pd.read_csv(file_path, sep=sep, dtype=str)

        required_cols = ["image_path", task_name, "damage_severity"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {file_path}")

        df = df.dropna(subset=required_cols).copy()
        df["image_path"] = df["image_path"].astype(str).str.strip()
        df[task_name] = df[task_name].astype(str).str.strip()
        df["damage_severity"] = df["damage_severity"].astype(str).str.strip()

        df = df[df[task_name].str.lower() != "nan"]
        df = df[df["image_path"].str.lower() != "nan"]
        df = df[df["damage_severity"].str.lower() != "nan"]

        self.X = df["image_path"].tolist()
        self.y_raw = df[task_name].tolist()
        self.severity = df["damage_severity"].tolist()

        if class_to_idx is None:
            self.classes = sorted(set(self.y_raw))
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.class_to_idx = dict(class_to_idx)
            self.classes = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

        if severity_to_idx is None:
            self.severity_classes = sorted(set(self.severity))
            self.severity_to_idx = {c: i for i, c in enumerate(self.severity_classes)}
        else:
            self.severity_to_idx = dict(severity_to_idx)
            self.severity_classes = [c for c, _ in sorted(self.severity_to_idx.items(), key=lambda kv: kv[1])]

        self.samples = []
        self._unknown_label_count = 0
        for rel_path, y, severity in zip(self.X, self.y_raw, self.severity):
            if y in self.class_to_idx:
                self.samples.append((rel_path, self.class_to_idx[y], self.severity_to_idx[severity]))
            else:
                self._unknown_label_count += 1
                if not self.drop_unknown_labels:
                    raise ValueError(
                        f"Found label '{y}' in {file_path} not present in training mapping."
                    )

        self._bad_files = set()
        self._bad_printed = 0

        if self._unknown_label_count > 0:
            print(f"[WARN] {file_path}: dropped {self._unknown_label_count} unknown-label samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel_path, label, severity = self.samples[index]
        full_path = os.path.join(self.root_dir, rel_path)

        if full_path in self._bad_files:
            return None

        try:
            img = Image.open(full_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label, severity
        except Exception as e:
            self._bad_files.add(full_path)
            if self._bad_printed < 5:
                self._bad_printed += 1
                print(f"[BAD] {full_path} -> {repr(e)}")
            return None


def get_data_loader(model_name, task: str, batch_size: int, num_workers: int = 8):

    if model_name == "vit":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        print("Using ViT processor")
    elif model_name == "CLIP":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("Using CLIP processor")
    else: 
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    train_data = SingleTaskDataset(
        file_path=MEDIC_train,
        task_name=task,
        sep="\t",
        root_dir=image_dir,
        transform=make_transform(train_aug, processor),
        class_to_idx=None,
    )

    dev_data = SingleTaskDataset(
        file_path=MEDIC_dev,
        task_name=task,
        sep="\t",
        root_dir=image_dir,
        transform=make_transform(val_aug, processor),
        class_to_idx=train_data.class_to_idx,
        severity_to_idx=train_data.severity_to_idx,
        drop_unknown_labels=True,
    )

    test_data = SingleTaskDataset(
        file_path=MEDIC_test,
        task_name=task,
        sep="\t",
        root_dir=image_dir,
        transform=make_transform(val_aug, processor),
        class_to_idx=train_data.class_to_idx,
        severity_to_idx=train_data.severity_to_idx,
        drop_unknown_labels=True,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
    )

    dev_loader = DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
    )

    num_classes = len(train_data.class_to_idx)
    num_severity_classes = len(train_data.severity_to_idx)
    return train_loader, dev_loader, test_loader, num_classes, num_severity_classes,train_data.class_to_idx, train_data.severity_to_idx


if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.name.lower()
    task = args.task.lower()
    if model_name == "vit":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        print("Using ViT processor")
    elif model_name == "clip":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("Using CLIP processor")
    else: 
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
   
    train_loader, dev_loader, test_loader, num_classes, num_severity_classes,class_to_idx, severity_to_idx = get_data_loader(model_name,
        task=task, batch_size=16
    )
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class to index mapping: {class_to_idx}")
    print(f"Number of severity classes: {num_severity_classes}")
    print(f"Severity to index mapping: {severity_to_idx}")
