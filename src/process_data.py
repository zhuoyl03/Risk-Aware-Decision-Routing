import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import *
from PIL import ImageFile
from transformers import ViTModel, ViTImageProcessor
from pathlib import Path
import dataset
import argparse

MEDIC_train = "data/MEDIC_train.tsv"
MEDIC_dev = "data/MEDIC_dev.tsv"
MEDIC_test = "data/MEDIC_test.tsv"
image_dir = "data/"
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# This implementation is based on the original work by Firoj Alam
# (https://github.com/firojalam/medic/blob/main), licensed under CC BY-NC-SA 4.0.
# The code has been modified for this project.

class SingleTaskDataset(Dataset):
    """
    Dataset class for single-task image classification
    """

    def __init__(self, file_path, task_name, sep, root_dir, transform=None):
        """

        :param file_path: File containing image file path and labels
        :param sep: separator to read label csv file
        :param root_dir: root directory for image files
        :param transform: PIL transforms to apply
        """
        self.file_path = file_path
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(file_path, sep=sep, dtype=str)
        self.X = df['image_path'].tolist()
        self.y = df[task_name].tolist()
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = list(zip(self.X, [self.class_to_idx[i] for i in self.y]))

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(os.path.join(self.root_dir, path)) as img:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

    def _find_classes(self):
        classes_set = set(self.y)
        classes = list(classes_set)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    


def transform_images(img):
    inputs = processor(images=img, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0)  # remove batch dimension added by processor



# get data for finetuning
def get_data_loader(task,
                    batch_size,
                    num_workers=4):
        # custom dataset
        train_data = SingleTaskDataset(file_path=MEDIC_train, task_name=task, sep='\t', root_dir=image_dir, transform=transform_images)
        dev_data = SingleTaskDataset(file_path=MEDIC_dev, task_name=task, sep='\t', root_dir=image_dir, transform=transform_images)
        test_data = SingleTaskDataset(file_path=MEDIC_test, task_name=task, sep='\t', root_dir=image_dir, transform=transform_images)

        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=num_workers)
        
        dev_loader = torch.utils.data.DataLoader(dev_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=num_workers)
        
        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=num_workers)

        num_classes = len(train_data.classes)
        return train_loader, dev_loader, test_loader, num_classes, train_data.class_to_idx

if __name__ == "__main__":
    train_loader, dev_loader, test_loader, num_classes, class_to_idx = get_data_loader(task='informative', batch_size=16)
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class to index mapping: {class_to_idx}")