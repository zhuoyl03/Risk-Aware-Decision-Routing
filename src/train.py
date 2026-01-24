import os
import argparse
import json
import copy
import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import initialize_model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train_model(model, dataloaders, criterion, device, optimizer, task, batch_size, num_epochs, learning_rate):
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    phases = ['train', 'val']
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase], total = len(dataloaders[phase]), 
                                       desc="[{}/{}] {} Iteration".format(epoch, num_epochs - 1, phase.upper())):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(pixel_values=inputs)
                logits = outputs.logits
                loss= criterion(logits, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                preds = logits.argmax(dim=1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

        




        

    
