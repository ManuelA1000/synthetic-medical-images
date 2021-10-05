#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
import pandas as pd



def extract_features(model, optimizer, dataloader, device, prefix):
    model.eval()
    probs = []
    all_preds = []
    all_labels = []
    running_loss = 0.0
    running_corrects = 0
    all_features = []
    i = 0
    for inputs, labels in dataloader:
        i += 1
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            all_features.extend(outputs.detach().cpu().tolist())

    with open(f'./out/cnn/{prefix}_features.csv', 'w') as f:
        writer = csv.writer(f)
        for ft in all_features:
            writer.writerow(ft)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract=True, use_pretrained=True):
    model_ft = None
    input_size = 0
    model_ft = models.vgg11_bn(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224
    
    print(model_ft)

    return model_ft, input_size



def main(model_path, img_dir, prefix):
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 224
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 25
    NUM_WORKERS = 0


    val_test_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                              ])

    test_dataset = datasets.ImageFolder(img_dir, val_test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    num_classes = len(test_dataset.classes)

    model_ft, input_size = initialize_model(num_classes, use_pretrained=True)
    model_ft = model_ft.to(DEVICE)
    model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_ft.classifier[6] = nn.Sequential()

    params_to_update = model_ft.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()


    print()
    print(f'Test Directory: {img_dir}')
    print()
    print(f'Device: {DEVICE}')
    print()
    print(f'Initial Learning Rate: {LEARNING_RATE}')
    print(f'Num Workers: {NUM_WORKERS}')
    print(f'Num Epochs: {NUM_EPOCHS}')
    print(f'Batch Size: {BATCH_SIZE}')
    print(f'Image Size: {IMAGE_SIZE}')
    print()


    extract_features(model_ft, optimizer_ft, test_dataloader, DEVICE, prefix)



# main('./out/cnn/vgg11_real_set_123/model.pth', './pytorch_GAN_zoo/data/resized', 'real_images_sets_123')

# main('./out/cnn/vgg11_real_set_123/model.pth', './out/train/synthetic_PGAN/', 'PGAN_images')

# main('./out/cnn/vgg11_real_set_123/model.pth', './fid_images/cycle_gan/real/', 'real_images_set_3')

# main('./out/cnn/vgg11_real_set_123/model.pth', './fid_images/cycle_gan/synthetic/', 'cyclegan_forward_images')

main('./out/cnn/vgg11_real_set_123/model.pth', './fid_images/cycle_gan/synthetic_backward/', 'cyclegan_backward_images')

