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

torch.multiprocessing.freeze_support()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        print(model_ft)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()
MODEL_NAME = 'inception'
BATCH_SIZE = 32
IMAGE_SIZE = 299
TEST_DIR = './out/cnn/test/set_100/'
learning_rate = 0.0001

print()
print(f'Test Directory: {TEST_DIR}')
print()
print(f'Device: {DEVICE}')
print()
print(f'Model: {MODEL_NAME}')
print(f'Num Workers: {NUM_WORKERS}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Image Size: {IMAGE_SIZE}')
print()


val_test_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

test_dataset = datasets.ImageFolder(TEST_DIR, val_test_transforms)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

class_names = test_dataset.classes
num_classes = len(class_names)

model_ft, input_size = initialize_model(MODEL_NAME, num_classes, use_pretrained=True)
model_ft = model_ft.to(DEVICE)
model_ft.load_state_dict(torch.load('./out/cnn/model.pth', map_location=torch.device(DEVICE)))

params_to_update = model_ft.parameters()
optimizer_ft = optim.SGD(params_to_update, lr=learning_rate)
criterion = nn.CrossEntropyLoss()

## VGG11 + BN
# model_ft.classifier[6] = nn.Sequential(model_ft.classifier[6],
#                             nn.Softmax(dim=1))

## InceptionV3
model_ft.fc = nn.Sequential(model_ft.fc,
                            nn.Softmax(dim=1))

model_ft.eval()

running_loss = 0.0
running_corrects = 0

probs = []
all_preds = []
all_labels = []

for inputs, labels in test_dataloader:
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer_ft.zero_grad()

    with torch.set_grad_enabled(False):
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        
        probs.extend(outputs.detach().tolist())
        all_labels.extend(labels.detach().tolist())
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.detach().tolist())

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

epoch_loss = running_loss / len(test_dataloader.dataset)
epoch_acc = running_corrects.double() / len(test_dataloader.dataset)

print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))

np.savetxt('./out/cnn/all_preds.csv', all_preds)
np.savetxt('./out/cnn/all_labels.csv', all_labels)
np.savetxt('./out/cnn/probs.csv', probs, delimiter =", ")
