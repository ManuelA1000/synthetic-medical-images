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


def visualize_model(model, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=25, is_inception=False, scheduler=None):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 9999.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_dataloader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_dataloader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
            # if phase == 'val' and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './out/cnn/model.pth')
                print('===MODEL SAVED===')
                
            elif phase == 'val' and epoch_acc == best_acc and epoch_loss < best_loss:
            # elif phase == 'val' and epoch_loss == best_loss and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './out/cnn/model.pth')
                print('===MODEL SAVED===')
                
            if phase == 'val': 
                val_acc_history.append(epoch_acc)
                scheduler.step(epoch_loss)                

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


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
NUM_EPOCHS = 100
IMAGE_SIZE = 299
IMAGE_RESIZE = int(IMAGE_SIZE * 1.143)
TRAIN_DIR = './out/cnn/train/synthetic/'
VAL_DIR = './out/cnn/val/real/'
TEST_DIR = './out/cnn/test/real/'

learning_rate = 0.0001

print()
print(f'Train Directory: {TRAIN_DIR}')
print(f'Validation Directory: {VAL_DIR}')
print(f'Test Directory: {TEST_DIR}')
print()
print(f'Device: {DEVICE}')
print()
print(f'Model: {MODEL_NAME}')
print(f'Initial Learning Rate: {learning_rate}')
print(f'Num Workers: {NUM_WORKERS}')
print(f'Num Epochs: {NUM_EPOCHS}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Image Size: {IMAGE_SIZE}')
print()


train_transforms = transforms.Compose([transforms.Resize(IMAGE_RESIZE),
                                       transforms.RandomResizedCrop(IMAGE_SIZE),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomRotation(25),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

val_test_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])



train_dataset = datasets.ImageFolder(TRAIN_DIR, train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, val_test_transforms)
test_dataset = datasets.ImageFolder(TEST_DIR, val_test_transforms)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)



class_names = train_dataset.classes
num_classes = len(class_names)

model_ft, input_size = initialize_model(MODEL_NAME, num_classes, use_pretrained=True)
model_ft = model_ft.to(DEVICE)

params_to_update = model_ft.parameters()
optimizer_ft = optim.SGD(params_to_update, lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, verbose=True)
criterion = nn.CrossEntropyLoss()

model_ft, hist = train_model(model_ft, train_dataloader, val_dataloader, criterion, optimizer_ft,
                             device=DEVICE, num_epochs=NUM_EPOCHS, is_inception=(MODEL_NAME=="inception"), scheduler=scheduler)



model_ft, input_size = initialize_model(MODEL_NAME, num_classes, use_pretrained=True)
model_ft = model_ft.to(DEVICE)
model_ft.load_state_dict(torch.load('./out/cnn/model.pth'))


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
        
        probs.extend(outputs.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.detach().cpu().tolist())

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

epoch_loss = running_loss / len(test_dataloader.dataset)
epoch_acc = running_corrects.double() / len(test_dataloader.dataset)

print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))

np.savetxt('./out/cnn/all_preds.csv', all_preds)
np.savetxt('./out/cnn/all_labels.csv', all_labels)
np.savetxt('./out/cnn/probs.csv', probs, delimiter =", ")
