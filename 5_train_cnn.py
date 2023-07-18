import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring, EarlyStopping
from skorch.helper import predefined_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import shutil



def split_dataset_into_train_val_test(datadir, train_ratio=0.7, val_ratio=0.15):
    # Ensure input ratios are correct
    assert train_ratio >= 0 and train_ratio <= 1.0, "Invalid training set ratio"
    assert val_ratio >= 0 and val_ratio <= 1.0, "Invalid validation set ratio"
    assert train_ratio + val_ratio <= 1.0, "Sum of training and validation set ratios should be <= 1"

    # Get list of classes (subdirectories) in the input directory
    classes = os.listdir(datadir)
    for cls in classes:
        clsdir = os.path.join(datadir, cls)
        if not os.path.isdir(clsdir):
            continue

        # Get list of image files for this class
        images = os.listdir(clsdir)

        # Split the images into training, validation and test sets
        train_images, rem_images = train_test_split(images, train_size=train_ratio)
        val_images, test_images = train_test_split(rem_images, train_size=val_ratio/(1-train_ratio))

        # Create output directories if they do not exist
        for outdir in ['train', 'val', 'test']:
            if not os.path.isdir(os.path.join(datadir, outdir)):
                os.makedirs(os.path.join(datadir, outdir))

        # Copy the images into the corresponding directories
        for image in train_images:
            shutil.copy(os.path.join(clsdir, image), os.path.join(datadir, 'train', cls))
        for image in val_images:
            shutil.copy(os.path.join(clsdir, image), os.path.join(datadir, 'val', cls))
        for image in test_images:
            shutil.copy(os.path.join(clsdir, image), os.path.join(datadir, 'test', cls))



class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)


def extract(output_dir, net, datasets, test_types):
    net.module_.model.eval()
    net.module_.model.fc = nn.Identity()
    for dataset, test_type in zip(datasets, test_types):
        print()
        print(f'Extracting features from {test_type}.')
        features = net.forward(dataset)
        img_locs = [x[0] for x in dataset.samples]
        csv_data = []
        for index, img_loc in enumerate(img_locs):
            row_data = [img_loc]
            row_data.extend(features[index].tolist())
            csv_data.append(row_data)
        csv_name = os.path.join(output_dir, f'{test_type}_features.csv')
        pd.DataFrame(csv_data).to_csv(csv_name, index=False)
        print(f'Feature outputs written to: {csv_name}')


def test(output_dir, net, datasets, test_types, class_names, image_type):
    for data, test_type in zip(datasets, test_types):
        print()
        print(f'Predicting probabilities from {test_type}...')
        probs = net.predict_proba(data)
        preds = net.predict(data)
        img_paths = [x[0] for x in data.samples]
        labels = data.targets
        csv_data = {'image_path': img_paths,
                    'label': labels,
                    'prediction': preds}
        for index, class_name in enumerate(class_names):
            csv_data[class_name] = probs[:,index]
        if image_type == 'synthetic':
            csv_name = os.path.join(output_dir,
                                    f'{image_type}_{test_type}_probabilities.csv')
        else:
            csv_name = os.path.join(output_dir,
                                    f'{image_type}_{test_type}_probabilities.csv')
        pd.DataFrame(data=csv_data).to_csv(csv_name, index=False)
        print(f'Probability outputs written to: {csv_name}')


def configure_callbacks(image_type, output_dir, patience):
    f_params = os.path.join(output_dir, f'{image_type}_model.pt')
    f_history = os.path.join(output_dir, f'{image_type}_history.json')
    checkpoint = Checkpoint(monitor='valid_acc_best',
                            f_params=f_params,
                            f_history=f_history,
                            f_optimizer=None,
                            f_criterion=None)
    train_acc = EpochScoring(scoring='accuracy',
                             on_train=True,
                             name='train_acc',
                             lower_is_better=False)
    early_stopping = EarlyStopping(monitor='valid_acc',
                                   lower_is_better=False,
                                   patience=patience)
    callbacks = [checkpoint, train_acc, early_stopping]
    return callbacks


def configure_sampler(train_data):
    labels = train_data.targets
    c_names = [int(x) - 1 for x in train_data.classes]
    c_weights = np.array([1 / labels.count(c_name) for c_name in c_names])
    c_weights /= np.amin(c_weights)
    print(f'class_weights: {c_weights.tolist()}')
    print()
    img_weights = c_weights[labels]
    sampler = WeightedRandomSampler(img_weights,
                                    len(train_data),
                                    replacement=True)
    return sampler


def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.empty_cache()
    print(f'device: {device}')
    return device


def train(output_dir, train_data, val_data, image_type,
          batch_size=32, learning_rate=0.001, num_epochs=50, num_workers=16, patience=10):
    device = set_device()
    sampler = configure_sampler(train_data)
    callbacks = configure_callbacks(image_type, output_dir, patience)
    shuffle = True if sampler == None else False
    print('Training model.')
    net = NeuralNetClassifier(PretrainedModel,
                              criterion=nn.CrossEntropyLoss,
                              lr=learning_rate,
                              batch_size=batch_size,
                              max_epochs=num_epochs,
                              module__output_features=len(train_data.classes),
                              optimizer=optim.Adam,
                              iterator_train__num_workers=num_workers,
                              iterator_train__sampler=sampler,
                              iterator_train__shuffle=shuffle,
                              iterator_valid__shuffle=False,
                              iterator_valid__num_workers=num_workers,
                              train_split=predefined_split(val_data),
                              callbacks=callbacks,
                              device=device)
    net.fit(train_data, y=None)
    return net


def filter(img):
    img = np.array(img)
    img[img < 51] = 0
    img = Image.fromarray(img)
    return img


def prepare_data(input_dir, dataset,
                 image_type=None, image_size=(224,224)):
    if dataset == 'train':
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.Lambda(lambda img: filter(img)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(20),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        input_dir = os.path.join(input_dir, dataset, image_type)
    else:
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.Lambda(lambda img: filter(img)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        if dataset == 'synthetic':
            input_dir = os.path.join(input_dir, 'train', 'synthetic')
        elif dataset == 'val':
            input_dir = os.path.join(input_dir, dataset, image_type)
        else:
            input_dir = os.path.join(input_dir, dataset)
    data = ImageFolder(input_dir, transform)
    return data



if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    INPUT_DIR = 'D:/ROP/fscyyhg6vt-1/DATASET'
    OUTPUT_DIR = './out/cnn/'

    # Split the dataset into train, validation, and test sets
    split_dataset_into_train_val_test(INPUT_DIR)

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Prepare the datasets
    train_data = prepare_data(INPUT_DIR, 'train')
    val_data = prepare_data(INPUT_DIR, 'val')
    test_data = prepare_data(INPUT_DIR, 'test')

    # Train the model
    net = train(OUTPUT_DIR, train_data, val_data)

    class_names = train_data.classes

    # Test the model
    test(OUTPUT_DIR, net, [test_data], ['test_data'], class_names)

    # Extract features
    extract(OUTPUT_DIR, net, [train_data], ['synthetic'])
