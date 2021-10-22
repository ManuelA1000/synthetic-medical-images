#!/usr/bin/env python
# coding: utf-8

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



class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.vgg11_bn(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, output_features)
        # model = models.resnet18(pretrained=True)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)


def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.empty_cache()
    print(f'device: {device}')
    return device


def prepare_data(dataset, args):
    assert dataset in ['train', 'val', 'test', 'set_100', 'synth']
    if dataset == 'train':
        transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(45),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        input_dir = os.path.join(args.input_dir, dataset, args.image_type)
    else:
        transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        if dataset == 'synth':
            input_dir = os.path.join(args.input_dir, 'train', 'synth_even')
        else:
            input_dir = os.path.join(args.input_dir, dataset)
    data = ImageFolder(input_dir, transform)
    return data


def configure_sampler(train_data):
    image_labels = train_data.targets
    class_names = [int(x) - 1 for x in train_data.classes]
    class_weights = np.array([1 / image_labels.count(class_name) for class_name in class_names])
    class_weights /= np.amin(class_weights)
    print(f'class_weights: {class_weights.tolist()}')
    print()
    image_weights = class_weights[image_labels]
    sampler = WeightedRandomSampler(image_weights, len(train_data), replacement=True)
    return sampler


def configure_callbacks(args):
    f_params = os.path.join(args.output_dir, f'{args.image_type}_model.pt')
    f_history = os.path.join(args.output_dir, f'{args.image_type}_history.json')
    checkpoint = Checkpoint(monitor='valid_loss_best',
                            f_params=f_params,
                            f_history=f_history,
                            f_optimizer=None,
                            f_criterion=None)
    train_acc = EpochScoring(scoring='accuracy',
                             on_train=True,
                             name='train_acc',
                             lower_is_better=False)
    early_stopping = EarlyStopping(monitor='valid_loss',
                                   lower_is_better=True,
                                   patience=args.patience)
    callbacks = [checkpoint, train_acc, early_stopping]
    return callbacks


def train(train_data, val_data, args):
    device = set_device()
    sampler = configure_sampler(train_data)
    callbacks = configure_callbacks(args)
    print('Training model...')
    net = NeuralNetClassifier(PretrainedModel,
                              criterion=nn.CrossEntropyLoss,
                              lr=args.learning_rate,
                              batch_size=args.batch_size,
                              max_epochs=args.num_epochs,
                              module__output_features=len(train_data.classes),
                              optimizer=optim.Adam,
                              # optimizer__momentum=0.9,
                              iterator_train__num_workers=args.num_workers,
                              iterator_train__sampler=sampler,
                              iterator_train__shuffle=True if sampler == None else False,
                              iterator_valid__shuffle=False,
                              iterator_valid__num_workers=args.num_workers,
                              train_split=predefined_split(val_data),
                              callbacks=callbacks,
                              device=device)
    net.fit(train_data, y=None)
    return net


def test(net, datasets, types, class_names, args):
    net.module_.model.classifier[6] = nn.Sequential(net.module_.model.classifier[6],
                                                    nn.Softmax(dim=1))
    # net.module_.model.fc = nn.Sequential(net.module_.model.fc,
    #                                      nn.Softmax(dim=1))
    for data, type in zip(datasets, types):
        print()
        print(f'Predicting probabilities from {type}...')
        probs = net.predict_proba(data)
        preds = net.predict(data)
        img_paths = [x[0] for x in data.samples]
        labels = data.targets
        csv_data = {'image_path': img_paths, 'label': labels, 'prediction': preds}
        for index, class_name in enumerate(class_names):
            csv_data[class_name] = probs[:,index]
        csv_name = os.path.join(args.output_dir, f'{args.image_type}_{type}_probabilities.csv')
        pd.DataFrame(data=csv_data).to_csv(csv_name, index=False)
        print(f'Probability outputs written to: {csv_name}')


def extract(net, datasets, types, args):
    net.module_.model.eval()
    net.module_.model.classifier[6] = nn.Identity()
    # net.module_.model.fc = nn.Identity()
    for data, type in zip(datasets, types):
        print()
        print(f'Extracting features from {type}...')
        features = net.forward(data)
        img_locs = [x[0] for x in data.samples]
        csv_data = []
        for index, img_loc in enumerate(img_locs):
            row_data = [img_loc]
            row_data.extend(features[index].tolist())
            csv_data.append(row_data)
        csv_name = os.path.join(args.output_dir, f'{type}_features.csv')
        pd.DataFrame(csv_data).to_csv(csv_name, index=False)
        print(f'Feature outputs written to: {csv_name}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_type', type=str, choices=['real', 'synth'])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--input_dir', type=str, default='./data/')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='./out/cnn')
    parser.add_argument('--patience', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_args()
    print()
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    train_data = prepare_data('train', args)
    val_data = prepare_data('val', args)
    test_data = prepare_data('test', args)
    set_100 = prepare_data('set_100', args)

    net = train(train_data, val_data, args)

    class_names = train_data.classes
    datasets = [test_data, set_100]
    types = ['test_data', 'set_100']
    test(net, datasets, types, class_names, args)
    if args.image_type == 'real':
        synth_data = prepare_data('synth', args)
        extract(net, [train_data, synth_data], ['real', 'synth'], args)
