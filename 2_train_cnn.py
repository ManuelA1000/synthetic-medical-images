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

	last_update = 0

	val_acc_history = []

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	best_loss = 9999.0

	for epoch in range(num_epochs):
		print(f'Epoch {epoch + 1}/{num_epochs}')
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
				dataloader = train_dataloader
			else:
				model.eval()
				dataloader = val_dataloader

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloader:
				inputs = inputs.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					if is_inception and phase == 'train':
						outputs, aux_outputs = model(inputs)
						loss1 = criterion(outputs, labels)
						loss2 = criterion(aux_outputs, labels)
						loss = loss1 + 0.4*loss2
					else:
						outputs = model(inputs)
						loss = criterion(outputs, labels)

					_, preds = torch.max(outputs, 1)

					if phase == 'train':
						loss.backward()
						optimizer.step()
						   
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / len(dataloader.dataset)
			epoch_acc = running_corrects.double() / len(dataloader.dataset)

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			if phase == 'val' and epoch_acc > best_acc:
				last_update = 0
			# if phase == 'val' and epoch_loss < best_loss:
				best_acc = epoch_acc
				best_loss = epoch_loss
				best_model_wts = copy.deepcopy(model.state_dict())
				torch.save(model.state_dict(), './out/cnn/model.pth')
				print('===MODEL SAVED===')
				
			elif phase == 'val' and epoch_acc == best_acc and epoch_loss < best_loss:
				last_update = 0
			# elif phase == 'val' and epoch_loss == best_loss and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_loss = epoch_loss
				best_model_wts = copy.deepcopy(model.state_dict())
				torch.save(model.state_dict(), './out/cnn/model.pth')
				print('===MODEL SAVED===')
				
			if phase == 'val': 
				val_acc_history.append(epoch_acc)
				scheduler.step(epoch_loss)                

		last_update += 1
		if last_update == 20:
			break
		print()

	time_elapsed = time.time() - since
	print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
	print(f'Best val Acc: {best_acc:4f}')
	model.load_state_dict(best_model_wts)
	return model, val_acc_history


def test_model(model, optimizer, criterion, dataloader, prefix):
	model.eval()
	probs = []
	all_preds = []
	all_labels = []
	running_loss = 0.0
	running_corrects = 0	

	for inputs, labels in dataloader:
		inputs = inputs.to(DEVICE)
		labels = labels.to(DEVICE)
		optimizer.zero_grad()
		with torch.set_grad_enabled(False):
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			probs.extend(outputs.detach().cpu().tolist())
			all_labels.extend(labels.detach().cpu().tolist())
			_, preds = torch.max(outputs, 1)
			all_preds.extend(preds.detach().cpu().tolist())

		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(preds == labels.data)

	epoch_loss = running_loss / len(dataloader.dataset)
	epoch_acc = running_corrects.double() / len(dataloader.dataset)
	print(f'{prefix} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

	np.savetxt(f'./out/cnn/{prefix}_preds.csv', all_preds)
	np.savetxt(f'./out/cnn/{prefix}_labels.csv', all_labels)
	np.savetxt(f'./out/cnn/{prefix}_probs.csv', probs, delimiter =", ")


def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
	model_ft = None
	input_size = 0

	if model_name == "resnet":
		model_ft = models.resnet152(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs // 2),
									nn.ReLU(inplace=True),
									nn.Dropout(0.6),
									nn.Linear(num_ftrs // 2, num_classes))
		input_size = 224


	elif model_name in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:

		if model_name == 'vgg11':
			model_ft = models.vgg11_bn(pretrained=use_pretrained)
		elif model_name == 'vgg13':
			model_ft = models.vgg13_bn(pretrained=use_pretrained)
		elif model_name == 'vgg16':
			model_ft = models.vgg16_bn(pretrained=use_pretrained)
		else:
			model_ft = models.vgg19_bn(pretrained=use_pretrained)

		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
		input_size = 224


	elif model_name == "inception":
		model_ft = models.inception_v3(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)

		num_ftrs = model_ft.AuxLogits.fc.in_features
		model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs,num_classes)
		input_size = 299

	else:
		print("Invalid model name, exiting...")
		exit()

	return model_ft, input_size


BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
LEARNING_RATE = 0.01
MODEL_NAME = 'vgg11'
NUM_EPOCHS = 100
NUM_WORKERS = 16 #os.cpu_count()
SET_100_DIR = './out/cnn/posterior_view/set_100/'
TEST_DIR = './out/cnn/posterior_view/test/'
TRAIN_DIR = './out/cnn/posterior_view/synthetic/'
# TRAIN_DIR = './out/split_3/train/synthetic_balanced/'
# TRAIN_DIR = './out/cnn/train/real/'
VAL_DIR = './out/cnn/posterior_view/val/'


image_resize = int(IMAGE_SIZE * 1.143)

train_transforms = transforms.Compose([transforms.Resize(image_resize),
									   transforms.RandomResizedCrop(IMAGE_SIZE),
									   transforms.RandomHorizontalFlip(),
									   transforms.RandomVerticalFlip(),
									   transforms.RandomRotation(45),
									   transforms.ToTensor(),
									   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
									   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
									  ])

val_test_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
										  transforms.ToTensor(),
										  # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
										  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
										  ])



train_dataset = datasets.ImageFolder(TRAIN_DIR, train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, val_test_transforms)
test_dataset = datasets.ImageFolder(TEST_DIR, val_test_transforms)
set_100_dataset = datasets.ImageFolder(SET_100_DIR, val_test_transforms)



train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
set_100_dataloader = torch.utils.data.DataLoader(set_100_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)



class_names = train_dataset.classes
num_classes = len(class_names)

model_ft, input_size = initialize_model(MODEL_NAME, num_classes, use_pretrained=True)
model_ft = model_ft.to(DEVICE)

params_to_update = model_ft.parameters()
optimizer_ft = optim.SGD(params_to_update, lr=LEARNING_RATE)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, verbose=True)

counts = torch.unique(torch.tensor(train_dataset.targets), return_counts=True)[1].tolist()
majority_class = max(counts)
weight = torch.tensor([majority_class/counts[0], majority_class/counts[1], majority_class/counts[2]]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weight)

print()
print(f'Train Directory: {TRAIN_DIR}')
print(f'Validation Directory: {VAL_DIR}')
print(f'Test Directory: {TEST_DIR}')
print()
print(f'Device: {DEVICE}')
print()
print(f'Model: {MODEL_NAME}')
print(f'Initial Learning Rate: {LEARNING_RATE}')
print(f'Num Workers: {NUM_WORKERS}')
print(f'Num Epochs: {NUM_EPOCHS}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Image Size: {IMAGE_SIZE}')
print(f'Class weighting: {weight}')
print()

model_ft, hist = train_model(model_ft, train_dataloader, val_dataloader, criterion, optimizer_ft,
							 device=DEVICE, num_epochs=NUM_EPOCHS, is_inception=(MODEL_NAME=="inception"), scheduler=scheduler)


model_ft, input_size = initialize_model(MODEL_NAME, num_classes, use_pretrained=False)
model_ft = model_ft.to(DEVICE)
model_ft.load_state_dict(torch.load('./out/cnn/model.pth'))


if MODEL_NAME in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
	model_ft.classifier[6] = nn.Sequential(model_ft.classifier[6], nn.Softmax(dim=1))
elif MODEL_NAME == 'inception':
	model_ft.fc = nn.Sequential(model_ft.fc, nn.Softmax(dim=1))
elif MODEL_NAME == 'resnet':
	model_ft.fc = nn.Sequential(model_ft.fc, nn.Softmax(dim=1))
else:
	print('Did not append Softmax layer before testing!')


test_model(model_ft, optimizer_ft, criterion, test_dataloader, 'test')

test_model(model_ft, optimizer_ft, criterion, set_100_dataloader, 'set_100')
