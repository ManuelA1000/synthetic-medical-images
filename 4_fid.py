import os
import pathlib
from shutil import copyfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from pytorch_fid.src.pytorch_fid import fid_score

from sklearn.metrics.pairwise import euclidean_distances


def move_data(src, dst, label):
    for file in os.listdir(src):
        copyfile(os.path.join(src, file), os.path.join(dst, f'{label}_{file}'))


print('\nMoving images...')
for directory in ['train', 'synthetic', 'val_test']:
	pathlib.Path(f'./data/fid/{directory}').mkdir(parents=True, exist_ok=True)

for label in tqdm(range(1, 4)):
	move_data(f'./data/train/real/{label}', './data/fid/train', label)
	move_data(f'./data/train/synthetic/{label}', './data/fid/synthetic', label)
	move_data(f'./data/val/real/{label}', './data/fid/val_test', label)
	move_data(f'./data/test/{label}', './data/fid/val_test', label)


print('\nComputing FID...')
fid_score.main(['./data/fid/train/', './data/fid/val_test'])
fid_score.main(['./data/fid/train/', './data/fid/synthetic'])


print('\nReading activations...')
synth_df = pd.read_csv('./out/fid/activations-data_fid_synthetic.csv')
train_df = pd.read_csv('./out/fid/activations-data_fid_train.csv')
val_test_df = pd.read_csv('./out/fid/activations-data_fid_val_test.csv')

synth = synth_df.to_numpy()
train = train_df.to_numpy()
val_test = val_test_df.to_numpy()

print('\nCalculating distances: train vs val_test...')
dist = euclidean_distances(train[:, 1:], val_test[:, 1:])
print(np.nanmin(dist))
indices = np.unravel_index(np.nanargmin(dist), dist.shape)
print(train[indices[0], 0])
print(val_test[indices[1], 0])

print('\nCalculating distances: train vs synth...')
dist = euclidean_distances(train[:, 1:], synth[:, 1:])
print(np.nanmin(dist))
indices = np.unravel_index(np.nanargmin(dist), dist.shape)
print(train[indices[0], 0])
print(synth[indices[1], 0])


print('\nCalculating distances: train vs train...')
dist = euclidean_distances(train[:, 1:], train[:, 1:])
print(np.nanmin(dist))
indices = np.unravel_index(np.nanargmin(dist), dist.shape)
print(train[indices[0], 0])
print(train[indices[1], 0])
