import numpy as np
import pandas as pd
from tqdm import tqdm



synth_df = pd.read_csv('./out/fid/activations-out_train_synth_1.csv')
real_train_df = pd.read_csv('./out/fid/activations-out_train_real_1.csv')
real_test_df = pd.read_csv('./out/fid/activations-out_val_test_combined_1.csv')

synth = synth_df.to_numpy()
real_train = real_train_df.to_numpy()
real_test = real_test_df.to_numpy()

real_vs_real = []
real_vs_synth = []

for ia, aa in enumerate(tqdm(real_train[:, 1:])):
    for ib, bb in enumerate(real_test[:, 1:]):
        dist = np.linalg.norm(aa - bb)
        if dist == 0:
            print(real_train[ia, 0], real_test[ib, 0])
        real_vs_real.append(dist)
    for ic, cc in enumerate(synth[:, 1:]):
        dist = np.linalg.norm(aa - cc)
        if dist == 0:
            print(real_train[ia, 0], synth[ic, 0])
        real_vs_synth.append(dist)

print(f'Min real vs real {min(real_vs_real)}')
print(f'Min real vs synth {min(real_vs_synth)}')
