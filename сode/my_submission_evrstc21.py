import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


DATA_DIR = "data"
salt = 'hackathon-three-cities_'
l1_nrows = 1512
l2_nrows = 1248
n_horizons = 4
y_height = 3001

# Загружаем значения высот в узлах сетки для срезов L1 и L2

all_data_l1 = np.load(os.path.join(DATA_DIR, salt+"all_data_L1.npy"))
all_data_l2 = np.load(os.path.join(DATA_DIR, salt+"all_data_L2.npy"))

assert all_data_l1.shape == (l1_nrows, y_height), "Неправильный размер all_data_L1.npy"
assert all_data_l2.shape == (l2_nrows, y_height), "Неправильный размер all_data_L2.npy"

# Загружаем горизонты

l1_horizons_train = pd.read_csv(os.path.join(DATA_DIR, salt+"L1_horizons_train.csv"))
l2_horizons_train = pd.read_csv(os.path.join(DATA_DIR, salt+"L2_horizons_train.csv"))

assert l1_horizons_train.shape == (l1_nrows, n_horizons+1)
assert l2_horizons_train.shape == (l2_nrows, n_horizons+1)

sample_submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
my_submission = sample_submission.copy()

all_data_l1 -= all_data_l1.min()

start_idx = 522 #my_submission.x.min()
end_idx = 1451 #my_submission.x.max()

tmp = l1_horizons_train.loc[start_idx:end_idx].copy()
y_hat = np.zeros_like(tmp.x.values)

y_prev = l1_horizons_train.loc[start_idx-1, 'hor_2']
y_prev = int(y_prev)
print(y_prev)
print(len(y_hat))

N_neighbours = 21
for i, x in enumerate(tmp.x):
    mid_point = N_neighbours // 2
    neighbours = np.array([all_data_l1.T[y_prev+j, x] for j in range(-mid_point, mid_point + 1)])
    
    assert N_neighbours == len(neighbours)
                          
    upd = np.argmax(neighbours)
    y_prev += (upd - mid_point)
    y_hat[i] = y_prev





my_submission.y = y_hat
my_submission.to_csv('my_submission_evrstc21.csv', index=False)