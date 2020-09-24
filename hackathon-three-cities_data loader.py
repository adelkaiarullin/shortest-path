import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
l1_nrows = 1512
l2_nrows = 1248
n_horizons = 4
y_height = 3001

# Загружаем значения высот в узлах сетки для срезов L1 и L2

all_data_l1 = np.load(os.path.join(DATA_DIR, "all_data_L1.npy"))
all_data_l2 = np.load(os.path.join(DATA_DIR, "all_data_L2.npy"))

assert all_data_l1.shape == (l1_nrows, y_height), "Неправильный размер all_data_L1.npy"
assert all_data_l2.shape == (l2_nrows, y_height), "Неправильный размер all_data_L2.npy"

# Загружаем горизонты

l1_horizons_train = pd.read_csv(os.path.join(DATA_DIR, "L1_horizons_train.csv"))
l2_horizons_train = pd.read_csv(os.path.join(DATA_DIR, "L2_horizons_train.csv"))

assert l1_horizons_train.shape == (l1_nrows, n_horizons+1)
assert l2_horizons_train.shape == (l2_nrows, n_horizons+1)

# Нам необходимо предсказать значения второго горизонта среза L1 для всех x в промежутке 522, ..., 1451



# Посмотрим на пример посылки решения

sample_submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
my_submission = sample_submission.copy()

model = lambda x: 4
my_submission.y = my_submission.x.apply(model)

my_submission.to_csv("my_submission.csv", index=False)