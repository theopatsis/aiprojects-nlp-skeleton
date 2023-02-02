import pandas as pd

# read DataFrame
data = pd.read_csv("train.csv")

# no of csv files with row size
k = 2
size_train = 1044896
size_val = 261216 + size_train

print(data.shape)

df = data[0:size_train]
df.to_csv(f'train_80.csv', index=False)

df = data[size_train:size_val]
df.to_csv(f'train_val_20.csv', index=False)
