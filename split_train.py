import pandas as pd

# read DataFrame
data = pd.read_csv("train.csv")

# no of csv files with row size
k = 2
size = 363000

for i in range(k):
    df = data[size * i:size * (i + 1)]
    df.to_csv(f'train_{i + 1}.csv', index=False)

df_1 = pd.read_csv("train_1.csv")
df_1.to_csv("train_split_new.csv")

df_2 = pd.read_csv("train_2.csv")
df_2.to_csv("val_new.csv")