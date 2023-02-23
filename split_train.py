import pandas as pd

def upsample_insincere(df):
    sincere = df[df["target"] == 0].copy()
    insincere = df[df["target"] == 1].copy()
    upsampled = insincere.sample(len(sincere.index), axis=0, replace = True, ignore_index = True)
    return pd.concat([sincere, upsampled], axis=0, ignore_index = True)

# read DataFrame
data = pd.read_csv("/train.csv")

# no of csv files with row size
k = 2
size_train = 1044896
size_val = 261216 + size_train

print(data.shape)

df = data[0:size_train]
#sampled_df = upsample_insincere(df)
df.to_csv(f'train_80.csv', index=False)

df = data[size_train:size_val]
#ampled_df = upsample_insincere(df)
df.to_csv(f'train_val_20.csv', index=False)
