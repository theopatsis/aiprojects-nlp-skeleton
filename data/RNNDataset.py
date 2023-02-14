import torch
import pandas as pd
import numpy as np

class StartingDataset(torch.utils.data.Dataset):
    """
    Bag of Words Dataset
    """

    # TODO: dataset constructor.
    def __init__(self, data_path):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''

        # Preprocess the data. These are just library function calls so it's here for you
        # self.df = pd.read_csv(data_path, nrows=1175489)
        self.df = pd.read_csv(data_path)
        self.labels = self.df.target.tolist() # list of labels
        print(self.sequences.shape)
        self.data_df = self.oversample(pd.DataFrame({"data": self.sequences.toarray().tolist(),
                        "target": self.labels}))
        self.sequences = np.stack(self.data_df["data"].values, axis = 0); self.labels = self.data_df["target"].tolist()
        print(self.sequences.shape)
        self.token2idx = self.vectorizer.vocabulary_ # dictionary converting words to their counts
        self.idx2token = {idx: token for token, idx in self.token2idx.items()} # same dictionary backwards

    def oversample(self, df):
        sincere = df[df["target"] == 0].copy()
        insincere = df[df["target"] == 1].copy()
        upsampled = insincere.sample(round(len(sincere.index)/1.8), axis=0, replace = True, ignore_index = True)
        return pd.concat([sincere, upsampled], axis=0, ignore_index = True)

    # TODO: return an instance from the dataset
    def __getitem__(self, i):
        '''
        i (int): the desired instance of the dataset
        '''
        # return the ith sample's list of word counts and label

        return self.sequences[i, :], self.labels[i]

    # TODO: return the size of the datasetfloat(
    def __len__(self):
        return self.sequences.shape[0]