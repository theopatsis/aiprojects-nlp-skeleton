import torch
import torch.nn as nn

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(108, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1) #this is basically identical
        #self.fc3 = nn.Linear(10, 2)
        self.sigmoid = nn.Sigmoid() #reduntant to BCELogitsLoss

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = x.type(torch.float32)
        # print(x)
        # print(x.shape)
        x = self.fc1(x)
        x = torch.relu(x) #need activation function to model nonlinear relations, sigmoid is cringe
        x = self.fc2(x)
        x = torch.relu(x) 
        x = self.fc3(x)
        x = self.sigmoid(x)
        # print(x.shape)
        return x


