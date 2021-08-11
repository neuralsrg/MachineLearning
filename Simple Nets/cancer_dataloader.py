import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
class CancerDataset(Dataset):

    def __init__(self):
        bc = datasets.load_breast_cancer()
        self.x_data = torch.from_numpy(sc.fit_transform(bc.data).astype(np.float32))
        self.y_data = torch.from_numpy(bc.target.astype(np.float32))
        self.y_data = self.y_data.view(self.y_data.shape[0], 1)

        self.n_samples = self.x_data.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.n_samples


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


dataset = CancerDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)

num_epochs = 100
learning_rate = 0.1

n_samples = len(dataset)
n_features = dataset.x_data.shape[1]
num_batches = math.ceil(n_samples / 4)

model = Model(n_features)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        if (i + 1) % 50 == 0:
            total_prediction = model(dataset.x_data)
            total_loss = criterion(total_prediction, dataset.y_data)
            print(f'Epoch: {epoch + 1}/{num_epochs}, Step {i + 1}/{num_batches} '
                  f'| Loss {total_loss}')