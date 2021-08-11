import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, sampler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

# device configuration 
# we use Cuda on GPU if possible 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters 

input_size = 784 # images are 28x28
hidden_size = 100 # can be different 
num_classes = 10 # 10 different digits are available 
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# check random sample 

examples = iter(train_loader)
samples, label = examples.next()

# print(samples.shape, label.shape)

# Show some pictures 

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()

# Neural Network 

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out) # We don't apply softmax() here because we will use Cross Entropy below 
        return out


model = NeuralNet(input_size, hidden_size, num_classes)

# loss & optimizer 

criterion = nn.CrossEntropyLoss() # applying softmax()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop 

total_samples = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # (100, 1, 28, 28) --> (100, 784)
        images = images.reshape(-1, 28 * 28)

        # moving to GPU if possible 
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # info 
        if (i+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Step {i+1}/{total_samples} | Loss = {loss.item():.4f}')


# testing 

with torch.no_grad():
    n_correct = 0
    n_samples = 0 

    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1) # 1 = along rows
        n_correct += (predictions == labels).sum().item()
        n_samples += labels.shape[0]

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}%')

# SAVING model

FILE = 'mnist_model.pth'
torch.save(model.state_dict(), FILE)
