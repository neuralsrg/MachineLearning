import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, sampler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import sys
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('runs')

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

# ====================================================
# =
# = Making a Tensorboard image grid
# =
# =
# ====================================================

img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
# writer.close()

# sys.exit()


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

# ====================================================
# =
# = Making a Tensorboard graph
# =
# =
# ====================================================

writer.add_graph(model, samples.reshape(-1, 28*28))

# training loop 

total_samples = len(train_loader)

running_loss = 0
running_correct = 0

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

        _, predicted = torch.max(outputs, 1)
        running_loss += loss.item()
        running_correct += (predicted == labels).sum().item()

        # info 
        if (i+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Step {i+1}/{total_samples} | Loss = {loss.item():.4f}')
            writer.add_scalar('training_loss', running_loss / 100., global_step=epoch * total_samples + i)
            writer.add_scalar('accuracy', running_correct / 100., global_step=epoch * total_samples + i)

            running_loss = 0
            running_correct = 0


# testing and
# ====================================================
# =
# = Making a Precision-Recall Curves
# =
# =
# ====================================================

preds = []
values = []

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

        class_predictions = [F.softmax(output, dim=0) for output in outputs] # Array of (10,) tensors

        preds.append(class_predictions) # Array of arrays, where each array is batch predictions
        values.append(labels)


    preds = torch.cat([torch.stack(batch) for batch in preds])
    # torch.stack --> concatenates sequence of tensors along a new dimension
    # torch.cat --> concatenates the given sequence of seq tensors in the given dimension
    labels = torch.cat(values)

    classes = range(10)
    for i in classes:
        labels_i = labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    writer.close()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}%')