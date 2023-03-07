import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the FNN model
class FNNModel(nn.Module):
    def __init__(self, inp, hidden1, hidden2, out):
        super(FNNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(inp, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.l2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.out = nn.Linear(hidden2, out)

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x, dim=1)
        return x

# Define the training and testing datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# Define the data loaders for batching
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the FNN model
model = FNNModel(784, 64, 32, 10)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define the number of training epochs
num_epochs = 10

# Lists to store the training and testing losses
train_losses = []
test_losses = []

# Training loop
for epoch in range(num_epochs):
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update the training loss
        train_loss += loss.item()

    # Update the average training loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Compute the testing loss
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # Update the average testing loss for the epoch
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Print the training and testing losses for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

# Plot the training and testing loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Curves')
plt.legend()
plt.show()
