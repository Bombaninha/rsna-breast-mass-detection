import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
import splitfolders

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

splitfolders.ratio('results', output='output', seed=1337, ratio=(0.8, 0, 0.2)) 
# data_dir = '/results'
# classes = os.listdir(data_dir)
# image_paths = []
# labels = []

# # collect paths and labels for all images
# for i, class_name in enumerate(classes):
#     class_path = os.path.join(data_dir, class_name)
#     for image_name in os.listdir(class_path):
#         image_path = os.path.join(class_path, image_name)
#         image_paths.append(image_path)
#         labels.append(i)

# # split data into train and test sets
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_indices, test_indices = next(sss.split(image_paths, labels))
# train_paths = [image_paths[i] for i in train_indices]
# train_labels = [labels[i] for i in train_indices]
# test_paths = [image_paths[i] for i in test_indices]
# test_labels = [labels[i] for i in test_indices]

# # Load the dataset
# train_dataset = datasets.ImageFolder(root='data/train', transform=transforms.ToTensor())
# test_dataset = datasets.ImageFolder(root='data/test', transform=transforms.ToTensor())

# # Define the dataloaders
# batch_size = 64
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the transformation to apply to the images
def get_transform(image_size, num_channels):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    if num_channels == 1:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    return transform

def get_dataset(image_size, num_channels):
    transform = get_transform(image_size, num_channels)
    train_dataset = datasets.ImageFolder(root='./output/train', transform=transform)
    test_dataset = datasets.ImageFolder(root='./output/test', transform=transform)
    return train_dataset, test_dataset

# Define the data loader
def get_data_loader(image_size, num_channels, batch_size):
    train_dataset, test_dataset = get_dataset(image_size, num_channels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

# Define the model architecture
class CNN(nn.Module):
    def __init__(self, image_size, num_channels):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(128 * (image_size[0] // 8) * (image_size[1] // 8), 512)
        self.fc2 = nn.Linear(512, 8)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        #x = x.view(-1, 128 * (x.shape[2] // 8) * (x.shape[3] // 8))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the loss function and optimizer
def get_model(image_size, num_channels):
    model = CNN(image_size, num_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return model, criterion, optimizer


# print('device' + str(device))
# exit()
# Initialize the model and move it to the GPU if available
model = CNN((544, 814), 1).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader, test_loader, train_dataset, test_dataset = get_data_loader((544, 814), 1, 16)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
    train_loss = train_loss / len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)
    
    # Evaluate the model on the test set
    test_loss = 0.0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(test_dataset)
    test_accuracy = test_correct / len(test_dataset)
    
    # Print the training and test loss and accuracy for each epoch
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, train_accuracy, test_loss, test_accuracy))
