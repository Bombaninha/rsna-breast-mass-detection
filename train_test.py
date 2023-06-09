import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

# Define the transformation to apply to the images
def get_transform(image_size, num_channels):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if num_channels == 1:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    return transform

def get_dataset(image_size, num_channels):
    transform = get_transform(image_size, num_channels)
    train_dataset = datasets.ImageFolder(root='./output/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='./output/val', transform=transform)
    test_dataset = datasets.ImageFolder(root='./output/test', transform=transform)
    return train_dataset, test_dataset, val_dataset

# Define the data loader
def get_data_loader(image_size, num_channels, batch_size, device):
    train_dataset, test_dataset, val_dataset = get_dataset(image_size, num_channels)
    kwargs = {'pin_memory': True} if device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, train_dataset, test_dataset, val_dataset, val_loader

# Define the model architecture
class CNN(nn.Module):
    def __init__(self, image_size, num_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #self.pool2 = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2))

        self.fc1 = nn.Linear(32 * (image_size[0] // 8) * (image_size[1] // 8), 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, num_classes, bias=True)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.1, inplace=False)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == "__main__":

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = os.listdir('output/train') # get the class names automatically using the folder names
    
    img_size = (1024, 1024) # 454x678 , 544x814 , 512x512 , 1024x1024
    n_channels = 1 # 1
    n_classes = 3 # 3
    n_batches = 8 # 8
    accumulation_steps = 4 # 4

    torch.cuda.empty_cache()
    # Initialize the model and move it to the GPU if available
    model = CNN(img_size, n_channels, n_classes).to(device) 

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.8, weight_decay=0.75) # lr=0.001 | 0.003 | weight_decay = 0.9 | 0.8

    train_loader, test_loader, train_dataset, test_dataset, val_dataset, val_loader = get_data_loader(img_size, n_channels, n_batches, device)
    
    # Train the model
    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for batch_id, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(True):
                #optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                #optimizer.step()

                if ((batch_id + 1) % accumulation_steps == 0) or (batch_id + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            #torch.cuda.empty_cache()

        train_loss = train_loss / len(train_dataset)
        train_accuracy = train_correct / len(train_dataset)
        
        torch.cuda.empty_cache()

        predicted_lst = []
        labels_lst = []
        
        # Evaluate the model on the validation set
        val_loss = 0.0
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                labels_lst.append(labels)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

                predicted_lst.append(predicted)
                #torch.cuda.empty_cache()

        val_loss = val_loss / len(val_dataset)
        val_accuracy = val_correct / len(val_dataset)
        
        # Print the training and test loss and accuracy for each epoch
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Valid. Loss: {:.4f}, Valid. Acc: {:.4f}'
            .format(epoch+1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy))
        
        torch.cuda.empty_cache()

        if epoch == num_epochs-1: # record results from the last epoch tests
            fp = open('data/cnn_predicted' + str(epoch+1) + '.txt', 'w') 
            for i in range(len(predicted_lst)):
                pred_labels, class_labels = zip(*[(class_names[pl], class_names[cl]) for pl, cl in zip(predicted_lst[i].tolist(), labels_lst[i].tolist())])
                #class_labels = [class_names[j] for j in labels_lst[i].tolist()]
                
                for k in range(len(class_labels)):
                    fp.write('Epoch [{}/{}] - Batch {}: predicted {} was class {}\n'.format( 
                            epoch+1, num_epochs, i, pred_labels[k], class_labels[k]))

    
    torch.cuda.empty_cache()

    predicted_lst = []
    labels_lst = []
    # Evaluate the model on the test set
    test_loss = 0.0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            labels_lst.append(labels)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()

            predicted_lst.append(predicted)
            #torch.cuda.empty_cache()

    test_loss = test_loss / len(test_dataset)
    test_accuracy = test_correct / len(test_dataset)

    # Print the train loss and accuracy
    print('Final Model, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(test_loss, test_accuracy))

    fp = open('data/cnn_predicted_test.txt', 'w') 
    for i in range(len(predicted_lst)):
        pred_labels, class_labels = zip(*[(class_names[pl], class_names[cl]) for pl, cl in zip(predicted_lst[i].tolist(), labels_lst[i].tolist())])
        #class_labels = [class_names[j] for j in labels_lst[i].tolist()]
        
        for k in range(len(class_labels)):
            fp.write('Final Model - Batch {}: predicted {} was class {}\n'.format(i, pred_labels[k], class_labels[k]))