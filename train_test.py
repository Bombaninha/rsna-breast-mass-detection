import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import splitfolders

in_device = input('Choose a device to run on (cuda or cpu): ')

if(in_device == 'cuda'):
    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.backends.cudnn.benchmark = True
    # torch.set_default_tensor_type('torch.cuda.HalfTensor')
else:
    device = torch.device('cpu')

splitfolders.ratio('results', output='output', seed=1337, ratio=(0.8, 0, 0.2)) 

#class_names = ['1', '2', '3', '4a', '4b', '4c', '5', '6']
class_names = ['0', '1', '2']

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
    #kwargs = {'num_workers': 1, 'pin_memory': True} if in_device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))
    return train_loader, test_loader, train_dataset, test_dataset

# Define the model architecture
class CNN(nn.Module):
    def __init__(self, image_size, num_channels):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        #self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(128 * (image_size[0] // 8) * (image_size[1] // 8), 512)
        self.fc2 = nn.Linear(512, 3)
        #self.fc3 = nn.Linear(512, 3)
        #self.fc2 = nn.Linear(512, 8)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # x = self.relu(self.conv4(x))
        # x = self.pool(x)
        # x = self.relu(self.conv5(x))
        # x = self.pool(x)
        #x = x.view(-1, 128 * (x.shape[2] // 8) * (x.shape[3] // 8))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    in_device = input('Choose a device to run on (cuda or cpu): ')

    # Initialize the model and move it to the GPU if available
    model = CNN((454, 678), 1).to(device)
    #model = CNN((544, 814), 1).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 0.001

    train_loader, test_loader, train_dataset, test_dataset = get_data_loader((454, 678), 1, 8)
    #train_loader, test_loader, train_dataset, test_dataset = get_data_loader((544, 814), 1, 8)

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

            loss = loss.to('cpu')
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.to('cpu')
            labels = labels.to('cpu')
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_dataset)
        train_accuracy = train_correct / len(train_dataset)
        
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
                loss = loss.to('cpu')
                test_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to('cpu')
                labels = labels.to('cpu')
                test_correct += (predicted == labels).sum().item()

                predicted_lst.append(predicted)

        test_loss = test_loss / len(test_dataset)
        test_accuracy = test_correct / len(test_dataset)
        
        # Print the training and test loss and accuracy for each epoch
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
            .format(epoch+1, num_epochs, train_loss, train_accuracy, test_loss, test_accuracy))
        
        torch.cuda.empty_cache()

        if epoch == num_epochs-1: # record results from the last epoch tests
            fp = open('data/cnn_predicted' + str(epoch+1) + '.txt', 'w') 
            for i in range(len(predicted_lst)):
                pred_labels, class_labels = zip(*[(class_names[pl], class_names[cl]) for pl, cl in zip(predicted_lst[i].tolist(), labels_lst[i].tolist())])
                #class_labels = [class_names[j] for j in labels_lst[i].tolist()]
                
                for k in range(len(class_labels)):
                    fp.write('Epoch [{}/{}] - Batch {}: predicted {} was class {}\n'.format( 
                            epoch+1, num_epochs, i, pred_labels[k], class_labels[k]))
