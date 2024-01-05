import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt 


# Load the .npz file
mnist_data = np.load('mnist.npz')
# 60,000 training data points
# 10,000 testing data
print("Data loaded...")

# setup training and testing data
train_img = np.float32(mnist_data["x_train"]/255)
train_lab = mnist_data["y_train"]
test_img = np.float32(mnist_data["x_test"]/255)
test_lab = mnist_data["y_test"]
images_tensor = torch.from_numpy(train_img).float()  # Assuming images are float32
labels_tensor = torch.from_numpy(train_lab).long()  # Assuming labels are integers
training_data = torch.utils.data.TensorDataset(images_tensor, labels_tensor)

images_tensor = torch.from_numpy(test_img).float()  # Assuming images are float32
labels_tensor = torch.from_numpy(test_lab).long()  # Assuming labels are integers
test_data = torch.utils.data.TensorDataset(images_tensor, labels_tensor)

print("Data setup...")

n_epochs = 3
batch_size_train = 1
batch_size_test = 1
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_loader = torch.utils.data.DataLoader(training_data, batch_size = batch_size_train, shuffle = True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size_test, shuffle = True)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
#plt.imshow(example_data[12], cmap="gray")
#plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=100,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
                groups=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        print(x.size())
        x = x.view( -1)
        print(x.size())
        output = self.out(x)
        return output, x
cnn = CNN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
num_epochs = 10

def train(num_epochs, cnn):
    cnn.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            bx = Variable(images)
            by = Variable(labels)
            output = cnn(bx)[0]
            print(by)
            print(output)
            loss = loss_func(output, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

train(n_epochs, cnn)