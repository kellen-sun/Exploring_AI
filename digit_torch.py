import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt 


# Load the .npz file
mnist_data = np.load('mnist.npz')
# 60,000 training data points
# 10,000 testing data
# each [x] is a 28x28 array in grayscale
# each [y] is the corresponding number
print("Data loaded...")
def vis(data):
    # given the matrix format data of a mnist image print it out
    for i in data:
        for j in i:
            if j>0:
                print("% ", end="")
            else:
                print("  ", end="")
        print("")

# setup training and testing data
train_img = mnist_data["x_train"]
train_lab = mnist_data["y_train"]
test_img = mnist_data["x_test"]
test_lab = mnist_data["y_test"]
training_data = list(zip(train_img, train_lab))
test_data = list(zip(test_img, test_lab))

print("Data setup...")

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_loader = torch.utils.data.DataLoader(training_data ,batch_size = batch_size_train, shuffle = True)

test_loader = torch.utils.data.DataLoader(test_data , batch_size = batch_size_test, shuffle = True)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

plt.imshow(example_data[12], cmap="gray")
plt.show()
