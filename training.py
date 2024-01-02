import numpy as np
from nn import MLP
from engine import Value
import pickle

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
train_x = []
train_y = []
for i in range(len(train_img)):
    train_x.append([item for sublist in train_img[i] for item in sublist])
    train_y.append([Value(1) if j==train_lab[i] else Value(-1) for j in range(10)])
test_x = []
test_y = []
for i in range(len(test_img)):
    test_x.append([item for sublist in test_img[i] for item in sublist])
    test_y.append([Value(1) if j==test_lab[i] else Value(-1) for j in range(10)])
print("Data setup...")

# setup with two hidden layers of size 80 each
n = MLP(784, [20, 10, 10])
print("MLP created...")

# calculating loss
def findloss(predvals, train_y):
    losses = []
    for i in range(len(predvals)):
        err = 0
        for j in range(10):
            err += (predvals[i][j]-train_y[i][j])**2
            # still a value type
        losses.append(err)
    loss = sum(losses)
    return loss

#training process:

learningrate = 0.02
numofpasses = 10
batchsize = 20
counter = 0
for batch in range(1):
    print("Training batch", batch+1, "of 100...")
    for i in range(numofpasses):
        # forward pass
        predvals = [n(x) for x in train_x[counter:counter+batchsize]]
        loss = findloss(predvals, train_y)
        # backwards pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backwards()
        print("    Loss:", loss.data)
        print("    Training", i+1, "of", numofpasses, "...")
        # update
        for p in n.parameters():
            p.data+= - learningrate * p.grad
        
    counter+=batchsize
    if batch%5 == 0:
        # checks accuracy
        test_pred = [max(range(len(i)), key=lambda j: i[j].data) for i in [n(x) for x in test_x[:50]]]
        # returns highest activated neuron in output layer ie: which digit
        # test predictions, of first 30

        # check success rate on test data
        suc = 0
        for i in range(50):
            if test_pred[i] == test_lab[i]:
                suc+=1
        print("Success rate on test data:", suc/50.0)

# write weights to a file
with open('weights.pkl', 'wb') as file:
    pickle.dump([i.data for i in n.parameters()], file)