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

n = MLP(784, [20, 20, 10])
print("MLP created...")
# print("number of parameters", len(n.parameters()))

# calculating loss
def findloss(predvals, train_y):
    err = 0
    s = len(predvals)
    for i, j in zip(predvals, train_y):
        #sm = sum([scorei.exp() for scorei in i])
        #err += sum([(yi - scorei.exp()/sm)**2 for yi, scorei in zip(j, i)])
        err += sum([(yi - scorei)**2
                     for yi, scorei in zip(j, i)])
        # still a value type
    #print(max(range(10), key=lambda k: predvals[0][k].data), max(range(10), key=lambda k: train_y[0][k].data))
    acc = [max(range(10), key=lambda k: i[k].data) == max(range(10), key=lambda k: j[k].data) for i,j in zip(predvals, train_y)]
    return err, sum(acc)/s

#training process:
def train(bsize, alpha, num, bcount):
    learningrate = alpha
    numofpasses = num
    batchsize = bsize
    
    for i in range(numofpasses):
        counter = 0
        print("Training", i+1, "of", numofpasses, "...")
        for batch in range(bcount):
            print("    Training batch", batch+1, "of", bcount, "...")
            # forward pass
            predvals = list(map(n, train_x[counter:counter+batchsize]))
            loss, acc = findloss(predvals, train_y[counter:counter+batchsize])
            # backwards pass
            for p in n.parameters():
                p.grad = 0.0
            loss.backwards()
            print("    Loss:", loss.data, "Acc:", round(100*acc,2), "%")
            # update
            learningrate = alpha - alpha * i/numofpasses
            for p in n.parameters():
                p.data -= learningrate * p.grad
            
            counter+=batchsize
            if batch%5 == 0:
                # checks accuracy
                test_pred = [max(range(len(i)), key=lambda j: i[j].data) for i in
                            map(n, test_x[:100])]
                # returns highest activated neuron in output layer ie: which digit
                suc = 0
                for i in range(100):
                    if test_pred[i] == test_lab[i]:
                        suc+=1
                print("Accuracy on test data:", suc, "%")
    test_pred = [max(range(len(i)), key=lambda j: i[j].data) for i in map(n, test_x[:1000])]
    # returns highest activated neuron in output layer ie: which digit
    suc = 0
    for i in range(1000):
        if test_pred[i] == test_lab[i]:
            suc+=1
    print("Accuracy on test data (1000):", suc/1000.0)
    # write weights to a file
    with open('weights.pkl', 'wb') as file:
        pickle.dump([i.data for i in n.parameters()], file)

train(100, 1, 10, 100)

# b=12
# s=8

# newpred = list(map(n, train_x[b:b+s]))
# print("Prediction:     ", *newpred, sep="\n")

# while True:
#     inp = input()
#     if inp=="q":
#         break
#     if inp=="b":
#         b+=s

#     loss, acc = findloss(newpred, train_y[b:b+s])
#     print(loss.data, acc)
#     for p in n.parameters():
#         p.grad = 0.0
#     loss.backwards()
#     learningrate = 0.1
#     for p in n.parameters():
#         p.data -= learningrate * p.grad
#     newpred = list(map(n, train_x[b:b+s]))
#     #print("New Prediction1:", *newpred, sep="\n")
