import numpy as np
from math import log, exp

# Load the .npz file
mnist_data = np.load('mnist.npz')
# 60,000 training data [28x28 pixels]
# 10,000 testing data
print("Data loaded...")

def data_setup():
    # setup training and testing data
    train_img = mnist_data["x_train"]
    train_lab = mnist_data["y_train"]
    test_img = mnist_data["x_test"]
    test_lab = mnist_data["y_test"]
    print("Data setup...")
    #return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
    return train_img.reshape(len(train_img), -1)/255, np.eye(10)[train_lab], test_img.reshape(len(test_img), -1)/255, np.eye(10)[test_lab]

# parameters to optimize
W1 = np.random.uniform(-1, 1, size=(16, 784))
W2 = np.random.uniform(-1, 1, size=(16, 16))
W3 = np.random.uniform(-1, 1, size=(10, 16))
B1 = np.random.uniform(-1, 1, size=(16, ))
B2 = np.random.uniform(-1, 1, size=(16, ))
B3 = np.random.uniform(-1, 1, size=(10, ))

def softmax(X):
    s = sum(np.exp(X))
    return np.exp(X)/s

def forward(A0):
    global W1, W2, W3, B1, B2, B3
    A1 = np.tanh(np.dot(W1, A0)+B1)
    A2 = np.tanh(np.dot(W2, A1)+B2)
    A3 = np.tanh(np.dot(W3, A2)+B3)
    P = softmax(A3)
    return P, A3, A2, A1

def cross_entropy_loss(P, GT):
    # output and ground truth
    return sum(-yi*log(pi) for yi, pi in zip(GT, P))

def accuracy(Ps, GTs):
    acc = [max(range(10), key=lambda k: i[k]) == max(range(10), key=lambda k: j[k]) for i,j in zip(Ps, GTs)]
    return sum(acc)/len(acc)

def total_loss(Ps, GTs):
    acc = accuracy(Ps, GTs)
    loss = sum([cross_entropy_loss(Pi, GTi) for Pi, GTi in zip(Ps, GTs)])
    return loss, acc

def set_zero():
    return np.zeros((16, 784)), np.zeros((16, 16)), np.zeros((10, 16)), np.zeros((16,)), np.zeros((16, )), np.zeros((10,)), np.zeros((10,)), np.zeros((10,)), np.zeros((16, )), np.zeros((16,))

def backward(Ps, A3s, A2s, A1s, GTs, A0s):
    dW1, dW2, dW3, dB1, dB2, dB3, dP, dA3, dA2, dA1 = set_zero()
    global W1, W2, W3
    n = len(GTs) # batch size
    dP += sum([(-GTs[x])/(Ps[x]) for x in range(n)])
    #dP += sum([(Ps[x]-GTs[x]) for x in range(n)])
    dA3 += sum([Ps[x]*(dP - (np.dot(dP, Ps[x]))) for x in range(n)])
    #print(A3s)
    dB3 += sum([(1- (A3s[x]*A3s[x]))*dA3 for x in range(n)])
    #print(dA3, dB3, GTs, sep = "\n")
    W3T = W3.T
    dA2 += sum([np.array([np.dot(W3T[i], dB3) for i in range(16)]) for x in range(n)])
    dW3 += sum([np.array([A2s[x][i]*dB3 for i in range(16)]).T for x in range(n)])
    
    dB2 += sum([(1-A2s[x]*A2s[x])*dA2 for x in range(n)])
    W2T = W2.T
    dA1 += sum([np.array([np.dot(W2T[i], dB2) for i in range(16)]) for x in range(n)])
    dW2 += sum([np.array([A1s[x][i]*dB2 for i in range(16)]).T for x in range(n)])
    dB1 += sum([(1-A1s[x]*A1s[x])*dA1 for x in range(n)])
    dW1 += sum([np.array([A0s[x][i]*dB1 for i in range(784)]).T for x in range(n)])
    return dW1, dW2, dW3, dB1, dB2, dB3
    
def train(batchsize, batchcount, alpha, epochs, view, expo = True):
    global train_x, train_y
    global W1, W2, W3, B1, B2, B3
    test()
    for i in range(epochs):
        count = 0
        # check exponential learning rate decay or step
        if expo: a = alpha*exp(-i*0.3)
        else: a = alpha * (1- i/epochs)
        for j in range(batchcount):
            P, A3, A2, A1 = zip(*list(map(forward, train_x[count:count+batchsize])))
            if j%view == view - 1:
                test()
            dW1, dW2, dW3, dB1, dB2, dB3 = backward(P, A3, A2, A1, train_y[count:count+batchsize], train_x[count:count+batchsize])
            W1 -= dW1*a
            W2 -= dW2*a
            W3 -= dW3*a
            B1 -= dB1*a
            B2 -= dB2*a
            B3 -= dB3*a
            count += batchsize
        test()

def test():
    global test_x, test_y
    global W1, W2, W3, B1, B2, B3
    P, A3, A2, A1 = zip(*list(map(forward, test_x)))
    print(total_loss(P, test_y)[1])

train_x, train_y, test_x, test_y = data_setup()
train(1, 60000, 0.05, 12, 150000)
test()