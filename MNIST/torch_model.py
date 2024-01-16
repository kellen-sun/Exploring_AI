import torch
import torch.nn as nn
import numpy as np

mnist_data = np.load('MNIST\\mnist.npz')
print("Data loaded...")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30, 10), nn.Softmax(dim=1),)

    def forward(self, idx, targets=None):
        idx = torch.tensor(idx, dtype=torch.float32)
        idx = self.net(idx)
        return idx

def test_acc():
    max_values, max_indices = torch.max(model(xc), dim=1)
    max_values, max_indiceslab = torch.max(yc, dim=1)
    count = [1 if t==s else 0 for t, s in zip(max_indices, max_indiceslab)]
    print(sum(count)/len(count))

model = Model()

train_img = mnist_data["x_train"]
train_lab = mnist_data["y_train"]
test_img = mnist_data["x_test"]
test_lab = mnist_data["y_test"]
print("Data setup...")
xb = train_img.reshape(len(train_img), -1)/255
yb = torch.tensor(np.eye(10)[train_lab], dtype=torch.float32)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
criterion = nn.CrossEntropyLoss()
xc = torch.tensor(test_img.reshape(len(test_img), -1)/255, dtype=torch.float32)
yc = torch.tensor(np.eye(10)[test_lab], dtype=torch.float32)

for i in range(5):
    for steps in range(1000):
        xbb, ybb = xb[50*steps:50*steps+50], yb[50*steps:50*steps+50]
        outputs = model(xbb)
        loss = criterion(outputs, ybb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    test_acc()