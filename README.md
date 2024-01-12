# 💻 Exploring AI 

As the name of this repository suggests, this is a collection of small projects where I explore and learn about AI. I aim to complete simple projects with additional depth for an educational purpose.

## ✍️ MNIST

MNIST is a well-known dataset of handwritten digits which I've included in the ``mnist.npz`` file. It is often used to train machine learning models to recognize handwriting.

The dataset consists of 60,000 training images and 10,000 testing images each labelled with a number from 0 to 9 representing what the handwritten digit is. Each image consists of 28x28 (=784) pixels and has a brightness indicated from 0 to 255. The images are grayscale and contain relatively little amount of information. 

I trained a dense fully-connected network with 2 hidden layers of 16 nodes each using the ``tanh`` activation function ``softmax`` at the last layer and ``cross-entropy`` to calculate my loss. Then I used the gradient descent algorithm to tweak my weights.

### Code Walkthrough:

I first setup my training data and labels correctly from the ``.npz`` file. After reading from it, I make the data into a numpy array of 784 elements instead of a grid of 28x28. Then use the np.eye() function to reshape my labels into a list. Specifically, I want the label ``3`` to have the form ``[0,0,0,1,0,0,0,0,0,0]`` this way I can compare it with the output of my network. 

In the next portion, I create random values for my weights and biases which act between layers of the network.

Then, I created 4 functions that would calculate information about the network given inputs and assuming current weights and biases. I used ``numpy`` arrays to speed up this process. Importantly, I returned the activation of all layers as this would be necessary when calculating losses through the gradient descent algorithm. Softmax and cross-entropy are frequently used in classification neural networks, where softmax will translate outputs of an activation layer into probabilities (from 0 to 1) with the added key observation that the sum of the probabilities must be 1. Then cross-entropy, will calculate the loss by taking the correct label and our predicted probability on it and assigning it a loss of -log(p) where p is the probability we assigned. Doing a forward pass consists of matrix multiplication between weights and previous activations then adding the biases ``tanh(Wa+b)``.

At each point of weights, biases and activations, I keep track of their gradients and reset them to 0 before training. Then I manually calculated each derivative and applied the chain rule to find the respective gradients. Nudging those gradients by a small factor (denoted alpha) during training minimizes the loss function and brings my output closer to the intendended label.

### Result

I was able to train a model with an accuracy in the mid-90s after some tweaking of my variables and could easily reach high 80% in accuracy after < 5min of training on a laptop.

## 📁 File Search

In this project, I used a sentence transformer to map a short string (such as file or directory name) into a 384 dimensional vector that represents the meaning of the word. This allows for a file search system based on semantic search which matches the query to strings of similar meanings.