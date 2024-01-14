# 💻 Exploring AI 

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/kellen-sun/Exploring_AI.svg)
![Language](https://img.shields.io/badge/python.svg)


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

I was able to train a model with an accuracy in the mid-90s after some tweaking of my variables and could easily reach high 80% in accuracy after < 5min of training on a laptop (without GPU).

## 📁 File Search

In this project, I used a pre-trained sentence transformer to map a short string (such as file or directory name) into a 384 dimensional vector that represents the meaning of the word. This allows for a file search system based on semantic search which matches the query to strings of similar meanings.

Given a query and a starting directory, I used ``os.walk`` to obtain all the subdirectories and files they contain recursively. Then, I used generated the embeddings for each filename and directory name. Finally, I went through all of them and found the one that matched the query the closest.

The search algorithm could also be made faster by traversing the file structure as a graph. This uses the underlying assumption that folders which are grouped together likely have similar meanings. This would result in an approximate ``O(logn)`` time result which is a big step up from a brute force search of ``O(n)``, but might not always find the best item.

## 💻 NanoGPT

In this project, I learned about modern transformers by following Andrej Karpathy's YT tutorial, ["Let's build GPT: from scratch, in code, spelled out."](https://youtu.be/kCc8FmEb1nY)

I created a model with nearly 2 million parameters which implemented self-attention on a bigram language model. I trained on 40,000 lines of Shakespeare's existing work (``input.txt``) with the goal of generating similar text. The final output generated can be found in ``output.txt``.

This model has a character level generation. The model consists of 4 layers of 6 attention heads using a context window of 192 characters and 192 dimensions for embeddings. I trained it on a local cpu achieving a final loss of 1.4898. Each attention layer was fully connected with the next using a feed-forward layer with ReLU and dropout to prevent overfitting. Additionally, the model used an optimization by connecting previous layers directly to the end and from the source. This allows for faster training along the main branch which may not always need the head layers.

I learned about the transformer model, embeddings, normalizing variance (to prevent softmax from being too skewed) and layer normalization.
