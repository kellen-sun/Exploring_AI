import random, math
from engine import Value

class Neuron():
    def __init__(self, nin, nonlin = True):
        # nin is number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
    
    def __call__(self, x):
        # where x is an array of inputs
        # takes dot product of x with weights then add bias b
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return activation.tanh() if self.nonlin else activation

    def parameters(self):
        # returns list of all its parameters
        # aka changeable values (weights and bias)
        return self.w+[self.b]
    
class Layer():
    # a layer of neutrons
    def __init__(self, nin, nout, nonlin = True):
        # nin is number of inputs
        # nout is number of outputs
        self.neurons = [Neuron(nin, nonlin= nonlin) for _ in range(nout)]

    def __call__(self, x):
        # call the whole layer given the activation of previous layer x (an array)
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    
    def parameters(self):
        # returns lists of all parameters of each neuron in layer
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP():
    # multilayer perceptron
    def __init__(self, nin, nouts, softmax):
        #nin is size of input lists
        #nouts is a list of the size of each output layer
        sz = [nin] + nouts
        self.sm = softmax
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.sm:
            sp = sum(i.exp() for i in x)
            y = [i.exp()/sp for i in x]
            # print(x)
            return y
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    