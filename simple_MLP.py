# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 19:05:52 2018

@author: Christopher
"""

import numpy as np
import torch

dtype = torch.float
device = torch.device("cpu")



def load_XOR():
    X = []
    Y = []
    with open("XOR.txt") as f:
        for line in f:
            line = line.split(",")
            X.append([float(line[0]), float(line[1])])
            Y.append([float(line[2])])
    return X, Y

def load_iris():
    with open("iris_data.txt") as f:
        raw_data = [line.replace("\n", "").split(",") for line in f]
        data = []
        labels = []
        for line in raw_data:
            data.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
            if line[-1] == "Iris-setosa":
                labels.append([1,0,0])
            elif line[-1] == "Iris-versicolor":
                labels.append([0,1,0])
            else:
                labels.append([0,0,1])
    return data, labels

X, Y = load_iris()
X = torch.from_numpy(np.asarray(X))
Y = torch.from_numpy(np.asarray(Y))

x = torch.zeros(4,2)

print(Y)

N, D_in, H, D_out = 4, 4, 10, 3
"""
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)
"""
y = Y.float()

print(X,Y)

x, y = X.float(), Y.float()

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# testing
y_pred = x.mm(w1).clamp(min=0).mm(w2)
out_put = [[a, b] for a, b in zip(y, y_pred)]
print(out_put)
