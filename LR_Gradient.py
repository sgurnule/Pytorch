#import numpy as np
import torch

# f= w*x
# f=2*x

X= torch.tensor([1,2,3,4],dtype=torch.float32)
Y= torch.tensor([2,4,6,8],dtype=torch.float32)

w=torch.tensor(0.0,dtype=torch.float32, requires_grad=True)

#Model Prediction

def forward(x):
    return w*x

# loss = MSE

def loss (y_pred,y):
    return ((y_pred - y)**2).mean()

# # gradient
# # MSE = 1/N * (w*x -y)**2
# # dJ/dw = 1/N 2x(w*x -y)
#
# def gradient(x,y,y_pred):
#     return np.dot(2*x,y_pred-y).mean()



print(f'Prediction before training:f(5)={forward(5):.3f}')

#Training
learning_rate = 0.01
n_iters =100

for epoch in range(n_iters):
    y_val=forward(X)

    l= loss(Y,y_val)

    l.backward()
    with torch.no_grad():
       w -=learning_rate * w.grad

    w.grad.zero_()

    if epoch % 10 ==0:
       print(f'epoch {epoch +1}: w={w:.3f},loss={l:.8f}')

print(f'Prediction after training:f(5)={forward(5):.3f}')




