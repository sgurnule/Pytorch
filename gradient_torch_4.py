#import numpy as np

# 1) Design model (input size, outpur size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
# - forward pass : compute prediction
# - backward pass: gradients
# - Update weights

import torch
import torch.nn as nn

# f= w*x
# f=2*x

X= torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y= torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

X_test= torch.tensor([5],dtype=torch.float32)
n_samples,n_features = X.shape
print(n_samples,n_features)

input_size =n_features
output_size=n_features

model =nn.Linear(input_size,output_size)

print(f'Prediction before training:f(5)={model(X_test).item():.3f}')

#Training
learning_rate = 0.01
n_iters =100

loss=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),learning_rate)

for epoch in range(n_iters):
    y_val=model(X)
    l= loss(Y,y_val)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 ==0:
        [w,b]=model.parameters()
        print(f'epoch {epoch +1}: w={w[0][0].item():.3f},loss={l:.8f}')

print(f'Prediction after training:f(5)={model(X_test).item():.3f}')




