import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepare Data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1,noise=20,random_state=1)

X=torch.from_numpy(X_numpy.astype(np.float32))
y=torch.from_numpy(Y_numpy.astype(np.float32))
y=y.view(y.shape[0],1)

n_samples,n_features =X.shape

# Model
input_size= n_features
output_size=1

model = nn.Linear(input_size,output_size)

#Loss and Optimizer
learning_rate = 0.01
criteria = nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)

#Training loop

num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    y_predicted = model(X)
    loss=criteria(y_predicted,y)

    #Backward Pass
    loss.backward()

    #Update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch:{epoch+1},loss={loss.item():.4f}')
#plot
predicted= model(X).detach().numpy()

plt.plot(X_numpy,Y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()


