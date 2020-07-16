import numpy as np

# f= w*x
# f=2*x

X= np.array([1,2,3,4],dtype=np.float32)
Y=np.array([2,4,6,8],dtype=np.float32)

w=0.0

#Model Prediction

def forward(x):
    return w*x

# loss = MSE

def loss (y_pred,y):
    return ((y_pred - y)**2).mean()

# gradient
# MSE = 1/N * (w*x -y)**2
# dJ/dw = 1/N 2x(w*x -y)

def gradient(x,y,y_pred):
    return np.dot(2*x,y_pred-y).mean()

print(f'Prediction before training:f(5)={forward(5):.3f}')

#Training
learning_rate = 0.01
n_iters =10

for epoch in range(n_iters):
    y_val=forward(X)

    l= loss(Y,y_val)

    dw=gradient(X,Y,y_val)

    w -=learning_rate * dw

    if epoch % 1 ==0:
        print(f'epoch {epoch +1}: w={w:.3f},loss={l:.8f}')

print(f'Prediction after training:f(5)={forward(5):.3f}')




