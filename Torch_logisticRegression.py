import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc=datasets.load_breast_cancer()
X,y=bc.data,bc.target

n_samples, n_features =X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train= torch.from_numpy(X_train.astype(np.float32))
X_test= torch.from_numpy(X_test.astype(np.float32))
y_train= torch.from_numpy(y_train.astype(np.float32))
y_test= torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)

# Model

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.Linear =nn.Linear(n_input_features,1)

    def forward(self,x):
        y_predicted =torch.sigmoid(self.Linear(x))
        return y_predicted
model=LogisticRegression(n_features)

#loss and optimizer
learning_rate =0.01
criterion= nn.BCELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)

# Trainig loops
num_epochs =100
for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss=criterion(y_predicted,y_train)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if(epoch+1) %10 == 0:
        print(f'epoch:{epoch+1},loss ={loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc=y_predicted_cls.eq(y_test).sum()/float(y_test[0])
    print(f'accuracy ={acc:.4f}')






