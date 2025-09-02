import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as xp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error,mean_squared_error,mean_absolute_error,r2_score
from sklearn.datasets import load_diabetes

class BatchGdcRegressor:

    def __init__(self,lr:int=0.001,epochs:int=1000):
        self.lr = lr
        self.epoachs = epochs
        self.W = None
        self.B = 0
        self.losses = []

    def fit(self,X,y):
        if isinstance(X,pd.DataFrame):
            X = X.values
        if isinstance(X,pd.Series):
            y = y.values
        
        m,n = X.shape
        self.W = np.zeros(n)
        self.B = 0
        print(f"X shap: {X.shape} , W shap :{self.W.shape} , y shap :{y.size}")
        print(f"X shap: {type(X)} , W shap :{type(self.W)} , y shap :{type(y)}")
        flag:bool = True
        for epoch in range(self.epoachs):
            y_pred = np.dot(X,self.W) + self.B
            if flag:
                print(f"y_pred shap: {y_pred.shape}")
            error = y-y_pred
            
            # compute gradient over full batch
            dw = (- 2 / m) * np.dot(X.T , error)
            db = (- 2 / m) * np.sum(error)

            # update weight and bias
            self.W -= self.lr*dw
            self.B -= self.lr*db

            # compute loss mean square error
            loss =( 1 / m ) * np.sum(error**2)
            self.losses.append(loss)
            flag=False

    def predict(self,X):

        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.dot(X,self.W) +self.B

    

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
X = df.drop('target',axis=1)
y = df['target']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = BatchGdcRegressor(0.001,1000)
model.fit(x_train,y_train)

pred = model.predict(x_test)

mse = mean_squared_error(y_test,pred)
print(f"===MSE :{mse}")



