import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

class BatchGDRegressor:
    def __init__(self,lr=0.01,epochs=1000):
        self.epochs = epochs
        self.W = None
        self.B =0
        self.lr = lr
        self.losses = []
    
    def fit(self,x_train,y_train):
        if isinstance(x_train,pd.DataFrame):
            X = x_train.values
        if isinstance(y_train,pd.Series):
            y = y_train.values
            
        m,n = X.shape
        self.W =np.zeros(n)
        self.B = 0
        print(f" xtrain shap: {x_train.shape} , xtrain type: {type(x_train)} ytrain shap {y_train.shape} type: {type(y_train)}")
        print(f" X type: {type(X)}  y type: {type(y)}")
        print(f" X size: {X.size}, {X[0].size}  y size: {y.size} , {y[0].size}")
        flag= True
        for epoch in range(self.epochs):
            dot_prod = np.dot(X,self.W)
            yHat = dot_prod + self.B
            error = y - yHat
            dw = -(2/m) * np.dot(X.T,error)
            db = -(2/m) * np.sum(error)
            
            self.W -= self.lr * dw
            self.B -= self.lr * db
            
            if flag:
                print(f" dot prod shap : {dot_prod.shape} , yhat shap :{yHat.shape} , dw shap :{dw.shape} ")
                flag= False
            
            loss = (1/m) * np.sum(error**2)
            self.losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | W: {self.W[:3]}... | B: {self.B:.4f}")
    def predict(self,xtest):
        if isinstance(xtest,pd.DataFrame):
            X = xtest.values
        ypred = np.dot(X, self.W)+self.B
        return ypred
    
    
diabeties = load_diabetes()

df = pd.DataFrame(diabeties.data,columns=diabeties.feature_names)
df['targets'] = diabeties.target
print(df.head(5))
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
print(f" iloc Frame, X type: {type(X)}  y type: {type(y)}")

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = BatchGDRegressor(0.01,1000)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
print(f"rmse : ",np.sqrt(mse))

r2score = r2_score(y_test,y_pred)
print(f"r2 score : ",r2score)

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred,c='red',cmap='viridis')
plt.plot([y_test.min(),y_test.max()],[y_pred.min(),y_pred.max()])
plt.title('Demo Model for diabetic patient')
plt.xlabel('test data')
plt.ylabel('predictions')
plt.show()