# STOCHASTIC GRADIENT DESCENT
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

# Stochastic
class SGDMultipleLinearRegressor:
    
    def __init__(self,lr=0.01,epochs=1000):
        self.W =None
        self.B =0
        self.lr = lr
        self.epochs = epochs
        self.losses = []
        
    def fit(self,x_train,y_train):
        if isinstance(x_train , pd.DataFrame):
            X = x_train.values
        if isinstance(y_train, pd.Series):
            y = y_train.values
        
        m,n = X.shape
        self.W = np.zeros(n)

        for epoch in range(self.epochs):
            np.random.seed(42)
            indices = np.arange(m)
            np.random.shuffle(indices)
            
            for row in indices:
                # yhat = W*X+B 
                rxrow = X[row]
                ryrow = y[row]
                
                yhat = np.dot(rxrow,self.W) + self.B
                error = ryrow - yhat
                
                dw = -2 * rxrow * error
                db = -2 * error
                
                self.W = self.W - self.lr * dw
                self.B = self.B - self.lr * db
                 
            # # Loss for current epoch (MSE)
            # y_pred_all = np.dot(X, self.W) + self.B
            # loss = (1 / m) * np.sum((y - y_pred_all) ** 2)
            # self.losses.append(loss)

            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | W: {self.W[:3]}... | B: {self.B:.4f}")
                
                
                
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.dot(X, self.W) + self.B
        

np.random.seed(42)
diabeties = load_diabetes()
df = pd.DataFrame(diabeties.data,columns=diabeties.feature_names)
df['targets'] = diabeties.target
print(df.head(5))
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
print(f" iloc Frame, X type: {type(X)}  y type: {type(y)}")

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = SGDRegressor(
    loss='squared_error',
    penalty=None,
    learning_rate='constant',
    eta0=0.01,
    max_iter=1000,
    tol=None,
    random_state=42
)

#model = SGDMultipleLinearRegressor(lr=0.01, epochs=1000)

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


