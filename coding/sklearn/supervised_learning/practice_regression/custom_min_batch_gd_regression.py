import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
import math
class MiniBatchGDRegressor:
    
    def __init__(self,lr=0.01,epochs=1000,bach_size=50):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.B = 0
        self.batch_size = bach_size
        self.losses = []
        
    
    def fit(self, x_train, y_train):
        if isinstance(x_train,pd.DataFrame):
            X = x_train.values
        if isinstance(y_train, pd.Series):
            y = y_train.values
            
        m,n = X.shape 
        baches = m/self.batch_size
        for epoch in range(self.epochs):
     
            for batch in range(math.floor(baches)):
                arr = np.arange(m)
                indices = np.random.choice(arr,self.batch_size, replace=False)
                
                Xb = X[indices]
                Yb = y[indices]
                print(f"Xb size - {Xb.shape}")
                self.W = np.zeros(n)
                yhat = np.dot(Xb , self.W) + self.B
                error = Yb - yhat
                
                dw = (-2/self.batch_size) * np.dot(Xb.T , error)
                db = (-2/self.batch_size) * np.sum(error)
                
                self.W -= self.lr * dw
                self.B -= self.lr * db
                
    def predict(self, x_test):
        
        if isinstance(x_test , pd.DataFrame):
            X = x_test.values
        return np.dot(X,self.W) + self.B
    

np.random.seed(42)
diabeties = load_diabetes()
df = pd.DataFrame(diabeties.data,columns=diabeties.feature_names)
df['targets'] = diabeties.target
print(df.head(5))
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
print(f" iloc Frame, X type: {type(X)}  y type: {type(y)}")

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model = SGDRegressor(
#     loss='squared_error',
#     penalty=None,
#     learning_rate='constant',
#     eta0=0.01,
#     max_iter=1000,
#     tol=None,
#     random_state=42
# )
row,col = x_test.shape
model = MiniBatchGDRegressor(lr=0.01, epochs=1000,bach_size=32)

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


            
            
            
        


