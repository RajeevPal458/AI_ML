import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score,root_mean_squared_error
class MultipleLinearRegression:

    class Model:

        def __init__(self):
            self.W = 0.0
            self.B = 0.0

        def fit(self,X,y):
            # fit the model using least square formula
            # w = covarience(x,y)/varriance(x)
            # B = mean(y) - W*mean(x)

            X = np.array(X)
            y = np.array(y)

            x_mean = np.mean(X)
            y_mean = np.mean(y)

            numerator = np.sum((X - x_mean)*(y - y_mean))
            denomerator = np.sum((X - x_mean)**2)
            self.W = numerator/denomerator
            self.B = y_mean - self.W * x_mean

        def predict(self,X):
            X = np.array(X)
            return self.W * X + self.B
        
    
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    X = df.iloc[:,0]
    y = df.iloc[:,-1]

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # use with sklearn model
    XX = pd.DataFrame(X)
    x_train,x_test,y_train,y_test = train_test_split(XX,y,test_size=0.2,random_state=42)

    print(f"x_train :{x_train} , x_test :{y_train} ")
    print(f"x_train :{type(x_train)} , x_test :{type(y_train)} ")

    #model = Model()
    model = LinearRegression()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)

    print(f" predict {pred} actual {y_test}")

    print(f"x_train size {x_train.size}  x_test size {x_test.size}")

    x_train = x_train[0:x_test.size]
    plt.scatter(x_train,pred,c='red')
    plt.scatter(x_train,y_test,c='green')

    r2_score = r2_score(y_test,pred)
    rmse = root_mean_squared_error(y_test,pred)

    print(f" r2score : {r2_score} rmse: {rmse}") 


        
