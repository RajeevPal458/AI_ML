import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier


X,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaller = StandardScaler()
x_train_scalled = scaller.fit_transform(x_train)
x_test_scalled = scaller.transform(x_test)

model = CatBoostClassifier(iterations=100,learning_rate=0.01,depth=4,random_state=42,verbose=3)
model.fit(x_train_scalled,y_train)

y_pred = model.predict(x_test_scalled)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)

print(f"X shape : {X.shape}")
print(f"===accuracy score ====={accuracy}")
print(f"===precision score ====={precision}")



