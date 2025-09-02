import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score


X,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = AdaBoostClassifier(estimator= DecisionTreeClassifier(random_state=42),n_estimators=100,learning_rate=0.01,random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)

print(f"X shape : {X.shape}")
print(f"===accuracy score ====={accuracy}")
print(f"===precision score ====={precision}")



