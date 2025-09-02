import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,confusion_matrix
#from dtreeviz.trees import dtreeviz
import dtreeviz as dtreeviz
import matplotlib.pyplot as plt
from PIL import Image

iris = load_iris()

df = pd.DataFrame(iris.data , columns=iris.feature_names)
df['target'] = iris.target
print(df)

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

tree_model = DecisionTreeClassifier(max_leaf_nodes=50, max_depth=2)
tree_model.fit(x_train,y_train)

pred = tree_model.predict(x_test)
print(f"pred shape {pred.shape}   y_test shap : {y_test.shape}")

resultFrame = pd.DataFrame({'actual': y_test,'pred':pred})
print(resultFrame)

accuracyScore = accuracy_score(y_test,pred)
print(f"accuracyScore: {accuracyScore}")

precisionScore = precision_score(y_test,pred,average='macro')
print(f"precisionScore : {precisionScore}")

print("dtreeviz :{dtreeviz}")

print(f"x_train : {x_train.shape}  , y_train : {y_train.shape}  class names: {list(iris.target)}")


#Get feature importance
importances = tree_model.feature_importances_
plt.figure(figsize=(10,6))
plt.barh(range(len(importances)),importances,align='center')
plt.yticks(range(len(importances)),iris.feature_names)
plt.show()

viz = dtreeviz.model(model=tree_model,X_train=x_train,y_train=y_train,target_name='target',feature_names=iris.feature_names,class_names={0:'setosa',1:'versicolor',2:'verginika'})
viz.view()











