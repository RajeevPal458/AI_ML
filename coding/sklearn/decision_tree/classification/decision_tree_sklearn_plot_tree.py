import pandas as pd
import numpy as np
from sklearn.datasets import load_iris,fetch_openml,load_breast_cancer
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,confusion_matrix
#from dtreeviz.trees import dtreeviz
import dtreeviz as dtreeviz
import matplotlib.pyplot as plt
from PIL import Image
import ssl


df = pd.read_csv("C:\\Users\\user\\Downloads\\AI_ML\\AI_ML\\testFiles\\weather.csv")

# iris = load_iris()
# df = pd.DataFrame(iris.data , columns=iris.feature_names)
# df['target'] = iris.target
# print(df)



X = df.iloc[:,0:3]
y = df.iloc[:,-1].replace(['Yes','No'],[1,0])
print(X.shape)
print(y.shape)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

tree_model = DecisionTreeClassifier(criterion='entropy',splitter='random',min_samples_split=20, min_samples_leaf=20, max_depth=3)
tree_model.fit(x_train,y_train)

pred = tree_model.predict(x_test)
print(f"pred shape {pred.shape}   y_test shap : {y_test.shape}")

resultFrame = pd.DataFrame({'actual': y_test,'pred':pred})
##print(resultFrame)

accuracyScore = accuracy_score(y_test,pred)
print(f"accuracyScore: {accuracyScore}")

precisionScore = precision_score(y_test,pred,average='macro')
print(f"precisionScore : {precisionScore}")

print("dtreeviz :{dtreeviz}")

print(f"x_train : {x_train.shape}  , y_train : {y_train.shape}  class names: {type(y)}")


#Get feature importance
importances = tree_model.feature_importances_
plt.figure(figsize=(10,6))

plot_tree(tree_model,feature_names=['MinTemp','MaxTemp','Rainfall'],class_names={0:'No',1:'Yes'},filled=True,rounded=True,fontsize=10)

plt.title("Decision tree breast cancel dataset")
plt.show()









