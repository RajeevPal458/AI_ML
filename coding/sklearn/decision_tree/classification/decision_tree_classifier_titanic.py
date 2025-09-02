import pandas as pd
import numpy as np
from sklearn.datasets import load_iris,fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,confusion_matrix
#from dtreeviz.trees import dtreeviz
import dtreeviz as dtreeviz
import matplotlib.pyplot as plt
from PIL import Image
import ssl
from sklearn.preprocessing import LabelEncoder


readcsv = pd.read_csv("C:\\Users\\user\\Downloads\\AI_ML\\AI_ML\\testFiles\\titanic.csv")

df = readcsv[['PassengerId','Survived','Pclass','Embarked']]
print(f"unique targets : {df['Embarked'].unique()}")
df = df.loc[df['Embarked'] != '\\N']
#df.rename(columns={'Embarked':'target'},inplace=True)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['Embarked'])
df.drop(columns=['Embarked'],inplace=True)

X = df.iloc[:,0:3].copy()
y = df.iloc[:,-1].copy()

print(X.shape)
print(y.shape)
print(f" unique targets : {y.unique()}")
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

tree_model = DecisionTreeClassifier(criterion='entropy',splitter='random',min_samples_split=20, min_samples_leaf=20, max_depth=3)
tree_model.fit(x_train,y_train)

pred = tree_model.predict(x_test)
print(f"pred shape {pred.shape}   y_test shap : {y_test.shape}")

resultFrame = pd.DataFrame({'actual': y_test,'pred':pred})
#print(resultFrame)

accuracyScore = accuracy_score(y_test,pred)
print(f"accuracyScore: {accuracyScore}")

precisionScore = precision_score(y_test,pred,average='macro')
print(f"precisionScore : {precisionScore}")

print("dtreeviz :{dtreeviz}")

print(f"x_train : {x_train.shape}  , y_train : {y_train.shape}  class names: {type(y)}")


#Get feature importance
importances = tree_model.feature_importances_
plt.figure(figsize=(10,6))
plt.barh(range(len(importances)),importances,align='center')
plt.yticks(range(len(importances)),['PassengerId','Survived','Pclass'])
plt.show()

viz = dtreeviz.model(model=tree_model,X_train=x_train,y_train=y_train,target_name='target',feature_names=['PassengerId','Survived','Pclass'],class_names={0:'C',1:'S',2:'Q'})
viz.view()











