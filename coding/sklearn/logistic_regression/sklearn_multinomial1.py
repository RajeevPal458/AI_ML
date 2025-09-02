import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score,roc_curve,confusion_matrix,ConfusionMatrixDisplay,classification_report,auc


iris = load_iris()

df = pd.DataFrame(iris.data , columns=iris.feature_names)
df['target'] = iris.target

#print(df)

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# LogisticRegression(
#  penalty='l2', *,
#  dual=False,
#  tol=0.0001,
#  C=1.0,
#  fit_intercept=True,
#  intercept_scaling=1,
#  class_weight=None,
#  random_state=None,
#  solver='lbfgs',
#  max_iter=100,
#  multi_class='deprecated',
#  verbose=0,
#  warm_start=False,
#  n_jobs=None,
#  l1_ratio=None)

model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)

print(f'--------pred : {pred}')

coef = model.coef_
print(f'---------coef : {coef}')
intercept = model.intercept_
print(f'intercept : {intercept}')

classes = model.classes_
print(f"classes : {classes}")

accuracy = accuracy_score(y_test, pred)
print(f"accuracy : {accuracy}")

precisionScore = precision_score(y_test, pred,average='macro')
print(f"precisionScore : {precisionScore}")

recall = recall_score(y_test, pred,average='macro')
print(f"recall : {recall}")

f1score = f1_score(y_test, pred,average='macro')
print(f" f1score: {f1score}")

matrix = confusion_matrix(y_test, pred)
print(f"confusion matrix : {matrix}")

matrix = confusion_matrix(y_test, pred)
print(f"confusion matrix : {matrix}")

report = classification_report(y_test,pred)
print(f"report : -{report}")

#fpr, tpr, threshold = roc_curve(y_test,pred)
#print(f"fpr : {fpr} , tpr: {tpr} , threshold {threshold}")

#auc_prc = auc(fpr,tpr)
#print(f" auc_prc : {auc_prc}")

matrix_diplay = ConfusionMatrixDisplay(confusion_matrix=matrix)
matrix_diplay.plot()
# matrix_diplay.from_predictions(y_test,pred,cmap='viridis',colorbar='11')






















