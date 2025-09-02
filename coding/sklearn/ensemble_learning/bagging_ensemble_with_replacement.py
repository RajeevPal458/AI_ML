import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# iris = load_iris()

# df = pd.DataFrame(iris.data, columns = iris.feature_names)
# df['target'] = iris.target

# print(df.head(5))


X,y = make_circles(n_samples=2000,factor=0.5,noise=0.20,random_state=42)
print(f"==========X============type: {type(X)}")
print(X)

print(f"==========Y=============type: {type(y)}")
print(y)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

base_tree = DecisionTreeClassifier(min_samples_leaf=2, random_state=42)
base_tree.fit(x_train,y_train)

#Bagging ensemble with 50 trees
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),bootstrap=True,verbose=True,max_samples=0.08,n_estimators=50,oob_score=True,random_state=42)
bagging_model.fit(x_train,y_train)


def evaluate_model(name,model,x_test,y_test):
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:,1]
    print(f"----------{name}----------------")
    print(f"accuracy score: {accuracy_score(y_test,y_pred)}")
    print(f"precision score: {precision_score(y_test,y_pred)}")
    print(f"recall score: {recall_score(y_test,y_pred)}")
    print(f"f1 score: {f1_score(y_test,y_pred)}")
    print(f"f1 score: {roc_auc_score(y_test,y_pred)}")
    print()


evaluate_model("Single Decision tree",base_tree,x_test,y_test)
evaluate_model("bagging ensemble model",bagging_model,x_test,y_test)


def plot_decision_boundry(model,X,y,ax,title):
    x_min,x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min,y_max = X[:,1].min()-0.5, X[:,1].max()+0.5

    xx,yy = np.meshgrid(np.linspace(x_min,x_max,300),np.linspace(x_min,x_max,300))
    z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA','#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000','#00FF00'])
    
    ax.contourf(xx,yy,z,cmap=cmap_light,alpha=0.6)
    ax.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)
    ax.set_title(title)
    ax.set_xlabel("Features 1")
    ax.set_ylabel("Features 2")

# Plot Both
fig,axs = plt.subplots(1, 2, figsize=(14,10))
plot_decision_boundry(base_tree, x_test, y_test, axs[0],"Single Dicision Tree")
plot_decision_boundry(bagging_model, x_test, y_test, axs[1],"Bagging Ensambple (50 trees)")
plt.tight_layout()
plt.show()