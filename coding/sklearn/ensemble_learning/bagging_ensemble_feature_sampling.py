import numpy as np
import pandas as pd
from sklearn.datasets import make_circles,make_classification
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


X,y = make_classification(n_samples=1000,n_features=50,n_informative=10,n_redundant=10,n_repeated=0,n_classes=2,random_state=42)
print(f"==========X============type: {type(X)}")
print(X)

print(f"==========Y=============type: {type(y)}")
print(y)

# for ploting lets extract forst two features
x_vis = X[:,:2]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

x_vis_train = x_train[:,:2]
x_vis_test = x_test[:,:2]


base_tree = DecisionTreeClassifier(min_samples_leaf=10, random_state=42)
base_tree.fit(x_train,y_train)

#Bagging classifiers with features sampling
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=10,random_state=42),n_estimators=50,bootstrap=False,
                                   bootstrap_features=True, # featuers bagging enabled
                                   max_samples=0.8, # 80 percent of samples per estimator
                                   max_features=0.2, # 20 percent of features per estimators
                                   random_state=42)
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
    x_min,x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min,y_max = X[:,1].min()-1, X[:,1].max()+1

    xx,yy = np.meshgrid(np.linspace(x_min,x_max,300),np.linspace(y_min,y_max,300))
    
    # for visualization use only first two features
    gridPoints = np.c_[xx.ravel(),yy.ravel()]

    #Pad rest of the 48 featuers with means
    n_features = x_train.shape[1]
    mean_values = x_train.mean(axis=0)
    gridFull = np.tile(mean_values, (gridPoints.shape[0],1))
    gridFull[:,:2] = gridPoints # replace first two features

    z = model.predict(gridFull)
    z = z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA','#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000','#00FF00'])
    
    ax.contourf(xx,yy,z,cmap=cmap_light,alpha=0.4)
    ax.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=10,alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Features 1")
    ax.set_ylabel("Features 2")

# Plot Both
fig,axs = plt.subplots(figsize=(10,7))
plot_decision_boundry(bagging_model, x_vis_test, y_test, axs,"Bagging Ensambple (50 trees)")
plt.tight_layout()
plt.show()