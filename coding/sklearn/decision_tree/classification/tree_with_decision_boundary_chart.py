import numpy as np
import pandas as pd
from sklearn.datasets import make_classification,make_circles,make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


#X,y = make_classification(n_samples=20000,n_classes=2,n_redundant=0,n_clusters_per_class=1,class_sep=0.5,n_features=2,random_state=42,flip_y=0.05)
#X,y = make_circles(n_samples=20000,noise=0.1,factor=0.5,random_state=42)
X,y = make_moons(n_samples=20000,noise=0.1,random_state=42)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

tree_under = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_under.fit(x_train,y_train)

tree_balanced = DecisionTreeClassifier(max_depth=6, random_state=42)
tree_balanced.fit(x_train,y_train)

tree_overfitting = DecisionTreeClassifier(max_depth=None, random_state=42)
tree_overfitting.fit(x_train,y_train)

# plot decision boundry

def plot_decision_boundry(model,X,y,ax,title):
    x_min,x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min,y_max = X[:,1].min()-0.5, X[:,1].max()+0.5

    xx,yy = np.meshgrid(np.linspace(x_min,x_max,300),np.linspace(y_min,y_max,300))
    z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA','#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000','#00FF00'])
    
    ax.contourf(xx,yy,z,cmap=cmap_light,alpha=0.6)
    ax.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)
    ax.set_title(title)
    ax.set_xlabel("Features 1")
    ax.set_ylabel("Features 2")


# plot all Decision boundaries

# Plot Both
fig,axs = plt.subplots(1, 3, figsize=(18,6))
plot_decision_boundry(tree_under, x_test, y_test, axs[0],"tree underfitting")
plot_decision_boundry(tree_overfitting, x_test, y_test, axs[1],"tree overfitting")
plot_decision_boundry(tree_balanced, x_test, y_test, axs[2],"tree balanced")
plt.tight_layout()
plt.show()