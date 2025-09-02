import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

iris = load_iris()
iris_df = pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df['target'] = iris.target

X,y = load_iris(return_X_y=True)

print(y)

rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X,y)

importances = rf.feature_importances_

print(f"==importances==: {importances}")

feature_importances_df = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': importances
}).sort_values(by='importance',ascending=False)

print(f" ====feature_importances_df =====:{feature_importances_df}")

top_two_features = feature_importances_df['feature'].iloc[:2].values

print(f"top two features : {top_two_features}")
print()

# df with top two features only
df = iris_df[top_two_features]
df['target'] = iris_df['target']

X,y = df.iloc[:,:2].values,df.iloc[:,-1]

print(f"X type {type(X)}  x shap: {X.shape} , y type: {type(y)} , y shape: {y.shape}")


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# standarise the data
scaller = StandardScaler()
x_train_scalled = scaller.fit_transform(x_train)
x_test_scalled = scaller.transform(x_test)


# GridSerarchCv for Hyperparameter tuning

grid_params = {
    'n_estimators':[10,50,100],
    'max_depth':[2,3,5,None],
    'min_samples_split':[2,4,6],
    'criterion':['gini','entropy']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42,min_samples_leaf=2),grid_params,cv=5,n_jobs=-1)
grid_search.fit(x_train_scalled,y_train)
print(f"Best parameters from grid search cv : {grid_search.best_params_}")

best_model = grid_search.best_estimator_

# evaluate model

y_pred = best_model.predict(x_test_scalled)

print(f"accuracy : {accuracy_score(y_test,y_pred)}")

# plot decision boundaries

def plot_decision_boundry(model,X,y,ax,title):
    h = .02
    x_min,x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min,y_max = X[:,1].min()-0.5, X[:,1].max()+0.5

    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA','#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000','#00FF00'])
    
    plt.figure(figsize=(8,6))
    plt.contourf(xx,yy,z,alpha=0.4)
    plt.scatter(X[:,0],X[:,1],c=y*300,cmap=plt.cm.Set1,edgecolor='k',s=20)
    plt.title(title)
    plt.xlabel("Features 1")
    plt.ylabel("Features 2")
    plt.tight_layout()
    plt.show()

# Plot Both

plot_decision_boundry(best_model, x_test_scalled, y_test, axs,"RandomForest Ensamble")

