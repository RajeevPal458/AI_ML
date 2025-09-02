import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import root_mean_squared_error,r2_score
import matplotlib.pyplot as plt


# def gen_pollynomial_data(sample_size:int=200,noise:int=20,featues:int=2)-> pd.DataFrame:
#     np.random.seed(42)
#     features  = np.random.rand(sample_size,featues)
#     df1 = pd.DataFrame(features,columns=["Feature1","Feature2"])
#     x1 = df1['Feature1']
#     x2 = df1['Feature2']
#     targets = [10 +4*x1[i]**2 + 4*(x2[i]**2)+ noise  for i in range(sample_size)]
#     df2 = pd.DataFrame(targets,columns=["target"])

#     df = pd.concat([df1,df2],axis=1)
#     return df

def gen_pollynomial_data(sample_size:int=200,featues:int=2)-> pd.DataFrame:
    np.random.seed(42)
    x1 = np.random.uniform(-3,3,sample_size)
    x2 = np.random.uniform(-3,3,sample_size)
    noise = np.random.normal(0,5,sample_size)
    #targets = 10 +4*x1**2 + 4*(x2**2)+ 2*x1 + 3*x2 + 6*x1*x2 + noise 
    targets = 10 +4*x1**2 + 4*(x2**2) + noise 
    df = pd.DataFrame({'Feature1':x1,'Feature2':x2,'target':targets})   #columns=["Feature1","Feature2","target"]
    return df

df:pd.DataFrame = gen_pollynomial_data()
X = df.iloc[:,0:-1]
Y = df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print(f"x_train: {type(x_train)} y_train: {type(y_train)}")
#plt.scatter(x_train['Feature1'],y_train)
#plt.scatter(x_train['Feature2'],y_train)


#Polynomial Features

poly = PolynomialFeatures(degree=4,include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly  = poly.transform(x_test)

x_train_poly_T  = poly.transform(x_train)

linear_model = LinearRegression()
linear_model.fit(x_train_poly,y_train)
pred = linear_model.predict(x_test_poly)

pred2 = linear_model.predict(x_train_poly_T)
rmse = root_mean_squared_error(pred,y_test)
r2score = r2_score(pred,y_test)
print(f"rmse : {rmse}  , r2 score : {r2score}")

scaller = StandardScaler()
##create a grid for surface
f1_range = x_test['Feature1']
f2_range = x_test['Feature2']
f1_mess,f2_mess = np.meshgrid(f1_range,f2_range)
x1x2_combined = np.c_[f1_mess.ravel(),f2_mess.ravel()]
x1x2_poly = poly.transform(x1x2_combined)
y_pred_grid = linear_model.predict(x1x2_poly).reshape(f1_mess.shape)
fig = go.Figure()
print("add actual ponts")
fig = fig.add_trace(go.Scatter3d(x=x_train['Feature1'],
                                 y=x_train['Feature2'],
                                 z = y_train,
                   mode='markers',
                   marker=dict(size=5,color='red'),
                   name='Actual'))

# print("add predicted points")
# fig = fig.add_trace(go.Scatter3d(x=x_test['Feature1'],
#                                  y=x_test['Feature2'],
#                                  z=pred,
#                    mode='markers',
#                    marker=dict(size=5,color='green'),
#                    name='Predicted'))

print("add predicted points")
fig = fig.add_trace(go.Scatter3d(x=x_train['Feature1'],
                                 y=x_train['Feature2'],
                                 z=pred2.ravel(),
                   mode='markers',
                   marker=dict(size=5,color='green'),
                   name='Predicted'))

print("add hyperplane surface")
fig = fig.add_trace(go.Surface(x=f1_mess,y=f2_mess,z=y_pred_grid,
                   opacity=0.7,
                   colorscale='Viridis',
                   showscale=False,
                   name='Regression Plane'))


print("add layout")
fig.update_layout(title='3D scatter with regression hyperplane',
                  width=650,
                  height=500,
                  margin=dict(l=0,r=0,t=0,b=10),
                  scene=dict(aspectmode='cube',
                             xaxis_title='feature1',
                             yaxis_title='feature2',
                             zaxis_title='Target(y)'),
                             )

# import plotly.io as pio
# pio.renderers.default='browser'
fig.show()








