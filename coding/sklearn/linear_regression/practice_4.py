import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error,root_mean_squared_log_error

x,y = make_regression(n_samples=200,n_features=3,noise=15,random_state=42)

x_df = pd.DataFrame(x,columns=['A','B','C'])
y_df = pd.DataFrame(y,columns=['T'])

df = pd.concat([x_df,y_df],axis=1)

print(df)
print(f"shap of X is: {x.shape} and for Y is {y.shape}")

X = df[['A','B','C']]
Y = df['T']

x_train,x_test,y_train,y_test = train_test_split(x_df,Y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
f1 = x_test['A']
f2 = x_test['B']
# f1_range = np.linspace(f1.min(),f1.max(),40)
# f2_range = np.linspace(f2.min(),f2.max(),40)
f1_range = f1
f2_range = f2
f1_grid,f2_grid = np.meshgrid(f1_range,f2_range)

f3_mean = x_test['C'].mean()

grid_input = np.c_[f1_grid.ravel(),f2_grid.ravel(),np.full_like(f1_grid.ravel(),f3_mean)]

z_pred = model.predict(grid_input)
print(f"grid input shap is: {grid_input.shape} z_pred: {z_pred.shape}")
z_pred = z_pred.reshape(f1_grid.shape)
print(f"z_prid shape : {z_pred.shape}, f1_grid shap: {f1_grid.shape} , f2_grid shap: {f2_grid.shape}, f1_range shap: {f1_range.shape}, x_train shap :{x_train.shape} , x_test shap: {x_test.shape}")

fig = go.Figure()
print("add actual ponts")
fig = fig.add_trace(go.Scatter3d(x=x_test['A'],
                                 y=x_test['B'],
                                 z=y_test,
                   mode='markers',
                   marker=dict(size=5,color=np.abs(x_test['C']*200)),
                   name='Actual'))

print("add predicted points")
fig = fig.add_trace(go.Scatter3d(x=x_test['A'],
                                 y=x_test['B'],
                                 z=pred,
                   mode='markers',
                   marker=dict(size=5,color=np.abs(x_test['C']*200)),
                   name='Predicted'))

print("add hyperplane surface")
fig = fig.add_trace(go.Surface(x=f1_grid,y=f2_grid,z=z_pred,
                   opacity=0.6,
                   colorscale='viridis',
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

mae = mean_absolute_error(y_test,pred)
print("mean absolute error :",mae)

mse = mean_squared_error(y_test,pred)
print("mean square error :",mse)

rmse = root_mean_squared_error(y_test,pred)
print("root mean square error :",rmse)

rmsle = root_mean_squared_log_error(np.abs(y_test),np.abs(pred))
print("mean square log error :",rmsle)



