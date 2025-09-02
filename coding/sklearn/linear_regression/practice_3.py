import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
x,y = make_regression(n_samples=200,n_features=3,noise=15,random_state=42)
df = pd.DataFrame(x,columns=['A','B','C'])
df['T'] = y
print(df)
size_feature = np.random.rand(200)*100

print("x shape: ",x.shape)
print("y shape: ",y.shape)

fig =px.scatter_3d(df,x=df['A'],y=df['B'],z=df['T'],color=df['C'],size=np.abs(df['C'])*100,opacity=0.7,title='4D interactive chart',size_max=20)
fig.update_layout(width=650,height=500,
                  margin=dict(l=0,r=0,t=0,b=10),
                  scene=dict(aspectmode='cube',
                             xaxis_title='feature1',
                             yaxis_title='feature2',
                             zaxis_title='Target(y)'))
fig.update_traces(marker=dict(size=5))
fig.show()



