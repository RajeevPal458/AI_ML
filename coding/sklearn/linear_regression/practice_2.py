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

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111,projection='3d')
sc =ax.scatter(df['A'],df['B'],df['T'],c=df['C'],cmap='plasma',alpha=0.7)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('Target(Y)')
fig.colorbar(sc,label='fig 3 color')
plt.title('ploting random data')
plt.legend()
plt.grid(True)
plt.show()





