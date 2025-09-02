import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def curve2(x:int,y:int)-> int:
    return np.pow(y,2) + 4*x + 1

def curve(x:int,y:int)-> int:
    return np.sin(x) + np.cos(x)


y = np.random.choice([20,15,5,1,11],size=20)
x = np.arange(1,21)

print(x)
print(y)

plt.scatter(x, y, c='red', s=100,
            marker='D', alpha=0.5
)
plt.title("random X line chart")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()



