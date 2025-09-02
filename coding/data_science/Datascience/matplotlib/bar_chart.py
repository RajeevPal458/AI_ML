import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def curve2(x:int,y:int)-> int:
    return np.pow(y,2) + 4*x + 1

def curve(x:int,y:int)-> int:
    return np.sin(x) + np.cos(x)

x = np.arange(1,10)
np.random.shuffle(x)
print(x)
y = [curve(x,x) for x in np.arange(1,10)]
x = np.arange(1,10)
print(y)

plt.bar(x, y, color='green', edgecolor='blue', linewidth=2)
plt.title("random X line chart")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()

y = [curve2(x,x) for x in np.arange(1,10)]
plt.bar(x,y)
plt.title("random X line chart")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()


