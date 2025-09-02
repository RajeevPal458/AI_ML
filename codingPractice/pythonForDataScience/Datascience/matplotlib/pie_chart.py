import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def curve2(x:int,y:int)-> int:
    return np.pow(y,2) + 4*x + 1

def curve(x:int,y:int)-> int:
    return np.sin(x) + np.cos(x)

categ =np.arange(1,21)
y = np.random.choice([1,2,3,4,5],size=20)

print(categ)
print(y)

plt.pie(y,labels=categ)
plt.title("random X line chart")

plt.show()



