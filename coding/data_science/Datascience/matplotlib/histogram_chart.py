import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.choice([3,7,11,14,1],size=50)

plt.hist(x, bins=25, color='green', edgecolor='blue',
         linestyle='--', alpha=0.5
)
plt.title("histogram chart")
plt.xlabel("x axis range ")
plt.ylabel("frequency ranges")
plt.show()

