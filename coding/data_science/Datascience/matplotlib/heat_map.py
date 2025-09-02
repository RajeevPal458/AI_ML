import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

np.random.seed(0)
#np.random.default_rng()   # used for reproducibility
data = np.random.rand(10,10)
plt.imshow(data,cmap='viridis',interpolation='nearest')
plt.colorbar()
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Heatmap')
plt.show()

plt.imshow(data,cmap='magma',interpolation='nearest')
plt.colorbar()
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Heatmap')
plt.show()

plt.imshow(data,cmap='plasma',interpolation='nearest')
plt.colorbar()
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Heatmap')
plt.show()





#=====================================


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'Math': np.random.randint(50, 100, size=20),
    'Physics': np.random.randint(40, 90, size=20),
    'Chemistry': np.random.randint(45, 95, size=20),
    'English': np.random.randint(60, 85, size=20)
})

# Compute correlation matrix
corr_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Correlation Heatmap of Subjects")
plt.show()


#==============================
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [1, 2, np.nan, 4],
    "B": [np.nan, 2, 3, 4],
    "C": [1, np.nan, 3, 4]
})
sns.heatmap(df.isnull(), cmap='Reds', cbar=False)
plt.title("Missing Value Heatmap")
plt.show()
