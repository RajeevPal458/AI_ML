import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Sample datasets
normal_data = np.random.normal(loc=50, scale=10, size=100)  # normal distribution
skewed_data = np.random.exponential(scale=20, size=100)     # right-skewed
outlier_data = np.append(np.random.normal(30, 5, 95), [100, 110, 120, 130, 150])  # some large outliers
constant_data = np.full(100, 42)  # same value repeated
small_sample = np.random.normal(55, 2, size=5)  # very small sample

# print(normal_data)
# print("=============")
# print(skewed_data)
# print("=============")
# print(outlier_data)
# print("=============")
# print(constant_data)
# print("=============")
# print(small_sample)



# Combine data for plotting
data = [normal_data, skewed_data, outlier_data, constant_data, small_sample]

# Box plot
plt.figure(figsize=(12, 6))
plt.boxplot(data, patch_artist=True, labels=[
    "Normal", "Skewed", "Outliers", "Constant", "Small Sample"
])
plt.title("Box Plot Examples for Data Preprocessing")
plt.ylabel("Value Range")
plt.grid(True)
plt.show()
