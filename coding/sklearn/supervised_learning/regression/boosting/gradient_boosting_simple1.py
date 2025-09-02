import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt


# data = {
#     "Experience":[2,3,5,6],
#     "Degree": ["Graduate","Masters","Masters","PHD"],
#     "Salary":[50,70,80,100]
# }

data = {
    "Experience":[2,3,5,6],
    "Degree": [13,14,14,15],
    "Salary":[50,70,80,100]
}

df = pd.DataFrame(data)
print(df)

df['pred1'] = df['Salary'].mean()
print(df)

df['res1'] = df['Salary'] - df['pred1']



X,y = df[["Experience","Degree"]],df["res1"]
print("==========",X)
model = DecisionTreeRegressor(max_leaf_nodes=3)
model.fit(X,y)
df['leaf_ids'] = model.apply(X)
pred = model.predict(X)
# model 2 prediction encluding model one 
print("pred====",pred)
df['pred2'] = df['pred1'] + pred

df['res2'] = df['Salary'] - df['pred2']
print()
print(df)



plt.figure(figsize=(12,8))
plot_tree(model,feature_names=X.columns, filled=True)
# To print the leaf IDs:
print("Leaf IDs:", [i for i in range(model.tree_.node_count)
                    if model.tree_.children_left[i] == -1])
plt.show()
