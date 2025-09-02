import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
# messy_df = pd.read_csv("C:/Users/user/Documents/learning_notes/self_learning/eda/study_marks_data.csv")
# messy_df = pd.read_csv("C:/Users/user/Documents/learning_notes/self_learning/eda/Student_marks.csv")
messy_df = pd.read_csv("C:/Users/user/Documents/learning_notes/self_learning/eda/data.csv")
print(messy_df.head(2))

print("==========messay shap==========")
print(messy_df.shape)

# zscore =np.abs((zscore(messy_df)))
# df = messy_df[(zscore<1).all(axis=1)]
# df['Attendance_Hours'] = df['Attendance_Hours'].astype(int)
# df['Final_Marks'] = df['Final_Marks'].astype(int)

dbscan = DBSCAN(eps=2,min_samples=10)
labels = dbscan.fit_predict(messy_df)
messy_df['cluster'] = labels
df = messy_df[messy_df["cluster"]!=-1]

x = df.iloc[:,[-6]]
y = df.iloc[:,[-2]]

print(df.head(2))

print("=================x=======================")
print(x.shape)
print("=================y=======================")
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)


print("=================x_train test shape=======================")
print(x_train.shape,x_test.shape)
print("=================y_train test_shape=======================")
print(y_train.shape,y_test.shape)

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)

y_predict = linear_model.predict(x_test)

print("=================prediction comparision=======================")
dframe = pd.DataFrame()
dframe['y_test'] = y_test
dframe['y_predict'] = y_predict
print(dframe)

print("=================slop and interceptor===================")
print(linear_model.coef_)
print(linear_model.intercept_)

print("=================performance analysis===================")
mse = mean_squared_error(y_test,y_predict)
r2Score = r2_score(y_test,y_predict)
rmse = root_mean_squared_error(y_test,y_predict)
print(f"mse: {mse} square root of mse is {np.sqrt(mse)}")
print(f"r2Score: {r2Score}")
print(f"rmse: {rmse}")

# design graph

#plt.figure()
plt.figure(figsize=(14,10))
plt.scatter(x,y,
            c='blue',        # Map colors based on species label
            s=50,           # Adjust marker size
            alpha=0.7,      # Set transparency
            linewidths=0,   # Remove border around markers (optional)
            marker='D')
plt.plot(x_test,y_predict,c='red')
plt.title("linear model for student study and marks")
plt.xlabel("study")
plt.ylabel("marks")
plt.legend()
plt.grid(True)
plt.draw()
