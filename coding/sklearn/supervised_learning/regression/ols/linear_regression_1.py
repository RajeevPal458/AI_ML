import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
df = pd.read_csv("C:\DriveD\projects\gitprojects\data_science\linear_data\study_vs_marks.csv")
df['Attendance_Hours'] = df['Attendance_Hours'].astype(int)
df['Final_Marks'] = df['Final_Marks'].astype(int)

df.drop_duplicates()
print(df.shape)

x = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(f"x shape: {x.shape}, y shape: {y.shape}")
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
model = LinearRegression()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)

mse = mean_squared_error(y_test,y_predict)
r2 = r2_score(y_test,y_predict)

print()
print(f"mse: {mse} , r2: {r2}")

print(f"cofficient m: {model.coef_} , interceptor c: {model.intercept_}")
#On average, your model’s predictions deviate from the actual values by √24.49 ≈ 4.95 units.
#Squared error is used to penalize large errors more than small ones.
'''
R² shows the proportion of variance in the target variable that is explained by the model.
It ranges from:
    1 = perfect prediction
    0 = model predicts no better than mean
    <0 = model is worse than predicting the mean

📌 Your value:
R² = 0.8077
The model explains ~80.77% of the variance in the data. That’s a very decent fit for most real-world problems.
'''

error_df = pd.DataFrame()
error_df['y_test'] = y_test
error_df['y_pred'] = y_predict

print(error_df)

plt.scatter(x_train,y_train,color='blue',label='train data')
plt.plot(x_train,model.predict(x_train),c='red',linestyle='-',linewidth=2)
plt.title("Scatter Plot with Regression Line")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.grid(False)
plt.show()








