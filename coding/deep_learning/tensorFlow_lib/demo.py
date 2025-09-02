from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 2. Preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Create a DataFrame from X and y
df = pd.DataFrame(X, columns=housing.feature_names)
df['Target'] = y

# 3. Save to CSV file
#df.to_csv("california_housing.csv", index=False)

# 3. Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4. Train
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# 5. Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {mae:.4f}")



# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss over Epochs')
plt.show()


y_pred = model.predict(X_test).flatten()

# Scatter plot of predicted vs true
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  # diagonal line
plt.show()
