
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('traffic_data.csv')
data['Day'] = range(len(data))
X = data[['Day']]
y = data['TrafficVolume']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, predictions, color='red', label='Predicted')
plt.title('Traffic Volume Prediction')
plt.xlabel('Day')
plt.ylabel('Traffic Volume')
plt.legend()
plt.tight_layout()
plt.savefig("traffic_prediction_runtime.png")
