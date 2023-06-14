import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the stock price data
data = pd.read_csv('stock_data.csv')  # Replace 'stock_data.csv' with your dataset file

# Extract the relevant features and target variable
X = data[['Feature1', 'Feature2', ...]]  # Replace 'Feature1', 'Feature2', ... with your features
y = data['Target']  # Replace 'Target' with your target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict stock prices for the next n days
n = 10  # Number of days for prediction
X_future = ...  # Replace ... with the feature values for the next n days
predicted_prices = model.predict(X_future)

# Create a dataframe for the predicted prices
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=n+1, freq='D')[1:]
predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})

# Visualize the actual and predicted prices
plt.plot(data['Date'], data['Price'], label='Actual Price')
plt.plot(predicted_df['Date'], predicted_df['Predicted Price'], label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.show()
