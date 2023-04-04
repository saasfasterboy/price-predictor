# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('real_estate.csv') #You will need to replace 'real_estate.csv' with the name of your own dataset file.

# Split dataset into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model using training data
model.fit(X_train, y_train)

# Predict prices using testing data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
