# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('flower_data.csv')  # Replace with your file name
X = data[['Size']]  # Feature: Flower size (use double brackets to keep it a DataFrame)
y = data['Price']   # Target: Flower price

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make a sample prediction
sample_size = [[8]]  # Replace with your input size
predicted_price = model.predict(sample_size)

# Output the result
print(f"Predicted price for a flower size of {sample_size[0][0]} cm is ${predicted_price[0]:.2f}.")
