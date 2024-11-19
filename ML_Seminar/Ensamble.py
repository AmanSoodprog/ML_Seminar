# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('flower_data.csv')  # Replace with your file name
X = data[['Size']]  # Feature: Flower size
y = data['Price']   # Target: Flower price

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=10, random_state=42)  # 10 trees for simplicity
model.fit(X, y)

# Make a sample prediction
sample_size = [[8]]  # Replace with your input size
predicted_price = model.predict(sample_size)

# Output the result
print(f"Predicted price for a flower size of {sample_size[0][0]} cm is ${predicted_price[0]:.2f}.")
