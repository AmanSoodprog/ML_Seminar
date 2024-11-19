import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('flower_data.csv')  # Replace with your file name
sizes = data['Size'].values  # Flower sizes (states)
prices = data['Price'].values  # Flower prices (rewards)

# Initialize Q-values (same length as sizes)
q_values = np.zeros(len(sizes))

# Learning Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
episodes = 500  # Reduced episodes for simplicity

# Update Q-values based on rewards (prices)
for _ in range(episodes):
    for i in range(len(sizes) - 1):
        reward = prices[i]
        q_values[i] = q_values[i] + alpha * (reward + gamma * q_values[i + 1] - q_values[i])

# Predict price for a given flower size
def predict_price(size):
    if size in sizes:
        idx = np.where(sizes == size)[0][0]
        return prices[idx]
    else:
        return f"No data for size {size}."

# Test the prediction
sample_size = 2
predicted_price = predict_price(sample_size)
print(f"Predicted price for a flower size of {sample_size} cm is ${predicted_price}.")
