# Import libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('flower_data.csv')  # Replace with your file name
X = data[['Size', 'Price']]  # Use both Size and Price for clustering

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)  # Using 2 clusters for simplicity
data['Cluster'] = kmeans.fit_predict(X)

# Display cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# # Visualize the clusters
# plt.scatter(data['Size'], data['Price'], c=data['Cluster'], cmap='viridis', label='Data Points')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
#             color='red', marker='X', s=200, label='Cluster Centers')
# plt.title('K-Means Clustering (Size vs Price)')
# plt.xlabel('Size (cm)')
# plt.ylabel('Price ($)')
# plt.legend()
# plt.show()
