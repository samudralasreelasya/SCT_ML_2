# ==========================================
# Customer Segmentation using K-Means
# ==========================================

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans

# ==========================================
# Creating Required Folders
# ==========================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# ==========================================
# Loading Dataset
# ==========================================

df = pd.read_csv("Mall_Customers.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:\n")
print(df.head())

# ==========================================
# Dataset Information
# ==========================================

print("\nDataset Information:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# ==========================================
# Checking Missing Values
# ==========================================

print("\nMissing Values:\n")
print(df.isnull().sum())

# ==========================================
# Selecting Features for Clustering
# ==========================================

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ==========================================
# Visualizing Customer Distribution
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    data=df
)

plt.title('Customer Distribution')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

# Save Figure
plt.savefig("outputs/customer_distribution.png")

plt.show()

# ==========================================
# Finding Optimal Number of Clusters
# Using Elbow Method
# ==========================================

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# ==========================================
# Plotting Elbow Method Graph
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), wcss, marker='o')

plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

# Save Figure
plt.savefig("outputs/elbow_method.png")

plt.show()

# ==========================================
# Applying K-Means Clustering
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

# Predicting Clusters
y_kmeans = kmeans.fit_predict(X)

# ==========================================
# Adding Cluster Labels to Dataset
# ==========================================

df['Cluster'] = y_kmeans

print("\nDataset with Cluster Labels:\n")
print(df.head())

# Save clustered dataset
df.to_csv("clustered_customers.csv", index=False)

# ==========================================
# Cluster Analysis
# ==========================================

print("\nCluster Analysis:\n")

cluster_analysis = df.groupby('Cluster')[[
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]].mean()

print(cluster_analysis)

# ==========================================
# Visualizing Customer Segments
# ==========================================

plt.figure(figsize=(10, 7))

# Cluster 1
plt.scatter(
    X.iloc[y_kmeans == 0, 0],
    X.iloc[y_kmeans == 0, 1],
    s=100,
    c='red',
    label='Cluster 1'
)

# Cluster 2
plt.scatter(
    X.iloc[y_kmeans == 1, 0],
    X.iloc[y_kmeans == 1, 1],
    s=100,
    c='blue',
    label='Cluster 2'
)

# Cluster 3
plt.scatter(
    X.iloc[y_kmeans == 2, 0],
    X.iloc[y_kmeans == 2, 1],
    s=100,
    c='green',
    label='Cluster 3'
)

# Cluster 4
plt.scatter(
    X.iloc[y_kmeans == 3, 0],
    X.iloc[y_kmeans == 3, 1],
    s=100,
    c='cyan',
    label='Cluster 4'
)

# Cluster 5
plt.scatter(
    X.iloc[y_kmeans == 4, 0],
    X.iloc[y_kmeans == 4, 1],
    s=100,
    c='magenta',
    label='Cluster 5'
)

# Plotting Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='yellow',
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

# Save Figure
plt.savefig("outputs/customer_clusters.png")

plt.show()

# ==========================================
# Final Message
# ==========================================

print("\nK-Means Clustering Completed Successfully!")
print("Graphs saved in outputs folder.")
print("Clustered dataset saved as clustered_customers.csv")