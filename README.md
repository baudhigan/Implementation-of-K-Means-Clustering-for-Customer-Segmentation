# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Choose the number of clusters, K.

2. Initialize K centroids randomly.

3. Assign each data point to the nearest centroid.

4. Recalculate centroids as the mean of assigned points.

5. Repeat steps 3 and 4 until centroids stabilize (no major change).

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: BAUDHIGAN D
RegisterNumber: 212223230028

import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print("Sample Data:")
print(data.head())

X = data.iloc[:, [3, 4]].values  

wcss = [] 

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', color='purple')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    n_init=10,
    random_state=42
)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='yellow',
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.grid(True)
plt.show()

*/
```

## Output:


<img width="719" height="169" alt="Screenshot 2025-10-29 104957" src="https://github.com/user-attachments/assets/d6e667cb-d211-4780-b2be-37a12e86bb47" />


<img width="953" height="598" alt="image" src="https://github.com/user-attachments/assets/f02fa983-86e7-4fad-8ddf-af55c1a4b6fa" />


<img width="944" height="713" alt="image" src="https://github.com/user-attachments/assets/029c4c65-8947-4fa7-96e9-932367c2dd1c" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
