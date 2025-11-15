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
