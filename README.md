# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1.Choose the number of clusters (K): 
          Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

 2.Initialize cluster centroids: 
        Randomly select K data points from your dataset as the initial centroids of the clusters.

 3.Assign data points to clusters: 
      Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically  done using Euclidean distance, but other distance metrics can also be used.

  4.Update cluster centroids: 
      Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

  5.Repeat steps 3 and 4: 
      Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

  6.Evaluate the clustering results: 
      Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

  7.Select the best clustering solution: 
      If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements
      

## Program:

Program to implement the K Means Clustering for Customer Segmentation.

Developed by: Vinush.CV

RegisterNumber: 212222230176

```python
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Dataset-20230524.zip")
data

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="yellow",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="pink",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="purple",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```



## Output:

### DATA.HEAD():
![image](https://github.com/aldrinlijo04/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118544279/f557d020-af3f-476d-b65e-a3bc746e682c)

### DATA.info():
![image](https://github.com/aldrinlijo04/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118544279/d6db0d45-48c9-444d-88f6-f6d18ac243fd)

### NULL VALUES:
![image](https://github.com/aldrinlijo04/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118544279/abf77bbe-8f0e-4eb3-9faa-8ae346449a85)

### ELBOW GRAPH:
![image](https://github.com/aldrinlijo04/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118544279/b42c8ad6-86a0-43f8-8a4e-5bdfad0c8431)

### CLUSTER FORMATION:
![image](https://github.com/aldrinlijo04/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118544279/6620c4a7-3a3f-4a77-9093-1f381da5b32b)

### PREDICICTED VALUE:
![image](https://github.com/aldrinlijo04/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118544279/a7a2992e-c9c0-41c2-b962-c9575e49ff72)

### FINAL GRAPH(D/O):
![image](https://github.com/aldrinlijo04/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118544279/271e9d59-d48d-4f17-b09c-227bb466f397)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
