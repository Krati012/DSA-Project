import matplotlib . pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

sns.set () # Setting plot style
#Generating the sample data. We generate data distributed in clusters about four centers
#such that the standard deviation for a set of points belonging to a cluster is 0.6. X is the
#sample data while y_true are the true centroids of the generated clusters.
X , y_true = make_blobs ( n_samples =300 , centers =4 , cluster_std = 0.60 , random_state = 0)

#Plotting the scatter plot for the customer population density.
plt.scatter ( X [: ,0] , X [: ,1] , s = 50)
plt.show()

#We know by the data we generated that the best value for \textit{k} is 4. We now run 
#the K-means algorithm for different values of k and plot the Elbow curve.

#Run the Kmeans algorithm and get the index of data points clusters
wcss = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X)
    wcss.append(km.inertia_) #km.inertia_ is the wcss value for a given k

# Plot wcss against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, wcss, '-o')
plt.xlabel(r'Number of clusters k', fontweight = 'bold')
plt.ylabel('Within-cluster sum of squares', fontweight = 'bold');
plt.show()
