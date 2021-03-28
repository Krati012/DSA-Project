import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

'''Implementation of K-means algorithm using in-built function'''
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)          #Inbuilt Kmeans algorithm module in sklearn.
y_kmeans = kmeans.cluster_centers_  
#y_kmeans are the final centroids output by the inbuilt KMeans algorithm module of sklearn library.

'''Our implementation of K-means algorithm'''
#The following is our implementation of the K-means algorithm from scratch using the 6 steps 
#mentioned previously.

def err(y1, y2):   #error function finds distance between new centers and previous centers
    return np.sum(np.square(np.linalg.norm(y1-y2)))

def find_clusters(X, n_clusters, error, rseed = 2):
        #X is a (300,2) shaped array containing 300 samples distributed across the x-y coordinate system 
        #in 4 clusters. The aim is to identify these clusters and corresponding cluster centroids.
        #n_clusters is the number of clusters we divide the data into. Here n_clusters = 4.
        #rseed is the seed value for a pseudo random number generator.
        
        rng = np.random.RandomState(rseed)
        i = rng.permutation(X.shape[0])[:n_clusters]
        centers = X[i] #We randomly initialize the centroids

        while True: #The loop runs till the centroids converge
            labels = pairwise_distances_argmin(X, centers)
            #labels is the array such that the value of label of i-th data point, labels[i] is equal
            # to the number of the cluster that the i-th data point belongs to.
            new_centers = np.array([X[labels==i].mean(0) for i in range(n_clusters)])
            #New centroid for a given cluster is calculated by finding the arithmetic mean of the points
            #assigned to that cluster.
            
            #if the centroids converge then the algorithm ends
            if err(centers, new_centers)<error: 
                break
            centers = new_centers   
    
        return centers,labels   #return final coordinates of centroids of the clusters and label of each data point        

centers,labels = find_clusters(X,4, 0.0001) 

print(y_kmeans)  #The centroids for the clusters obtained from the inbuilt module for K-means in sklearn. 
print(centers)   #The centroids for the clusters obtained from the implementation of the k-means algorithm by us.

'''
The following is the output of the above code for the centroid coordinates by inbuilt 
K-means module and our implementation of K-means algorithm respectively:-

[[-1.37324398  7.75368871] 
 [ 0.94973532  4.41906906] 
 [-1.58438467  2.83081263]
 [ 1.98258281  0.86771314]]

[[ 0.94973532  4.41906906]
 [-1.37324398  7.75368871]
 [ 1.98258281  0.86771314]
 [-1.58438467  2.83081263]]
'''

#Plotting the results obtained
plt.scatter(X[:,0],X[:,1],c=labels, s = 50, cmap = 'viridis') 
#Plotting scatter plot for the sample population density with points colored according to the different clusters they have been assigned to.
plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5) 
#Plotting the centroids of the clusters obtained from the K-means algorithm
plt.show()

'''
 We can observe that the centroids obtained from the inbuilt K-means module by sklearn
 and the centroids obtained by our implementation of the K-means algorithm are the 
 same. 
 The centroid coordinates indicate the optimal positions for the Walmart stores. In the 
 following output scatter plot, the locations of centroids corresponding to the 
 different clusters are indicated.
 '''
