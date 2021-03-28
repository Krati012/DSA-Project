import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


sns.set() #Setting plot style

#Generating the sample data. We generate data distributed in clusters about four centers
#such that the standard deviation for a set of points belonging to a cluster is 0.6. X is the
#sample data while y_true are the true centroids of the generated clusters.
X,y_true = make_blobs(n_samples=300, centers =4, cluster_std = 0.60, random_state = 0)

#Plotting the scatter plot for the customer population density.
plt.scatter(X[:,0], X[:,1], s = 50)
plt.show()
