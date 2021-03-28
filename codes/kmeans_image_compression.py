import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
import warnings; warnings.simplefilter('ignore')
from sklearn.cluster import MiniBatchKMeans
'''
Following is the function for K-Means algorithm
y is the data-set of the 273280 pixels, each pixel having 3 values(corresponding to RGB)
in the range (0,1) associated with it. Thus the shape of y is (3,273280).
cn is the the matrix corresponding to the RGB values of the intial K-centroids. The 
shape of cn is (3,K). For this case K = 16.
''' 
def kmean(y,cn): 
    f, r, p, sumt =[], [], [], 0
    #Taking null array p.
    #p[i] will indicate the number of points associated with the i-th centroid.
    
    p = [0 for j in range(0,k)]
    
    #c_new will contain the updated centroids.
    c_new=np.zeros((y.shape[0], k)) 

    #storing nearest centroid for each point in array r
    for i in range(0,y.shape[1]):
        f = [np.square(np.linalg.norm(y[:,i]-cn[:,j])) for j in range(0,k)]
        r.append(np.argmin(f))
    
    #changing each centroid to centroid of nearest points
    for i in range(0,y.shape[1]):
        c_new[:,r[i]]=c_new[:,r[i]] + y[:,i]
        p[r[i]]=p[r[i]]+1
    for i in range(0,k):
        if(p[i]!=0):
            c_new[:,i]=(c_new[:,i]/p[i])  #centroid is arithmetic mean of all points in cluster
   
    #checking for convergence condition
    for i in range(0,k):
        sumt=sumt+np.square(np.linalg.norm(c_new[:,i]-cn[:,i]))
    
    print(sumt-eps, '\n') #eps is the user set tolerance for convergence. 
    
    if(sumt < eps):
        print("enter\n")
        print(c_new) #The final centroids 
        ans = y
        for i in range(0,y.shape[1]):
            ans[:,i] = c_new[:,r[i]]
        #ans is the dataset of the 272380 pixels with their RGB values replaced by 
        #those of the centroid of the cluster they belong to.
        return ans 
    else:
        return kmean(y,c_new)

#Reading Input image
x = load_sample_image('flower.jpg')
ax = plt.axes(xticks=[],yticks=[])
ax.imshow(x);
ax.set_title('Original Image: 16 million colors space', size = 16)
plt.show()

#Tolerance for convergence
eps = 9*1e-5 

#Taking number of clusters as 16 for the given problem as we aim to generate 16 clusters
k = 16 
y = x.reshape(x.shape[2],x.shape[0]*x.shape[1])

#Initializing cn
cn = np.zeros((y.shape[0], k))
o = 0

#Dividing By 255 for standard input should be in between 0 to 1
y = np.array(y) / 255.0

#Initializing first Centroid as centroid of first N/k points , second
#Centroid as centroid of second interval of N/K points and so on 
for i in range(0,k):
    for j in range(o,o-1+int(y.shape[1]/k)):
        cn[:,i] = cn[:,i] + y[:,j]
    cn[:,i] = cn[:,i]/(int(y.shape[1]/k))
    o=o+int(y.shape[1]/k)

#calling Function for K-means
final = kmean(y,cn)
x_recolored = final.reshape(x.shape)
#x_recolored is the final modified array of shape
#(427 x 640 x 3) which we plot as the output image.

'''
Following is the implementation of a modified version of K-means known as Mini batch 
K-means. We implement this using an inbuilt module of sklearn library.
'''
y_2 = x/255.0
y_2 = y_2.reshape(x.shape[0]*x.shape[1],x.shape[2])
kmeans = MifunctionniBatchKMeans(16)
kmeans.fit(y_2)
new_colors = kmeans.cluster_centers_[kmeans.predict(y_2)]
china_recolored_= new_colors.reshape(x.shape)
'''
We now plot the three images - Original image, the basic K-means compressed image and 
the minibatch K-means compressed image.
'''
fig, ax = plt.subplots(3,1, figsize = (16,6), subplot_kw=dict(xticks=[],yticks=[]))
fig.subplots_adjust(wspace=0.5)
ax[0].imshow(x)
ax[0].set_title('Original Image: 16 million colors space', size = 16)
ax[1].imshow(x_recolored)
ax[1].set_title('Basic Kmeans compressed image: 16 colors space', size = 16)
ax[2].imshow(china_recolored_)
ax[2].set_title('MiniBatchKMeans compressed image: 16 colors space',size=16)
plt.show()
