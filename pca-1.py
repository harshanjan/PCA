import pandas as pd
import numpy as np
df1 = pd.read_csv("C:/Users/user/Desktop/DATASETS/wine.csv") #loading the dataset
df1.describe() # gives min,max,IQR and std values
df1.info
df = df1.iloc[:,1:] #ignoring frst column as it doesnt have anything to do with calculation

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#normalizing the data
df_norm = scale(df)
pca = PCA(n_components=13)
pca_values = pca.fit_transform(df_norm)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

pca.components_ #weights

#cumilative variance

var1 = np.cumsum(np.round(var,decimals =4) * 100)
var1
# Variance plot for PCA components obtained 
plt.plot(var1,color = 'yellow')

#pca scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5","comp6","comp7","comp8","comp9","comp10","comp11","comp12"

final = pca_data.iloc[:,0:3]

##hirarchical clustering##
# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(final, method='single',metric= 'euclidean')

#dendrogram
plt.figure(figsize=(14,6));plt.title('h-clustering');plt.xlabel('index');plt.ylabel('distance')
sch.dendrogram(z);plt.show()


# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(final);plt.show() 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

final['cluster'] = cluster_labels # creating a new column and assigning it to new column 
final.describe()
final1 = final.iloc[:, [3,0,1,2]] #bring cluster column to 0th index
final1.head()

# Aggregate mean of each cluster
final.groupby(final1.cluster).mean()


## K-means clustering#

#scree plot or elbow curve 
TWSS = []
k = list(range(2, 8))

from sklearn.cluster import KMeans
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(final)

model.labels_ # getting the labels of clusters assigned to each row 
clusters = pd.Series(model.labels_)  # converting numpy array into pandas series object 
final['cluster'] = clusters # creating a  new column and assigning it to new column 


final2 = final.iloc[:,[3,0,1,2]]
final2.head()

final.groupby(final2.cluster).mean()
