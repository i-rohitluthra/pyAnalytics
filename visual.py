#clustering using iris dataset

#libraries

import nympy as np
import pandas as pd
import matplotlib.pyplot as plt

from pydataset import data #mtcars dataset is available in this data set
import seaborn as sns #enhances matplot
df = data('iris')
df.head()

df.Species.value_counts()

df1= df.select_dtypes(exclude=['object'])
df1.head()

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
df1_scaled = scalar.fit_transform(df1)

type(df1_scaled)
#data2_scaled.describe() #it converts to different format
pd.DataFrame(df1_scaled).describe()

#kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)  #hyper parameters

kmeans.fit(df1_scaled)
kmeans.inertia_  #sum of sq distances of samples to their centeroid
kmeans.cluster_centers_ #prints the cluster centers mean with the mean values for all parameters
kmeans.labels_ #informs which row has gone in which cluster
kmeans.n_iter_  #iterations to stabilise the clusters
kmeans.predict(df1_scaled)

clusterNos = kmeans.labels_
clusterNos
type(clusterNos)

df1.groupby([clusterNos]).mean()
#pd.options.display.max_columns =None
df.groupby(['Species']).mean()

data2_scaled[1:5]

data.columns
data2.columns
NCOLS = data2.columns 


#%%

#hierarchical clustering
import scipy.cluster.hierarchy as shc
dend = shc.dendrogram(shc.linkage(df1_scaled, method='ward'))
#data2_scaled
plt.figure(figsize = (10,7))
plt.title("Dendrogram")
dend = shc.dendrogram(shc.linkage(df1_scaled, method='ward'))
#plt.axhline(y=6, color='r', linestyle='--')
plt.show();

#another method for Hcluster from sklearn
from sklearn.cluster import AgglomerativeClustering
aggCluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
aggCluster.fit_predict(df1_scaled)
aggCluster
aggCluster.labels_

#compare
compare = pd.DataFrame({'kmCluster': kmeans.labels_, 'HCaggl': aggCluster.labels_, 'Diff': kmeans.labels_ - aggCluster.labels_})
compare
compare.Diff.sum()
compare.kmCluster.value_counts()
compare.HCaggl.value_counts()   

#aggregate value of the clusters
clusterNo2 = aggCluster.labels_
clusterNo2
type(clusterNo2)
data2
data2.groupby([clusterNo2]).mean()

data2.groupby([clusterNos]).mean()
#Customer Segmentation
#Product Segmentation
