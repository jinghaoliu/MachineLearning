from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

kk=7

X, y = make_blobs(n_samples=3000,n_features=2,centers=kk,random_state=1)

print("X:\n", X)
print("y:\n", y)
## Kmeans center
import random
import numpy as np

def random_center(k,X):
  centers = []

  for i in range(k):
    x = np.random.randint(np.min(X[:,0]),np.max(X[:,0]))
    y = np.random.randint(np.min(X[:,1]),np.max(X[:,1]))
    centers.append(np.array([x,y]))

  
  return centers

centers=random_center(kk,X)

def distance(x,y):
  dis=(x-y)**2
  dis=np.sqrt(dis)
  dis=np.sum(dis)
  return dis


while(True):

  category=[]

  for d in range(len(X)):
    dist=[]

    for c in range(len(centers)):
      dis=distance(X[d],centers[c])
      dist.append(dis)
      
    category.append(np.argmin(dist))

  varDict = locals()
  category=np.array(category)
  for c_idx_i in range(kk):
   
    idx=np.argwhere(category==c_idx_i).flatten() 

    varDict['cat_idx_'+str(c_idx_i)] = idx

  if (np.mean(X[cat_idx_0],axis=0)==centers[0]).all():
    break
  for n_c in range(len(centers)):
    M=np.mean(X[eval("cat_idx_"+str(n_c))],axis=0)
    if  (~np.isnan(M).all()):
      centers[n_c]=np.mean(X[eval("cat_idx_"+str(n_c))],axis=0)
  