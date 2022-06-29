from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

kk=3

X, y = make_blobs(n_samples=300,n_features=2,centers=kk,random_state=1)

print("X:\n", X[:10])
print("y:\n", y[:10])


from sklearn.model_selection import train_test_split
x_train , x_test ,y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(x_train[:10])
print(y_train[:10])

print(x_test[:10])
print(y_test[:10])

def distance(x,y):
  dis=(x-y)**2
  dis=np.sqrt(dis)
  dis=np.sum(dis)
  return dis

# import numpy as np

# K=3
# d=[]
# ids=[]
# for i_test in x_test:
#   for i_train in x_train:
#     d.append(distance(i_train,i_test))
#   ids.append(np.argsort(d))
#   d=[]

# idx2element=[]
# label=[]
# for ii in range(len(ids)):
#   for row in ids[ii]:
#     idx2element.append(y_train[row])
  
#   counts = np.bincount(idx2element[:K])  
#   label.append(np.argmax(counts))
#   idx2element=[]

import numpy as np

K=10   #2-10
d=[]
ids=[]
K_predict=[]
for i_test in x_test:
  for i_train in x_train:
    d.append(distance(i_train,i_test))
  ids.append(np.argsort(d))
  d=[]

for k in range(1,K):
  idx2element=[]
  predict=[]
  for ii in range(len(ids)):
    for row in ids[ii]:
      idx2element.append(y_train[row])
  
    counts = np.bincount(idx2element[:k])  
    predict.append(np.argmax(counts))
    idx2element=[]
  K_predict.append(np.array(predict))
print(K_predict)

from sklearn.metrics import classification_report
print(classification_report(y_test, K_predict[0]))
print(0)