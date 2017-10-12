from bmm_sampler import bmm_sample
from bmm_sparse import BMM_Sparse,toSparse
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import random

def get_best(predictions):
  resp = []
  for pred in predictions:
    best = 0
    bval = 0
    for i in range(len(pred)):
      if bval < pred[i]:
        bval = pred[i]
        best = i
    resp.append(best)
  return resp


def accuracy(arr1, arr2):
  l = min(len(arr1),len(arr2))
  sum = 0.0
  for i in range(l):
     if arr1[i] == arr2[i]: sum+=1
  return sum/l


def init(classes, data, targets, non_classified, classified):
   for cl in classes:     
     li_uclass = []
     for i in range(len(targets)):
        if targets[i] == cl:
           li_uclass.append(data[i]) 

     randm = random.randint(0, len(li_uclass)-1)
     li_class = [li_uclass[randm]]
     del li_uclass[randm]   
     
     non_classified[cl] = li_uclass
     classified[cl] = li_class


def get_data(dmap):
   data = []
   targets =[]
   cindex = []
   for key, value in dmap.items():
      data.extend(value)
      targets.extend([key]*len(value)) 
      cindex.extend(range(len(value))) 
   return (data, targets, cindex)


def max_index(array):
   max_val = -1
   max_indx = 0
   for i in range(len(array)):
     if max_val < array[i]:
       max_val = array[i]
       max_indx = i
   return (max_indx,max_val)


def min_index(array):
   min_val = 100000
   min_indx = 0
   for i in range(len(array)):
     if min_val > array[i]:
       min_val = array[i]
       min_indx = i
   return (min_indx,min_val)

   


def al_run():
  sample = bmm_sample(0.6, 0.2, 5, [0.33, 0.33, 0.33], 90, 20)

  non_classified = {}
  classified = {}
  classes = set(sample.targets.train)
  init(classes, sample.train, sample.targets.train, non_classified, classified)

  resp = []
  num=3
  for k in range(47):
     (data, targets, cindex) = get_data(classified)
     clf = BernoulliNB()
     clf.fit(data, targets)
     (tdata, ttargets, cindex) = get_data(non_classified)
     pred_nb = clf.predict( tdata )
     acc = accuracy(ttargets,pred_nb)

     resp.append(acc)
     num+=1
   
     probs = clf.predict_proba(tdata)
     p_min = 1
     indx_min = -1
     for i in range(len(probs)):
        (indx, p) = max_index(probs[i])

        if p_min >= p:
           p_min = p           
           indx_min = i
     
     cat_indx = ttargets[indx_min]
     #print(cindex)
     #print(str(indx_min)+" "+str(cat_indx)+" "+str(cindex[indx_min])+" "+str(len(non_classified[cat_indx])))
     if len(non_classified[cat_indx]) > 0:
       classified[cat_indx].append( non_classified[cat_indx][cindex[indx_min]] )   
       del non_classified[cat_indx][cindex[indx_min]]

  return resp


'''
sample = bmm_sample(0.6, 0.2, 5, [0.33, 0.33, 0.33], 90, 20)
mtx = toSparse(sample.train)
N, D = mtx.get_shape()
bmm = BMM_Sparse(mtx, N , D , 3)

predictions = bmm.predictAll(mtx)
best = get_best(predictions)
print(best)
'''




'''
non_classified = {}
classified = {}
init(classes, sample.train, sample.targets.train, non_classified, classified)


num=3
for k in range(47):
   (data, targets, cindex) = get_data(classified)
   clf = BernoulliNB()
   clf.fit(data, targets)
   (tdata, ttargets, cindex) = get_data(non_classified)
   pred_nb = clf.predict( tdata )
   acc = accuracy(ttargets,pred_nb)
   print("Accuracy: "+str(acc))
   outfile.write(str(num)+" "+str(acc)+"\n")
   num+=1
   
   probs = clf.predict_proba(tdata)
   p_min = 1
   indx_min = -1
   for i in range(len(probs)):
      (indx, p) = max_index(probs[i])
      #print(str(ttargets[i])+" "+str(tdata[i]))
      #print(str(probs[i])+"\n")
      if p_min > p:
         p_min = p
         indx_min = indx
         #print(" -->"+str(p)+" "+str(indx_min))
 
   item = non_classified[indx_min][cindex[indx_min]]
     
   #print( "-->"+str(len(non_classified[indx_min])) )
   #print( str(ttargets[indx_min]) + " " + str(cindex[indx_min]) )

   classified[indx_min].append( non_classified[indx_min][cindex[indx_min]] )   
   del non_classified[indx_min][cindex[indx_min]]

   #print(classified)

outfile.close()
'''

def get_results(matrix):
  r = len(matrix[0])
  c = len(matrix)

  mins = []
  maxs = []
  avrg = []
  for i in range(r):
     sums=0.0
     tmin=100000;
     tmax=-1
     for j in range(c):
       sums+=matrix[j][i]
       if tmin > matrix[j][i]: tmin = matrix[j][i]
       if tmax < matrix[j][i]: tmax = matrix[j][i]

     mins.append(tmin)
     maxs.append(tmax)
     avrg.append(sums/c)

  return (avrg, maxs, mins) 

resp = []

for i in range(50):
  resp.append(al_run())  



(avrg, maxs, mins) = get_results(resp)
outfile = open('results.dat', 'w') 
for i in range(len(avrg)):
  outfile.write(str(i+3)+" "+str(avrg[i])+" "+str(mins[i])+" "+str(maxs[i])+"\n")	
outfile.close()


#for i in range( len(sample.targets.train) ):
#  print(sample.targets.train[i]+" ")


#print( sample.test )
#print( sample.targets.test )

