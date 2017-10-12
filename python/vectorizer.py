from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from bmm_sparse import BMM_Sparse
from scipy.sparse import csr_matrix
import numpy as np
import sys,math

def binary_search(arr, num):
   arr1 = [0] + arr + [sys.maxsize]
   
   r=len(arr1)
   l=0   
   m=int(l+(r-l)/2)
   while r != l:
     m=math.ceil(l+(r-l)/2)
     if arr1[m-1] <= num and num <= arr1[m]: return (m-1)
     else:
       if num < arr1[m-1]: r = m
       else: l = m


print("Loading data...")

remove = ('headers', 'footers', 'quotes')
#categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
categories = ['alt.atheism']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=remove)
corpus = twenty_train.data[:30]


categories = ['comp.graphics']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=remove)
corpus.extend(twenty_train.data[:30])


print("Data loaded")

#from sklearn.feature_selection import SelectKBest, chi2
#ch2 = SelectKBest(chi2, k=1000)
#corpus = ch2.fit_transform(corpus)

'''
corpus = [
       'This is the first document.',
       'This is the second second document.',
       'And the third one.',
       'Is this the first document?',]
'''

#corpus = [
#       'Aaaaa Baaaa Caaaa Daaaa',
#       'Aaaaa Baaaa Caaaa Daaaa',
#       'Eaaaa Faaaa Gaaaa Haaaa Iaaaa Jaaaa',
#       'Eaaaa Faaaa Gaaaa Haaaa Iaaaa Jaaaa',]

'''
corpus = [
       'Aaaaa Caaaa Daaaa',
       'Aaaaa Baaaa Caaaa Daaaa',
       'Eaaaa Faaaa Haaaa Iaaaa Jaaaa',
       'Kaaaa',
       'Maaaa',
       'Eaaaa Aaaaa Faaaa Gaaaa Haaaa Iaaaa Jaaaa',]
'''

print("Removing features...")

vectorizer = CountVectorizer(min_df=1, binary=True)
X = vectorizer.fit_transform(corpus)

X_clone = X.tocsc()
X_clone.data = np.ones( X_clone.data.shape )
NumNonZeroElementsByColumn = X_clone.sum(0)
NumNonZeroElementsByRow = X_clone.sum(1)

low_threshold = 4
high_threshold = 10
arr = np.array(NumNonZeroElementsByColumn)[0]
ndocs, nfeats = X.get_shape()

allndocs = ndocs
print("Original dimmensions: "+str(ndocs) +" docs X "+str(nfeats) +" features")
print("Removing features...")

cfeat = 0
rtmp = []
ctmp = []
dtmp = []
featr = []
for i in range(nfeats):
  if arr[i] > low_threshold and arr[i] < high_threshold: 
    row = X.getcol(i)
    for ind in row.tocsc().indices:
      indx = binary_search(rtmp, ind)
       
      ctmp.insert(indx, cfeat)
      rtmp.insert(indx, ind)
      dtmp.append(1)
    cfeat+=1
  else: featr.append(i)

#print(rtmp)
#print(ctmp)

off=0
docrem = []
for i in range(len(rtmp)-1):
  tmp=rtmp[i]
  rtmp[i]-=off 
  if tmp < rtmp[i+1]-1: 
    #print(str(tmp)+" "+str(rtmp[i+1]))
    off+=(rtmp[i+1]-tmp-1)
    docrem.extend(range(tmp+1, rtmp[i+1]))

rtmp[len(rtmp)-1]-=off
ndocs=rtmp[len(rtmp)-1]+1

#print(rtmp)

row = np.array(rtmp)
col = np.array(ctmp)
data = np.array(dtmp)  

mtx = csr_matrix((data, (row, col)), shape=(ndocs, cfeat))

print("New dimmensions: "+str(ndocs) +" docs X "+str(cfeat) +" features")
print("Documents removed: "+str(docrem))


def getBests(predictions):
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

N, D = mtx.get_shape()
#print( str(N) + " " + str(D) )
bmm = BMM_Sparse(mtx, N , D , 2)
#bmm.printModel()
predictions = bmm.predictAll(mtx)
bestclusters = getBest(predictions)


j=0
printdocuments=False
for i in range(allndocs):
    if i in docrem:
      print("Document "+str(i)+": removed")      
    else:
      print("Document "+str(i)+": "+str(bestclusters[j])) 
      #print("Document "+str(i)+": "+str(predictions[j]))
      j+=1
    if printdocuments : print(corpus[i])
      
 


#print(X)
#print(type(X))


