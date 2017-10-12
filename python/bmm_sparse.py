import random
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from time import time
import numpy as np

###############################################################################
# Class BMM sparse
#
###############################################################################
class BMM_Sparse:

  def __init__(self, sparse_matrix, N, D, K, verbose=False):
    self.data = sparse_matrix

    self.N = N 
    self.D = D
    self.K = K
    self.verbose = verbose

    # model parameters
    # random initialization
    self.z = [[0 for k in range(self.K)] for n in range(self.N)]
    self.pi = [1/self.K for k in range(self.K)]
    self.mu = [[random.random() for d in range(self.D)] for k in range(self.K)]    

    # update parameters
    t0 = time() 
    self.learn()
    print("Updated. Done in %0.3fs." % (time() - t0))

  # probability of x for the k component
  def logpk(self, data, row, k):
    resp=0
    for i in range(self.D):
      if data[row,i]==1: resp+=np.log(max(self.mu[k][i],0.000001) )
      else: resp+=np.log( max(1-self.mu[k][i],0.000001) )
    return resp+np.log(self.pi[k])


  # e-m algorithm to learn the parameters
  def learn(self, data=None):
    change = True
    niter=0

    while change:
      change = False

      # E step
      for n in range(self.N): 
        
        for k in range(self.K): 
          self.z[n][k] = self.logpk(self.data,n,k) 
        if self.verbose: print( self.z[n] )        
        sumz=logsumexp(self.z[n])
        for k in range(self.K):   
          self.z[n][k]-=sumz

      if self.verbose: print("\n\n Z:")
      if self.verbose: print(np.exp(self.z))

      # M step
      N_m = [0 for k in range(self.K)]
      z_x = [[0 for d in range(self.D)] for k in range(self.K)]
      newpi = [1/self.K for k in range(self.K)]
      newmu = [[1/self.D for d in range(self.D)] for k in range(self.K)]
      for k in range(self.K):
        
        for d in range(self.D):         
          tmp = [] 
          for n in range(self.N): 
            if self.data[n,d]!=0: tmp.append(self.z[n][k])
          z_x[k][d] = logsumexp(tmp)
        N_m[k] = logsumexp([self.z[i][k] for i in range(self.N)])
        for d in range(self.D):
          newmu[k][d] = np.exp(z_x[k][d]-N_m[k]) 
        newpi[k] = np.exp(N_m[k] - np.log(self.N))

      if self.verbose: print("\nN_m:")
      if self.verbose: print(np.exp(N_m))

      if self.verbose: print("\nz_x:")      
      if self.verbose: print(z_x)
  
      for k in range(self.K):
        if self.pi[k] != newpi[k]: 
          change=True
          self.pi[k] = newpi[k]
        for d in range(self.D):
          if self.mu[k][d] != newmu[k][d]:
            change=True
            self.mu[k][d] = newmu[k][d]
      niter+=1
    if self.verbose: print("Finished in "+str(niter)+" iterations")

  def printModel(self):
     print("Mu: "+str(self.mu))
     print("Pi: "+str(self.pi))

  def predict(self, data, row):
    resp = []
    for k in range(self.K):
      resp.append( np.exp(self.logpk(data, row, k)) )        
    return resp


  def predictAll(self, data):
     nrow, ncol = data.get_shape()
     resp = []
     for row in range(nrow):
      resp.append(self.predict(data, row))
     return resp
###############################################################################
# End class BMM Sparse


###############################################################################
#  toSparse
###############################################################################
def toSparse(dense):
  rtmp = []
  ctmp = []
  dtmp = []
  nrows = len(dense)
  ncols = len(dense[0])
  for r in range(nrows):
    for c in range(ncols):
      if dense[r][c] != 0:
        rtmp.append(r)
        ctmp.append(c)
        dtmp.append(1)
  row = np.array(rtmp)
  col = np.array(ctmp)
  data = np.array(dtmp)  
  mtx = csr_matrix((data, (row, col)), shape=(nrows, ncols))
  return mtx


def test(dense, k):
  spmat = toSparse(dense)
  bmm = BMM_Sparse(spmat, len(dense), len(dense[0]), k)
  bmm.printModel()
  for r in bmm.predictAll(spmat):
    print(r)
  print("\n") 

##
#dense1 = [[1,1,1,1,0,0,0,0,0,0],
#         [1,1,1,1,0,0,0,0,0,0],
#         [0,0,0,0,1,1,1,1,1,1],
#         [0,0,0,0,1,1,1,1,1,1]]

#test(dense1,2)

#dense2 = [[1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]]  


#print(spmat[0,4])

#test(dense2,2)


#print("\n\n\n")

