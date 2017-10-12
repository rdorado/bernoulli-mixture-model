import random

class BMM:

  def __init__(self, data, K):
    self.data =data

    self.N = len(data) 
    self.D = len(data[0])
    self.K = K
    
    # model parameters
    # random initialization
    self.z = [[0 for k in range(self.K)] for n in range(self.N)]
    self.pi = [1/self.K for k in range(self.K)]
    self.mu = [[random.random() for d in range(self.D)] for k in range(self.K)]    

    # update parameters
    self.learn()

  # probability of x for the k component
  def pk(self, x, k):
    resp=1
    for i in range(len(x)):
      if x[i]==1: resp*=self.mu[k][i]
      else: resp*=(1-self.mu[k][i])
    return resp*self.pi[k]

  # e-m algorithm to learn the parameters
  def learn(self, data=None):
    change = True
    niter=0
    while change:
      change = False
      # E step
      for n in range(self.N): 
        sumz=0
        for k in range(self.K): 
          self.z[n][k] = self.pk(self.data[n], k)         
          sumz+=self.z[n][k]
        for k in range(self.K):   
          self.z[n][k]/=sumz

      print("\nZ:")
      print(self.z)


      # M step
      N_m = [0 for k in range(self.K)]
      z_x = [[0 for d in range(self.D)] for k in range(self.K)]
      newpi = [1/self.K for k in range(self.K)]
      newmu = [[1/self.D for d in range(self.D)] for k in range(self.K)]
      for k in range(self.K):
        for n in range(self.N):
          N_m[k]+=self.z[n][k] 
          for d in range(self.D):
            z_x[k][d] += self.z[n][k]*self.data[n][d]
        for d in range(self.D):
          newmu[k][d] = z_x[k][d]/N_m[k] 
        newpi[k] = N_m[k]/self.N

      print("\nN_m:")
      print(N_m)

      print("\nz_x:")      
      print(z_x)

      for k in range(self.K):
        if self.pi[k] != newpi[k]: 
          change=True
          self.pi[k] = newpi[k]
        for d in range(self.D):
          if self.mu[k][d] != newmu[k][d]:
            change=True
            self.mu[k][d] = newmu[k][d]
      niter+=1
    print("Finished in "+str(niter)+" iterations")

  def printModel(self):
     print("Mu: "+str(self.mu))
     print("Pi: "+str(self.pi))

  def predict(self, x):
    resp = []
    for k in range(self.K):
      resp.append(self.pk(x, k))        
    return resp

  def predictAll(self, data):
     resp = []
     for dat in data:
       resp.append(self.predict(dat))
     return resp


data = [[1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,1],
        [0,0,0,0,1,1,1,1,1,1]]


bmm = BMM(data, 2)
bmm.printModel()
for r in bmm.predictAll(data):
  print(r)

print("\n\n\n")

