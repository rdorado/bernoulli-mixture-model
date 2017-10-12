import numpy as np

class targets:
  def __init__(self):
     self.train = []
     self.test = []

class dataset:
  def __init__(self):
     self.train = []
     self.test = []
     self.targets = targets()

def bmm_sample_2(mu, pi, N):
  resp = data()
  sizes = np.random.multinomial(N, pi, size=1)[0]
  print( sizes )   
  for i in range(len(sizes)):
    n = len(mu[i])
    #print(n)
    for j in range(sizes[i]):
      print( np.random.binomial(np.ones(n, dtype=np.int8), mu[i]) )
  return resp

def bmm_sample(p1, p2,fpc, pi, Ntrain, Ntest):
  resp = dataset()
  nc = len(pi)
  strain = np.random.multinomial(Ntrain, pi, size=1)[0]
  stest = np.random.multinomial(Ntest, pi, size=1)[0]
  for i in range( nc ):
    probs = np.append( np.ones(i*fpc, dtype=np.int8)*p2, np.ones(fpc, dtype=np.int8)*p1 )
    probs = np.append( probs, np.ones((nc-1-i)*fpc, dtype=np.int8)*p2 )

    for j in range(strain[i]):
       resp.train.append( np.random.binomial(np.ones(fpc*nc, dtype=np.int8), probs).tolist() )
       resp.targets.train.append(i)

    for j in range(stest[i]):
       resp.test.append( np.random.binomial(np.ones(fpc*nc, dtype=np.int8), probs).tolist() )
       resp.targets.test.append(i)
    #print( probs )

  return resp

#sample = bmm_sample([[0.9,0.9,0,0],[0,0,0.9,0.9]],[0.2,0.8],8)
#sample = bmm_sample(0.8, 0.1, 5, [0.33, 0.33, 0.33], 100, 20)
#print( sample.train )
#print( sample.targets.train )
#print( sample.test )
#print( sample.targets.test )

