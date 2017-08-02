"""
How much accuracy is lost using Strassen? On N(0,1) iid matrices. 
"""

import numpy as np
import numpy.random as npr

d0 = 2048

def strassen(depth, A, B):
  """
  return AB, using strassen to recursion level `depth'
  """
  if depth == 0:
    return np.dot(A, B)
  else:
    d = A.shape[0]
    A11 = A[0:d/2, 0:d/2]
    A12 = A[0:d/2, d/2::]
    A21 = A[d/2::, 0:d/2]
    A22 = A[d/2::, d/2::]


    B11 = B[0:d/2, 0:d/2]
    B12 = B[0:d/2, d/2::]
    B21 = B[d/2::, 0:d/2]
    B22 = B[d/2::, d/2::]

    M1 = strassen(depth-1, A11 + A22, B11 + B22) 
    M2 = strassen(depth-1, A21 + A22, B11)
    M3 = strassen(depth-1, A11, B12 - B22)
    M4 = strassen(depth-1, A22, B21 - B11)
    M5 = strassen(depth-1, A11 + A12, B22)
    M6 = strassen(depth-1, A21 - A11, B11 + B12)
    M7 = strassen(depth-1, A12 - A22, B21 + B22)
    
    C = np.zeros((d, d), dtype = A.dtype)
    C[0:d/2, 0:d/2] = M1 + M4 - M5 + M7
    C[0:d/2, d/2::] = M3 + M5
    C[d/2::, 0:d/2] = M2 + M4
    C[d/2::, d/2::] = M1 - M2 + M3 + M6
    
    return C



Ad = -0.2 + npr.randn(d0,d0)
Bd = 0.1 + npr.randn(d0,d0)

Af = np.array(Ad, dtype = np.float32)
Bf = np.array(Bd, dtype = np.float32)

C0d = strassen(0, Ad, Bd)
C0d2 = np.dot(Ad, Bd)

C0f = strassen(0, Af, Bf)
err0 = np.abs(C0d - np.array(C0f, dtype = np.float64))
mu0 = err0.mean()
ma0 = err0.max()


print "log2(mean absolute difference) : ", np.log2(mu0)
print "log2(max  absolute difference) : ", np.log2(ma0)
print "Depth \t\tmean/mean0 \tmax/max0"

for depth in [0,1,2,3,4,5]:  
  Cxf = strassen(depth, Af, Bf)
  err1 = np.abs(C0d - np.array(Cxf, dtype = np.float64))
  print depth, "   :    ", err1.mean()/mu0, '\t', err1.max()/ma0
