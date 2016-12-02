#// Backward Weights GEMM dimms in col major for Deepbench
#// tA, tB, M, N, K, lda, ldb, ldc

def first_version():
  
  inputdims =  """1, 0, 9, 16, 23040, 23040, 23040, 16 
  1, 0, 27, 64, 2916, 2916, 2916, 64 
  1, 0, 27, 64, 50176, 50176, 50176, 64 
  1, 0, 27, 64, 50176, 50176, 50176, 64 
  1, 0, 100, 32, 26939, 26939, 26939, 32
  1, 0, 144, 32, 5760, 5760, 5760, 32 
  1, 0, 147, 64, 12544, 12544, 12544, 64 
  1, 0, 192, 64, 784, 784, 784, 64 
  1, 0, 288, 64, 1440, 1440, 1440, 64 
  1, 0, 512, 192, 196, 196, 196, 192 
  1, 0, 576, 64, 2916, 2916, 2916, 64 
  1, 0, 576, 128, 360, 360, 360, 128 
  1, 0, 576, 128, 12544, 12544, 12544, 128
  1, 0, 832, 256, 49, 49, 49, 256 
  1, 0, 1152, 128, 729, 729, 729, 128 
  1, 0, 1152, 256, 196, 196, 196, 256 
  1, 0, 1152, 256, 3136, 3136, 3136, 256 
  1, 0, 1600, 32, 6308, 6308, 6308, 32 
  1, 0, 2304, 512, 49, 49, 49, 512 
  1, 0, 2304, 512, 784, 784, 784, 512 
  1, 0, 4608, 512, 196, 196, 196, 512 
  1, 0, 4608, 512, 49, 49, 49, 512 
  1, 0, 4800, 32, 784, 784, 784, 32 
  1, 0, 12800, 48, 196, 196, 196, 48 
  1, 0, 20800, 128, 49, 49, 49, 128 """
  
  
  print """/*  m , n , k , lda , ldb , ldc , tA , tB */ """
  for l in inputdims.split("\n"):
    tA, tB, M, N, K, lda, ldb, ldc = [x.replace(",", "") for x in l.split()]
    tA = "false"*(tA == "0") + "true"*(tA == "1")
    tB = "false"*(tB == "0") + "true"*(tB == "1")  
  
    print """std::make_tuple(%s, %s, %s, %s, %s, %s, %s, %s), """%(M, N, K, lda, ldb, M, tA, tB)
  


#def second_version():
stroo = """tC:0 tA:1 tB:0 colMaj:1 m:100 n:32 k:26939 lda:26939 ldb:26939 ldc:100 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:100 n:32 k:26939 lda:26939 ldb:26939 ldc:100 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:100 n:32 k:26939 lda:26939 ldb:26939 ldc:100 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:100 n:32 k:26939 lda:26939 ldb:26939 ldc:100 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:1600 n:32 k:6308 lda:6308 ldb:6308 ldc:1600 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:1600 n:32 k:6308 lda:6308 ldb:6308 ldc:1600 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:1600 n:32 k:6308 lda:6308 ldb:6308 ldc:1600 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:1600 n:32 k:6308 lda:6308 ldb:6308 ldc:1600 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:9 n:16 k:23040 lda:23040 ldb:23040 ldc:9 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:144 n:32 k:5760 lda:5760 ldb:5760 ldc:144 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:288 n:64 k:1440 lda:1440 ldb:1440 ldc:288 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:576 n:128 k:360 lda:360 ldb:360 ldc:576 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:27 n:64 k:2916 lda:2916 ldb:2916 ldc:27 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:576 n:64 k:2916 lda:2916 ldb:2916 ldc:576 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:1152 n:128 k:729 lda:729 ldb:729 ldc:1152 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:1152 n:256 k:196 lda:196 ldb:196 ldc:1152 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:2304 n:512 k:49 lda:49 ldb:49 ldc:2304 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:27 n:64 k:50176 lda:50176 ldb:50176 ldc:27 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:576 n:128 k:12544 lda:12544 ldb:12544 ldc:576 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:1152 n:256 k:3136 lda:3136 ldb:3136 ldc:1152 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:2304 n:512 k:784 lda:784 ldb:784 ldc:2304 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:4608 n:512 k:196 lda:196 ldb:196 ldc:4608 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:4608 n:512 k:49 lda:49 ldb:49 ldc:4608 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:27 n:64 k:50176 lda:50176 ldb:50176 ldc:27 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:576 n:128 k:12544 lda:12544 ldb:12544 ldc:576 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:1152 n:256 k:3136 lda:3136 ldb:3136 ldc:1152 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:2304 n:512 k:784 lda:784 ldb:784 ldc:2304 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:4608 n:512 k:196 lda:196 ldb:196 ldc:4608 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:4608 n:512 k:49 lda:49 ldb:49 ldc:4608 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:147 n:64 k:12544 lda:12544 ldb:12544 ldc:147 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:4800 n:32 k:784 lda:784 ldb:784 ldc:4800 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:192 n:64 k:784 lda:784 ldb:784 ldc:192 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:12800 n:48 k:196 lda:196 ldb:196 ldc:12800 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:512 n:192 k:196 lda:196 ldb:196 ldc:512 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:832 n:256 k:49 lda:49 ldb:49 ldc:832 a_offset:0 b_offset:0 c_offset:0
tC:0 tA:1 tB:0 colMaj:1 m:20800 n:128 k:49 lda:49 ldb:49 ldc:20800 a_offset:0 b_offset:0 c_offset:0"""
linesboo = stroo.split("\n")
lines = []
for l in linesboo:
  if l not in lines:
    lines.append(l)
    M = l.split("m:")[1].split(" ")[0]
    N = l.split("n:")[1].split(" ")[0]
    K = l.split("k:")[1].split(" ")[0]

    lda = l.split("lda:")[1].split(" ")[0]
    ldb = l.split("ldb:")[1].split(" ")[0]
    ldc = l.split("ldc:")[1].split(" ")[0]
    
    tA = l.split("tA:")[1].split(" ")[0]
    tB = l.split("tB:")[1].split(" ")[0]
    tA = "false"*(tA == "0") + "true"*(tA == "1")
    tB = "false"*(tB == "0") + "true"*(tB == "1")
    
    #print l
    print """std::make_tuple(%s, %s, %s, %s, %s, %s, %s, %s), """%(M, N, K, lda, ldb, M, tA, tB)
