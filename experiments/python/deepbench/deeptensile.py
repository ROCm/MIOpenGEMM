#DeepBench problem dimensions using Tensile. 
def get_tensile_floppage(nruns = 4):
  """
  tensout1.txt was obtained with Tensile.py benchmark --problems-path (path-to-GEMM-xml) --solutions-path (some-or-other-path) --backend opencl >> tensout1.txt
  tensout4.txt was obtained similarly (with tensout4.txt), but after setting control.benchmark to 4 (Thus average time over 4 runs).  
  will return a dictionary, keys are frags of the form m%s_n%s_k%s_tA%s_tB%s, values are gflop/s.
  """

  
  solns = {}
  
  if nruns == 1:
    filly = open("./data/tensout1.txt", "r")
  
  elif nruns == 4:
    filly = open("./data/tensout4.txt", "r")
   
  else:
    raise RuntimeError("nruns invalid, should be 1 or 4.")
    
  lines = filly.readlines()
  problems = []
  flops = []
  maxflops = 0.
  for l in lines:
    if "Status: Pro" in l:
      m = int(l.split("i:")[1].split(",")[0].split("]")[0].split(")")[0])
      n = int(l.split("j:")[1].split(",")[0].split("]")[0].split(")")[0])
      k = int(l.split("k:")[1].split(",")[0].split("]")[0].split(")")[0])
      tA = int("k,i" in l)
      tB = int("j,k" in l)
      problems.append({'tA':tA, 'tB':tB, 'm':m, 'n':n, 'k':k})
      if maxflops != 0:
        flops.append(maxflops)  
        maxflops = 0.
  
    elif "GFlop/s =" in l:
      maxflops = max(maxflops, float(l.split("GFlop/s = ")[1].split(";")[0]))
  
  flops.append(maxflops)
  
  for p,f in zip(problems, flops):
    solns["m%s_n%s_k%s_tA%s_tB%s"%(p['m'],p['n'],p['k'],p['tA'], p['tB'])] = f

  
  return solns
    
  
