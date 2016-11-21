
def get_tensile_floppage(nruns):
  solns = {}
  
  if nruns == 1:
    filly = open("tensout1.txt", "r")
  
  elif nruns == 4:
    filly = open("tensout4.txt", "r")
   
  else:
    raise RuntimeError("nruns invalid")  
  lines = filly.readlines()
  problems = []
  flops = []
  maxflops = 0.
  for l in lines:
    #print l
    if "Status: Pro" in l:
      m = int(l.split("i:")[1].split(",")[0].split("]")[0].split(")")[0])
      n = int(l.split("j:")[1].split(",")[0].split("]")[0].split(")")[0])
      k = int(l.split("k:")[1].split(",")[0].split("]")[0].split(")")[0])
      tA = int("k,i" in l)
      tB = int("j,k" in l)
      #print tA,tB,m,n,k
      problems.append({'tA':tA, 'tB':tB, 'm':m, 'n':n, 'k':k})
      if maxflops != 0:
	flops.append(maxflops)  
	maxflops = 0.
    elif "GFlop/s =" in l:
    #print l
      maxflops = max(maxflops, float(l.split("GFlop/s = ")[1].split(";")[0]))
  
  flops.append(maxflops)
  
  
  for p,f in zip(problems, flops):
    solns["m%s_n%s_k%s_tA%s_tB%s"%(p['m'],p['n'],p['k'],p['tA'], p['tB'])] = f
  
    #print solns
    
  return solns
    
  
