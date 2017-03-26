import matplotlib.pyplot as pl
import numpy as np
import seaborn as sns
import pandas


import os
basedirn = "/home/james/tinygemmout/"

results = {}


df = pandas.DataFrame(columns=('gflops', 'ice', 'geom'))

index = 0
for dirn in [os.path.join(basedirn,"deepbench" + str(i)) for i in range(10)]:
  for fn in os.listdir(dirn):
    
    #if "ICE" not in fn:
      #continue
    
    if fn not in results.keys():
      results[fn] = []
      
    m = int(fn.split("_m")[1].split("_")[0])
    n = int(fn.split("_n")[1].split("_")[0])
    k = int(fn.split("_k")[1].split("_")[0])
    tA = int(fn.split("_tA")[1].split("_")[0])
    tB = int(fn.split("_tB")[1].split(".")[0])
    filly = open(os.path.join(dirn,fn), 'r')
    lines = filly.readlines()
    #print m,n,k,tA,tB,lines[-1], lines[-1].split()
    gflops = float(lines[-1].split()[-1])
    results[fn].append(gflops)
    
    print fn, "ICE" in fn
    
    df.loc[index] = [gflops, "ICE" in fn, fn.replace("C_ICE1", "")]
    
    index += 1


allgflops = [np.array(results[fn]) for fn in results.keys()]
print allgflops

#pl.violinplot(allgflops, showmeans = False, showmedians = True)
#pl.violinplot(allgflops, showmeans = True, showmedians = True)

#sns.violinplot(x="day", y="total_bill", data=tips)

sns.violinplot(x = "geom", y = "gflops", hue = "ice", data = df, palette = "Set2", split = True)
